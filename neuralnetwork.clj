(ns ^{:author "Vladimir Urosevic"}
  master-predikcija.neuralnetwork
  (:require [master-predikcija.data :refer :all]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]))

(defrecord Neuronetwork [
                         hidden-layers                       ;; vector of hidden layers
                         output-layer                        ;; output layer

                         tmp1                                ;; tmp1 vector
                         tmp2                                ;; tmp2 vector

                         temp-matrix                         ;; output for layers

                         temp-vector-o-gradients             ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-vector-o-gradients2            ;; matrix, row=1, output gradient, dim number of output neurons

                         temp-vector-vector-h-gradients      ;; output gradient, dim number of output neurons

                         temp-matrix-gradients               ;; gradients for hidden layers, vectors by layer
                         temp-vector-matrix-delta            ;; delta weights for layers
                         temp-prev-delta-vector-matrix-delta ;; previous delta vector matrix layers
                         temp-vector-matrix-delta-momentum   ;; delta weights for layers - momentum

])

(def max-dim 512)

(def unit-vector (dv (replicate max-dim 1)))
(def unit-matrix (dge max-dim max-dim (repeat 1)))

(defn prepare-unit-vector
  "preparing unit vector for other calculations"
  [n]
  (if (<= n max-dim)
    (subvector unit-vector 0 n)
    (dv [0])))

(defn random-number
  "random number in interval [0 .. 0.1]"
  []
  (rand 0.47))

(defn create-random-matrix
  "Initialize a layer"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (dge dim-y dim-x (repeatedly random-number))
    ))

(defn create-null-matrix
  "Initialize a layer"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (dge dim-y dim-x))
    )

(defn layer-output
  "generate output from layer"
  [input weights result o-func]
  (o-func (mm! 1.0 weights input 0.0 result)))

(defn dtanh!
  "calculate dtanh for vector or matrix"
  [y result]
  (if (matrix? y)
    (let [unit-mat (submatrix unit-matrix (mrows y) (ncols y))]
      (do (sqr! y result)
          (axpy! -1 unit-mat result)
          (scal! -1 result)))

    (let [unit-vec (subvector unit-vector 0 (dim y))]
      (do (sqr! y result)
          (axpy! -1 unit-vec result)
          (scal! -1 result)))
    )
  )

(defn create-network
  "create new neural network"
  [number-input-neurons vector-of-numbers-hidden-neurons number-output-neurons]
  (let [tmp1 (into (vector number-input-neurons) vector-of-numbers-hidden-neurons)
        tmp2 (conj vector-of-numbers-hidden-neurons number-output-neurons)
        temp-matrix (for [x tmp2]
                      (conj (#(create-null-matrix x 1))))
        hidden-layers (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                        (conj (#(create-random-matrix (first x) (second x)))))
        output-layer (create-random-matrix (first (last (map vector tmp1 tmp2)))
                                           (second (last (map vector tmp1 tmp2))))
        temp-vector-vector-h-gradients (vec (for [x tmp2] (dge x 1 (repeat x 0))))
        temp-vector-o-gradients  (dge number-output-neurons 1 (repeat number-output-neurons 0))
        temp-vector-o-gradients2 (dge number-output-neurons 1 (repeat number-output-neurons 0))

        temp-matrix-1 (dge number-output-neurons 1)
        temp-matrix-2 (dge number-output-neurons 1)
        temp-matrix-3 (dge number-output-neurons 1)
        temp-matrix-4 (dge number-output-neurons 1)
        ]


     (->Neuronetwork hidden-layers
                     output-layer
                     tmp1
                     tmp2
                     temp-matrix
                     temp-vector-o-gradients
                     temp-vector-o-gradients2
                     temp-vector-vector-h-gradients
                     temp-matrix-1
                     temp-matrix-2
                     temp-matrix-3
                     temp-matrix-4)
    )
  )

(defn feed-forward
  "feed forward propagation"
  [network input-mtx]
  (let [input-vec-dim (mrows input-mtx)
        net-input-dim (first (:tmp1 network))
        temp-matrix (:temp-matrix network)
        number-of-layers (count temp-matrix)
        ]
    (if (= input-vec-dim net-input-dim)
      (do
        ;; set o-gradients to zero
        (copy! (dge (last (:tmp2 network)) 1 (replicate (last (:tmp2 network)) 0)) (:temp-vector-o-gradients network))
        (copy! (dge (last (:tmp2 network)) 1 (replicate (last (:tmp2 network)) 0)) (:temp-vector-o-gradients2 network))


        (layer-output input-mtx (trans (nth (:hidden-layers network) 0)) (nth temp-matrix 0) tanh!)
        (doseq [y (range 0 (- number-of-layers 2))]
          (layer-output (nth temp-matrix y) (trans (nth (:hidden-layers network) (inc y))) (nth temp-matrix (inc y)) tanh!))
        (layer-output (nth temp-matrix (- number-of-layers 2)) (trans (:output-layer network)) (nth temp-matrix (- number-of-layers 1)) tanh!)
        (nth temp-matrix (dec number-of-layers)))
      (throw (Exception. (str "Input dimmensions is not correct")))
      )
    )
  )

(defn backpropagation
  "learn network with one input vector"
  [network inputmtx no targetmtx speed-learning]
  (let [hidden-layers (:hidden-layers network)
        output-layer (:output-layer network)
        layers (vec (concat (:hidden-layers network) (vector (:output-layer network))))
        temp-matrix (:temp-matrix network)
        temp-vector-o-gradients (:temp-vector-o-gradients network)
        temp-vector-o-gradients2 (:temp-vector-o-gradients2 network)
        temp-vector-vector-h-gradients (:temp-vector-vector-h-gradients network)
        input (submatrix inputmtx 0 no (first (:tmp1 network)) 1)
        target (submatrix targetmtx 0 no (last (:tmp2 network)) 1)
        ]
     (do
       (feed-forward network input)

       ;; calculate output gradients
       (axpy! -1 (last temp-matrix) temp-vector-o-gradients)
       (axpy! 1 target temp-vector-o-gradients)
       (dtanh! (last temp-matrix) temp-vector-o-gradients2)
       (mul! temp-vector-o-gradients2 temp-vector-o-gradients temp-vector-o-gradients)
       (copy! temp-vector-o-gradients (last temp-vector-vector-h-gradients))

       ;; calculate hidden gradients

       (for [x (range (- (count temp-matrix) 1) 0 -1)]
         (do
           ;; (mm! (nth layers x)
           ;;     (nth (:temp-vector-vector-h-gradients network) x)
           ;;     (nth (:temp-vector-vector-h-gradients network) (dec x)) )

           (mm! 1.0 (nth layers x)
                    (nth (:temp-vector-vector-h-gradients network) x)
                0.0 (nth (:temp-vector-vector-h-gradients network) (dec x)))

           (mul! (nth temp-matrix (dec x)) (nth (:temp-vector-vector-h-gradients network) (dec x)))
           )
           )
         )



     )
  )




;; -------------------------------------------------------------------------



