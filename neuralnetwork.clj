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
                         biases                              ;; biases of each neuron, dim number of output neurons

                         tmp1                                ;; tmp1 vector
                         tmp2                                ;; tmp2 vector

                         temp-matrix                         ;; output for layers

                         temp-vector-o-gradients             ;; matrix, row=1, output gradient, dim number of output neurons
                         temp-vector-o-gradients2            ;; matrix, row=1, output gradient, dim number of output neurons

                         temp-vector-vector-h-gradients      ;; output gradient, dim number of output neurons

                         temp-matrix-gradients               ;; gradients for hidden layers, vectors by layer
                         temp-vector-matrix-delta            ;; delta weights for layers
                         temp-vector-matrix-delta-biases     ;; delta biases for layers
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
  (rand 0.071))


(import '(java.util Random))
(def normals
  (let [r (Random.)]
    (take 10000 (repeatedly #(-> r .nextGaussian (* 0.3) (+ 1.0))))
    ;;(map #(/ % 10) (take 10000 (repeatedly #(-> r .nextGaussian (* 0.3) (+ 1.0)))))
    ))


(defn create-random-matrix-by-gaussian
  "Initialize a layer"
  [dim-y dim-x]
  (do
    (if (> dim-y max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    (if (> dim-x max-dim)
      (throw (Exception. (str "Error. Max number of neurons is " max-dim))))
    ;; (dge dim-y dim-x (repeatedly random-number))
    (dge dim-y dim-x (take (* dim-x dim-y) normals))
    ))

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
  [input weights biases result o-func]
  (do
    (mm! 1.0 weights input 0.0 result)
     (for [x (range (count (cols result)))]
          (axpy! biases (cols result x))
       )
    (o-func result)
    )
 )

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
        biases (vec (for [x tmp2] (dge x 1 (repeat x 1))))
        temp-vector-vector-h-gradients (vec (for [x tmp2] (dge x 1 (repeat x 0))))
        temp-vector-o-gradients  (dge number-output-neurons 1 (repeat number-output-neurons 0))
        temp-vector-o-gradients2 (dge number-output-neurons 1 (repeat number-output-neurons 0))

        temp-matrix-1 (dge number-output-neurons 1)
        temp-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                (conj (#(create-null-matrix (first x) (second x)))))
                                              (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                            (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-biases (vec (for [x tmp2] (dge x 1 (repeat x 0))))
        temp-prev-delta-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                           (conj (#(create-null-matrix (first x) (second x)))))
                                                         (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                     (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-momentum (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                         (conj (#(create-null-matrix (first x) (second x)))))
                                                       (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                   (second (last (map vector tmp1 tmp2)))))))
        ]
     (->Neuronetwork hidden-layers
                     output-layer
                     biases
                     tmp1
                     tmp2
                     temp-matrix
                     temp-vector-o-gradients
                     temp-vector-o-gradients2
                     temp-vector-vector-h-gradients
                     temp-matrix-1
                     temp-vector-matrix-delta
                     temp-vector-matrix-delta-biases
                     temp-prev-delta-vector-matrix-delta
                     temp-prev-delta-vector-matrix-delta)
    )
  )

(defn create-network-gaussian
  "create new neural network"
  [number-input-neurons vector-of-numbers-hidden-neurons number-output-neurons]
  (let [tmp1 (into (vector number-input-neurons) vector-of-numbers-hidden-neurons)
        tmp2 (conj vector-of-numbers-hidden-neurons number-output-neurons)
        temp-matrix (for [x tmp2]
                      (conj (#(create-null-matrix x 1))))
        hidden-layers (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                        (conj (#(create-random-matrix-by-gaussian (first x) (second x)))))
        output-layer (create-random-matrix-by-gaussian (first (last (map vector tmp1 tmp2)))
                                           (second (last (map vector tmp1 tmp2))))
        biases (vec (for [x tmp2] (dge x 1 (repeat x 1))))
        temp-vector-vector-h-gradients (vec (for [x tmp2] (dge x 1 (repeat x 0))))
        temp-vector-o-gradients  (dge number-output-neurons 1 (repeat number-output-neurons 0))
        temp-vector-o-gradients2 (dge number-output-neurons 1 (repeat number-output-neurons 0))

        temp-matrix-1 (dge number-output-neurons 1)
        temp-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                (conj (#(create-null-matrix (first x) (second x)))))
                                              (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                          (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-biases (vec (for [x tmp2] (dge x 1 (repeat x 0))))
        temp-prev-delta-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                           (conj (#(create-null-matrix (first x) (second x)))))
                                                         (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                     (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-momentum (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                         (conj (#(create-null-matrix (first x) (second x)))))
                                                       (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                   (second (last (map vector tmp1 tmp2)))))))
        ]
    (->Neuronetwork hidden-layers
                    output-layer
                    biases
                    tmp1
                    tmp2
                    temp-matrix
                    temp-vector-o-gradients
                    temp-vector-o-gradients2
                    temp-vector-vector-h-gradients
                    temp-matrix-1
                    temp-vector-matrix-delta
                    temp-vector-matrix-delta-biases
                    temp-prev-delta-vector-matrix-delta
                    temp-prev-delta-vector-matrix-delta)
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


        (layer-output input-mtx (trans (nth (:hidden-layers network) 0)) (nth (:biases network) 0)  (nth temp-matrix 0) tanh!)
        (doseq [y (range 0 (- number-of-layers 2))]
          (layer-output (nth temp-matrix y) (trans (nth (:hidden-layers network) (inc y))) (nth (:biases network) (inc y)) (nth temp-matrix (inc y)) tanh!))

        (layer-output (nth temp-matrix (- number-of-layers 2)) (trans (:output-layer network)) (last (:biases network)) (nth temp-matrix (- number-of-layers 1)) tanh!)
        (nth temp-matrix (dec number-of-layers)))
      (throw (Exception. (str "Input dimmensions is not correct")))
      )
    )
  )

(defn feed-forward-all
  "feed forward propagation"
  [network input-mtx]
  (let [input-vec-dim (mrows input-mtx)
        net-input-dim (first (:tmp1 network))
        tmp2 (:tmp2 network)
        temp-matrix (for [x tmp2]
                      (conj (#(create-null-matrix x (ncols input-mtx)))))
        number-of-layers (count temp-matrix)
        ]
    (if (= input-vec-dim net-input-dim)
      (do

        (layer-output input-mtx (trans (nth (:hidden-layers network) 0)) (nth (:biases network) 0) (nth temp-matrix 0) tanh!)
        (doseq [y (range 0 (- number-of-layers 2))]
          (layer-output (nth temp-matrix y) (trans (nth (:hidden-layers network) (inc y))) (nth (:biases network) (inc y)) (nth temp-matrix (inc y)) tanh!))
        (layer-output (nth temp-matrix (- number-of-layers 2)) (trans (:output-layer network)) (last (:biases network)) (nth temp-matrix (- number-of-layers 1)) tanh!)
        (nth temp-matrix (dec number-of-layers)))
      (throw (Exception. (str "Input dimmensions is not correct")))
      )
    )
  )

(defn copy-matrix-delta
  "save delta matrix for momentum"
  [network]
  (let [delta-matrix (:temp-vector-matrix-delta network)
        prev-delta-matrix (:temp-prev-delta-vector-matrix-delta network)
        layers-count (count delta-matrix)]
    (for [x (range layers-count)]
      (copy! (nth delta-matrix x) (nth prev-delta-matrix x)))
    )
  )

(defn backpropagation
  "learn network with one input vector"
  [network inputmtx no targetmtx speed-learning alpha]
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

      (if (not (= alpha 0))
        (copy-matrix-delta network)
        )

      ;; calculate output gradients
      (axpy! -1 (last temp-matrix) temp-vector-o-gradients)
      (axpy! 1 target temp-vector-o-gradients)
      (dtanh! (last temp-matrix) temp-vector-o-gradients2)
      (mul! temp-vector-o-gradients2 temp-vector-o-gradients temp-vector-o-gradients)
      (copy! temp-vector-o-gradients (last temp-vector-vector-h-gradients))

      ;; calculate hidden gradients

      (doseq [x (range (- (count temp-matrix) 1) 0 -1)]
        (do
          (mm! 1.0 (nth layers x)
               (nth (:temp-vector-vector-h-gradients network) x)
               0.0 (nth (:temp-vector-vector-h-gradients network) (dec x)))
          (mul! (nth temp-matrix (dec x)) (nth (:temp-vector-vector-h-gradients network) (dec x)))
          ))

      ;; calculate delta for weights
      (doseq [row_o (range (- (count (conj (:temp-matrix network) input)) 2) -1 -1)]
        (let [layer-out-vector (col (nth (conj (:temp-matrix network) input) row_o) 0)
              cols-num (ncols (nth (:temp-vector-matrix-delta network) row_o))]
          (doseq [x (range cols-num)]
            (axpy! speed-learning layer-out-vector
                   (col (nth (:temp-vector-matrix-delta network) row_o) x))
            )))

      (doseq [layer-grad (range (count (:temp-vector-vector-h-gradients network)))]
        (let []
          (doseq [x (range (mrows (nth (:temp-vector-vector-h-gradients network) layer-grad)))]
            (scal! (entry (row (nth (:temp-vector-vector-h-gradients network) layer-grad) x) 0)
                   (col (nth (:temp-vector-matrix-delta network) layer-grad) x)
                   )
            )
          )

        (axpy! (nth (:temp-vector-matrix-delta network) layer-grad)
               (nth layers layer-grad))

        ;; update biases
        (mul! (nth (:temp-vector-vector-h-gradients network) layer-grad)
              (nth (:biases network) layer-grad)
              (nth (:temp-vector-matrix-delta-biases network) layer-grad)
        )

        (scal! speed-learning (nth (:temp-vector-matrix-delta-biases network) layer-grad))
        (axpy! (nth (:temp-vector-matrix-delta-biases network) layer-grad)
               (nth (:biases network) layer-grad))

        ;; momentum, if alpha <> 0
        (if (not (= alpha 0))
          (axpy! alpha (nth (:temp-prev-delta-vector-matrix-delta network) layer-grad)
               (nth layers layer-grad))
         )
        )


      )
     )
  )

(defn predict
  "feed forward propagation - prediction consumptions for input matrix"
  [network input-mtx]
  (let [net-input-dim  (mrows (first (:hidden-layers network)))
        input-vec-dim  (mrows input-mtx)
        input-vec-rows (ncols input-mtx)]

    (if (= net-input-dim input-vec-dim)
        (feed-forward-all network input-mtx)
      (throw (Exception.
               (str "Error. Input dimensions is not correct. Expected dimension is: " net-input-dim)))
      ))
   )



(defn evaluate
  "evaluation - detail view"
  [output-mtx target-mtx]
  (let [num (ncols output-mtx)]
    (for [i (range num)]
      {:output      (entry output-mtx 0 i)
       :target      (entry target-mtx 0 i)
       :percent-abs (Math/abs (* (/ (- (entry output-mtx 0 i) (entry target-mtx 0 i)) (entry target-mtx 0 i)) 100))}
      )))

(defn evaluate-abs
  "evaluation neural network - average report by absolute deviations"
  [input-mtx target-mtx]
  (let [u (count (map :percent-abs (evaluate input-mtx target-mtx)))
        s (reduce + (map :percent-abs (evaluate input-mtx target-mtx)))]
    (/ s u)))


(defn learning-decay-rate
  "Calculate learninig decay rate for epoch"
  [speed-learning decay-rate epoch-no]
  (/ decay-rate (+ 1 (* decay-rate epoch-no)))
  )

(defn train-network
  "train network with input/target vectors"
  [network input-mtx target-mtx iteration-count speed-learning alpha]
  (let [line-count (dec (ncols input-mtx))]
    (str
      (doseq [y (range iteration-count)]
        (doseq [x (range line-count)]
           (backpropagation network input-mtx x target-mtx speed-learning alpha)
        )
        (let [os (mod y 30)]
          (if (= os 0)
            (let [mape-value (evaluate-abs (predict network input_test_matrix2) target_test_matrix2)
                  mape-valueIN (evaluate-abs (predict network input_matrix2) target_matrix2)
                  ]
              (do
                (println y ": " mape-value)
                (println y ": " mape-valueIN)
                (println "---------------------")
                (write-file "konvg_test_20180912.csv" (str y "," mape-value "\n")))
              )


            )
          )
        ))))

(defn train-network-with-learning-decay-rate
  "train network with input/target vectors"
  [network input-mtx target-mtx iteration-count speed-learning alpha decay-rate]
  (let [line-count (dec (ncols input-mtx))]
    (str
      (doseq [y (range iteration-count)]
        (doseq [x (range line-count)]
          (backpropagation network input-mtx x target-mtx (learning-decay-rate speed-learning decay-rate y) alpha)
          )
        (let [os (mod y 100)]
          (if (= os 0)
            (let [mape-value (evaluate-abs (predict network input_test_matrix2) target_test_matrix2)]
              (do
                (println y ": " mape-value)
                (write-file "konvg_test_20180912_learnig_decay_rate_002.csv" (str y "," mape-value "\n")))
              )


            )
          )
        ))))

(defn get-network-config
  "get network config from file file"
  [filename]
  (let [c-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "CONFIGURATION")
        l-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "LAYERS")]
    (map read-string (get (vec (map #(string/split % #",")
                          (take 1 (nthnext
                                    (string/split
                                      (slurp (str "resources/" filename)) #"\n") (inc c-index)))))0))
    )
)

(defn load-network-configuration-biases
  "get a output part of data from file"
  [filename x]
  (let [o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "BIAS," (inc x)))
        e-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "BIAS," (+ x 2)))
        e-index2 (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "LAYERS")
        ]

    (if (= e-index -1)
      (map #(string/split % #",")
           (take (dec (- e-index2 o-index))
                 (nthnext
                   (string/split
                     (slurp (str "resources/" filename)) #"\n") (inc o-index))))
      (map #(string/split % #",")
           (take (dec (- e-index o-index))
                 (nthnext
                   (string/split
                     (slurp (str "resources/" filename)) #"\n") (inc o-index))))
      )
    ))

(defn load-network-configuration-hidden-layer
  "get a output part of data from file"
  [filename x]
  (let [o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "LAYER," (inc x)))
        e-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") (str "LAYER," (+ x 2)))
        e-index2 (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "OUTPUT")
        ]

      (if (= e-index -1)
        (map #(string/split % #",")
             (take (dec (- e-index2 o-index))
                   (nthnext
                     (string/split
                       (slurp (str "resources/" filename)) #"\n") (inc o-index))))
        (map #(string/split % #",")
             (take (dec (- e-index o-index))
                   (nthnext
                     (string/split
                       (slurp (str "resources/" filename)) #"\n") (inc o-index))))
      )
 ))


(defn load-network-configuration-output-layer
  "get a output part of data from file"
  [filename]
  (let [o-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "OUTPUT")
        e-index (.indexOf (string/split (slurp (str "resources/" filename)) #"\n") "END")]
    (do
      (map #(string/split % #",")
           (take (dec (- e-index o-index))
                 (nthnext
                   (string/split
                     (slurp (str "resources/" filename)) #"\n") (inc o-index)))))))

(defn create-network-from-file
  "create new neural network and load state from file"
  [filename]
  (let [layers-count (count (vec (get-network-config filename)))
        tmp1 (take (dec layers-count) (vec (get-network-config filename)))
        tmp2 (drop 1 (vec (get-network-config filename)))
        temp-matrix (for [x tmp2]
                      (conj (#(create-null-matrix x 1))))
        number-output-neurons (last tmp2)
        hidden-layers (let [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                        (for [y (range (dec (count (map vector tmp1 tmp2))))]
                          (conj
                            (trans (dge (second (nth x y)) (first (nth x y)) (reduce into [] (map #(map parse-float %)
                                                                                  (load-network-configuration-hidden-layer filename y))))
                                   )
                            ))
                        )
        o-layer-conf (load-network-configuration-output-layer filename)
        output-layer (dge (last tmp1) (last tmp2) (reduce into [] (map #(map parse-float %) o-layer-conf)))
        biases (vec (for [x (range (count tmp2))] (dge (nth tmp2 x) 1 (reduce into [] (map #(map parse-float %)
                                                                                           (load-network-configuration-biases filename x))))))
        temp-vector-vector-h-gradients (vec (for [x tmp2] (dge x 1 (repeat x 0))))
        temp-vector-o-gradients  (dge number-output-neurons 1 (repeat number-output-neurons 0))
        temp-vector-o-gradients2 (dge number-output-neurons 1 (repeat number-output-neurons 0))
        temp-matrix-1 (dge number-output-neurons 1)
        temp-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                (conj (#(create-null-matrix (first x) (second x)))))
                                              (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                          (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-biases (vec (for [x tmp2] (dge x 1 (repeat x 0))))
        temp-prev-delta-vector-matrix-delta (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                           (conj (#(create-null-matrix (first x) (second x)))))
                                                         (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                     (second (last (map vector tmp1 tmp2)))))))
        temp-vector-matrix-delta-momentum (vec (concat (for [x (take (dec (count (map vector tmp1 tmp2))) (map vector tmp1 tmp2))]
                                                         (conj (#(create-null-matrix (first x) (second x)))))
                                                       (vector (create-null-matrix (first (last (map vector tmp1 tmp2)))
                                                                                   (second (last (map vector tmp1 tmp2)))))))
        ]


    (->Neuronetwork hidden-layers
                    output-layer
                    biases
                    tmp1
                    tmp2
                    temp-matrix
                    temp-vector-o-gradients
                    temp-vector-o-gradients2
                    temp-vector-vector-h-gradients
                    temp-matrix-1
                    temp-vector-matrix-delta
                    temp-vector-matrix-delta-biases
                    temp-prev-delta-vector-matrix-delta
                    temp-prev-delta-vector-matrix-delta)
    )
  )

(defn xavier-initialization-update
  [network]

  (let [layer-neurons (map vector (:tmp1 network) (:tmp2 network))

       ]
    (do
      ;; prepare weights for hidden layers
      (doseq [x (range (dec (count layer-neurons))) ]
        (scal! (Math/sqrt (/ 2 (+ (first (nth layer-neurons x)) (second (nth layer-neurons x)))))
               (nth (:hidden-layers network) x)
        )
      )

      ;; prepare weights for output layer
      (scal! (Math/sqrt (/ 2 (+ (first (last layer-neurons)) (second (last layer-neurons)))))
             (:output-layer network))
    )
  )

  )

;; -------------------------------------------------------------------------



