(ns ^{:author "Vladimir Urosevic"}
  master-predikcija.example
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.linalg :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]
            [master-predikcija.data :refer :all]
            [master-predikcija.neuralnetwork :refer :all]
            ))

(create-network 2 [4 4] 3)

(def mreza3 (create-network 50 [64] 1))
(feed-forward mreza3 (submatrix input_matrix2 0 1 50 1))
(backpropagation mreza3 (submatrix input_matrix2 0 0 50 1) 0
                 (submatrix target_matrix2 0 0 1 1) 1)

(entry (row (backpropagation mreza3 (submatrix input_matrix2 0 0 50 1) 0
                             (submatrix target_matrix2 0 0 1 1) 1) 0) 0)



(def mreza5 (create-network 3 [4 8 8 4] 2))

(def mreza4 (create-network 3 [4 4] 2))
(def input4 (dge 3 1 [0.5 0.5 0.7]))
(def target4 (dge 2 1 [0.8 0.2]))

(feed-forward mreza4 input4)
(entry (col (feed-forward mreza4 input4) 0) 0)

(backpropagation mreza4 input4 0 target4 1)
(entry (col (backpropagation mreza4 input4 0 target4 1) 0) 1)

;; hidden output gradient
(mul (second (:temp-matrix mreza4)) (mm (:output-layer mreza4) (:temp-vector-o-gradients mreza4)))

;; jedinicna matrica
(dtr 3 (range 1) {:diag :unit})

;; racunanje delte

(col (nth (:temp-vector-matrix-delta mreza4) 2) 0)
(col (nth (:temp-matrix mreza4) 1) 0)

;; priprema kopiranje ulaza iz sloja pre
;; ubaciti skalar alpha - brzina ucenja
(axpy! (col (nth (:temp-matrix mreza4) 1) 0)
       (col (nth (:temp-vector-matrix-delta mreza4) 2) 1))

;; posle pomnoziti sa gradijentom






