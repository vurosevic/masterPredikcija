(ns ^{:author "Vladimir Urosevic"}
  master-predikcija.example
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]
            [uncomplicate.neanderthal.linalg :refer :all]
            [criterium.core :refer :all]
            [clojure.string :as string]
            [clojure.core :as core]
            [master-predikcija.data :refer :all]
            [master-predikcija.neuralnetwork :refer :all]
            ))

(create-network 2 [4 4] 3)

(def mreza3 (create-network 50 [64] 1))

(def mreza3 (atom (create-network 50 [64 64] 1)))
(feed-forward @mreza3 (submatrix input_matrix2 0 1 50 1))

(feed-forward mreza3 (submatrix input_matrix2 0 1 50 1))
(backpropagation mreza3 (submatrix input_matrix2 0 0 50 1) 0
                 (submatrix target_matrix2 0 0 1 1) 1)

(submatrix target_matrix2 0 0 1 1)

(str (for [x (range 12000)]
                (backpropagation @mreza3 (submatrix input_matrix2 0 0 50 1) 0
                          (submatrix target_matrix2 0 0 1 1) 0.1)))

(entry (row (backpropagation mreza3 (submatrix input_matrix2 0 0 50 1) 0
                             (submatrix target_matrix2 0 0 1 1) 1) 0) 0)



(def mreza5 (create-network 3 [8 8 8 8] 2))

(def mreza4 (create-network 3 [4 4] 2))
(def input4 (dge 3 1 [0.5 0.5 0.7]))
(def target4 (dge 2 1 [0.8 0.2]))

(feed-forward mreza4 input4)
(entry (col (feed-forward mreza4 input4) 0) 0)

(backpropagation mreza4 input4 0 target4 1)


(feed-forward mreza5 input4)
(str (for [x (range 100000)]
       (constantly
         (backpropagation mreza5 input4 0 target4 0.1))))


(def ignore (constantly nil))

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

(def mreza3 (atom (create-network 50 [64 64] 1)))
(feed-forward @mreza3 (submatrix input_matrix2 0 1 50 1))

(train-network @mreza3 (submatrix input_matrix2 0 0 50 2)
               (submatrix target_matrix2 0 0 1 2) 50000 0.1)

(train-network @mreza3 input_matrix2
               target_matrix2 14000 0.000015)

;; evaluate result
(evaluate (predict @mreza3 input_test_matrix2) target_test_matrix2)



;; average error
(evaluate-abs (predict @mreza3 input_test_matrix2) target_test_matrix2)

(evaluate (predict @mreza3 input_matrix2) target_matrix2)
(evaluate-abs (predict @mreza3 input_matrix2) target_matrix2)


;; test 2 sloja
(def mreza4 (atom (create-network 50 [64] 1)))
(feed-forward @mreza4 (submatrix input_matrix2 0 1 50 1))
(evaluate (predict @mreza4 input_matrix2) target_matrix2)
(evaluate (predict @mreza4 input_test_matrix2) target_test_matrix2)
(evaluate-abs (predict @mreza4 input_matrix2) target_matrix2)
(evaluate-abs (predict @mreza4 input_test_matrix2) target_test_matrix2)

(train-network @mreza4 input_matrix2
               target_matrix2 2000 0.00005)

(time (train-network @mreza4 input_matrix2
                     target_matrix2 10000 0.0015 0.3))

(save-network-to-file @mreza4 "nn_01.csv")

(def mreza20 (atom (create-network-from-file "nn_01.csv")))
(feed-forward @mreza20 (submatrix input_matrix2 0 1 50 1))
(evaluate-abs (predict @mreza20 input_test_matrix2) target_test_matrix2)

(time (train-network @mreza20 input_matrix2
                     target_matrix2 5000 0.0015 0.003))


;; test

(def mreza5 (atom (create-network-gaussian 50 [64 64 64 64 64 64] 1)))
(def mreza5 (atom (create-network 50 [64 64 64 64 64 64] 1)))
(def mreza5 (atom (create-network 50 [64 64 64] 1)))

(def mreza5 (atom (create-network-gaussian 62 [64 64 64] 1)))
(def mreza5 (atom (create-network-gaussian 62 [128 128] 1)))
(def mreza5 (atom (create-network 62 [128 128] 1)))

(def mreza5 (atom (create-network 50 [64 64] 1)))
(feed-forward @mreza5 (submatrix input_matrix2 0 1 50 1))
(evaluate (predict @mreza5 input_matrix2) target_matrix2)
(evaluate (predict @mreza5 input_test_matrix2) target_test_matrix2)
(evaluate-abs (predict @mreza5 input_matrix2) target_matrix2)
(evaluate-abs (predict @mreza5 input_test_matrix2) target_test_matrix2)

(evaluate (predict @mreza5 (:normalized-matrix norm-input-730)) (:normalized-matrix norm-target-730))
(evaluate-abs (predict @mreza5 (:normalized-matrix norm-input-730)) (:normalized-matrix norm-target-730))

(xavier-initialization-update @mreza5)

(save-network-to-file @mreza5 "nn_64_64_162b.csv")
(save-network-to-file @mreza5 "nn_64_64_183b.csv")
(def mreza5 (atom (create-network-from-file "nn_64_64_169.csv")))
(def mreza5 (atom (create-network-from-file "nn_64_64_183b.csv")))

(time (train-network @mreza5 input_matrix2
                     target_matrix2 1000 0.015 0.9))

(time (train-network @mreza5 input_matrix2
                     target_matrix2 2000 0.005 0))

(time (train-network-with-learning-decay-rate @mreza5 input_matrix2
                     target_matrix2 2000 0.015 0 0.02))

(time (train-network-with-learning-decay-rate @mreza5 input_matrix2
                     target_matrix2 1000 0.015 0 0.002))

(time (train-network-with-learning-decay-rate @mreza5 input_matrix2
                     target_matrix2 2000 0.015 -0.9 0.0001))

(time (train-network-with-learning-decay-rate @mreza5 input_matrix2
                     target_matrix2 10000 0.015 -0.9 0.01))

;; daje dobre rezultate sa 50 64 64 1
(time (train-network @mreza5 input_matrix2
                     target_matrix2 3000 0.015 -0.9))

(time (train-network @mreza5 input_matrix2
                     target_matrix2 13000 0.0015 -0.9))

(def mreza5 (atom (create-network-gaussian 20 [64 64] 1)))
(def mreza5 (atom (create-network-gaussian 64 [80] 1)))
(def mreza5 (atom (create-network-gaussian 64 [128 128 128] 1)))
(def mreza5 (atom (create-network-gaussian 64 [80 80 80] 1)))

(native! mreza5)

(xavier-initialization-update @mreza5)

(evaluate (predict @mreza5 (:normalized-matrix norm-input-730)) (:normalized-matrix norm-target-730))
(evaluate-abs (predict @mreza5 (:normalized-matrix norm-input-730)) (:normalized-matrix norm-target-730))

(evaluate (predict @mreza5 (:normalized-matrix test-norm-input-310)) (:normalized-matrix test-norm-target-310))
(evaluate-abs (predict @mreza5 (:normalized-matrix test-norm-input-310)) (:normalized-matrix test-norm-target-310))

(save-network-to-file @mreza5 "nn_64_80_80_80_140b.csv")
(def mreza5 (atom (create-network-from-file "nn_56_80_80_80_80_442.csv")))

(time (train-network-with-learning-decay-rate @mreza5 (:normalized-matrix norm-input-730)
                      (:normalized-matrix norm-target-730) 200 0.015 0 0.002))

(time (train-network @mreza5 (:normalized-matrix norm-input-730)
                     (:normalized-matrix norm-target-730) 100 0.0015 -0.02))


(time (train-network-with-learning-decay-rate @mreza5 (:normalized-matrix norm-input-730)
                     (:normalized-matrix norm-target-730) 200 0.015 0 0.002))

(time (train-network-with-learning-decay-rate @mreza5 (:normalized-matrix norm-input-730)
                     (:normalized-matrix norm-target-730) 150 0.0015 -0.9 0.1))

(def mreza5 (atom (create-network 64 [80 80] 1)))


;; max odstupanje
(apply max (vec (map #(:percent-abs %)
                (evaluate (predict @mreza5 (:normalized-matrix test-norm-input-310))
                           (:normalized-matrix test-norm-target-310)))))


(doseq [x (range (count izlaz-real))]
  (write-file "izlaz-real.csv" (str (nth izlaz-real x) "\n"))
  )

(doseq [x (range (count izlaz-target))]
  (write-file "izlaz-target.csv" (str (nth izlaz-target x) "\n"))
  )

(def norm-izlaz-real (vec (map #(:output %)
                          (evaluate (predict @mreza5 (:normalized-matrix test-norm-input-310))
                                    (:normalized-matrix test-norm-target-310)))))

(def norm-izlaz-target (vec (map #(:target %)
                               (evaluate (predict @mreza5 (:normalized-matrix test-norm-input-310))
                                         (:normalized-matrix test-norm-target-310)))))

(def izlaz-real (map #(+ (* % (- 165513.7 73045.8)) 73045.8) norm-izlaz-real))
(def izlaz-target (map #(+ (* % (- 165513.7 73045.8)) 73045.8) norm-izlaz-target))


(restore-output-vector test-norm-target-310 (predict @mreza5 (:normalized-matrix test-norm-input-310)) 0)
(restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)


(evaluate-original (restore-output-vector test-norm-target-310 (predict @mreza5 (:normalized-matrix test-norm-input-310)) 0)
(restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)
  )

(evaluate-original-mape
     (evaluate-original (restore-output-vector test-norm-target-310 (predict @mreza5 (:normalized-matrix test-norm-input-310)) 0)
                        (restore-output-vector test-norm-target-310 (:normalized-matrix test-norm-target-310) 0)
                        )
     )