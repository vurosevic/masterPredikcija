(ns ^{:author "Vladimir Urosevic"}
  master-predikcija.data
  (:require [clojure.string :as string]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.vect-math :refer :all]
            [uncomplicate.neanderthal.native :refer :all]))

(defn parse-float [s]
  (Float/parseFloat s)
  )

(defn read-data-training
  "Read the csv file, split out each line and then each number, parse the tokens and break up the numbers so that the last is the target and everything else is the feature vector."
  []
  (as-> (slurp "resources/data_trening.csv") d
        (string/split d #"\n")
        (map #(string/split % #",") d)
        (map #(map parse-float %) d)
        (map (fn [s] {:x  (vec (drop-last s))  :y (last s)}) d)))

(defn read-data-test
  "Read the csv file, split out each line and then each number, parse the tokens and break up the numbers so that the last is the target and everything else is the feature vector."
  []
  (as-> (slurp "resources/data_test.csv") d
        (string/split d #"\n")
        (map #(string/split % #",") d)
        (map #(map parse-float %) d)
        (map (fn [s] {:x  (vec (drop-last s))  :y (last s)}) d)))


(defn write-file [filename data]
  (with-open [w (clojure.java.io/writer  (str "resources/" filename) :append true)]
    (.write w data)))

(defn save-network-to-file
  "save network state in file"
  [network filename]
  (do
    (write-file filename "CONFIGURATION\n")
    (write-file filename
                (str (string/join ""
                                  (drop-last
                                    (reduce str (map str (conj (:tmp1 network) (last (:tmp2 network)))
                                                     (replicate (count (conj (:tmp1 network) (last (:tmp2 network)))) ","))))) "\n")
                )

    (write-file filename "BIASES\n")
    (doall

      (doseq [y (range (count (:biases network)))]
        (write-file filename (str "BIAS," (inc y) "\n"))
        (doseq [x (range (mrows (nth (:biases network) y)))]
          (write-file filename
                      (str (string/join ""
                                        (drop-last
                                          (reduce str (map str (row (nth (:biases network) y) x)
                                                           (replicate (ncols (nth (:biases network) y)) ","))))) "\n"))))
      )

    (write-file filename "LAYERS\n")
    (doall

      (doseq [y (range (count (:hidden-layers network)))]
        (write-file filename (str "LAYER," (inc y) "\n"))
        (doseq [x (range (mrows (nth (:hidden-layers network) y)))]
          (write-file filename
                      (str (string/join ""
                                        (drop-last
                                          (reduce str (map str (row (nth (:hidden-layers network) y) x)
                                                           (replicate (ncols (nth (:hidden-layers network) y)) ","))))) "\n"))))
       )

    (write-file filename "OUTPUT\n")
    (doall
      (for [x (range (mrows (:output-layer network)))]
        (write-file filename
                    (str (string/join ""
                                      (drop-last
                                        (reduce str (map str (row (:output-layer network) x)
                                                         (replicate (ncols (:output-layer network)) ","))))) "\n"))))
    (write-file filename "END\n")
    ))


;; matrixs for training, 70% of all data
(def input_matrix2 (dge 50 276 (reduce into [] (map :x (read-data-training)))))
(def target_matrix2 (dge 1 276 (map :y (read-data-training))))

;; matrixs for test, 30% of all data
(def input_test_matrix2 (dge 50 91 (reduce into [] (map :x (read-data-test)))))
(def target_test_matrix2 (dge 1 91 (map :y (read-data-test))))
