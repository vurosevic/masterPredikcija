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


;; matrixs for training, 70% of all data
(def input_matrix2 (dge 50 276 (reduce into [] (map :x (read-data-training)))))
(def target_matrix2 (dge 1 276 (map :y (read-data-training))))

;; matrixs for test, 30% of all data
(def input_test_matrix2 (dge 50 91 (reduce into [] (map :x (read-data-test)))))
(def target_test_matrix2 (dge 1 91 (map :y (read-data-test))))
