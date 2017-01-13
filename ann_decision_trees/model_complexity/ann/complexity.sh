#!/bin/bash
train_file="/home/nick/Desktop/Datasets/credit/credit_train.csv"
test_file="/home/nick/Desktop/Datasets/credit/credit_test.csv"
model_file="/home/nick/Desktop/Datasets/credit/ann/complexity/model/"
output_file="/home/nick/Desktop/Datasets/credit/ann/complexity/raw/"
NUMBERS_C="0.1 0.2 0.3 0.4 0.5"
      z=10
      j=1
      
      java weka.classifiers.functions.MultilayerPerceptron -t $train_file \
        -L 0.3 -M 0.2 -N 500 -V 0 -S 0 -E 20 -H 0 -R -no-cv -o \
        -d $model_file"C"$z"M$j" \
        > $output_file"C"$z"M$j"

      start=`date +%s`
      java weka.classifiers.functions.MultilayerPerceptron \
        -l $model_file"C"$z"M$j" \
        -T $test_file \
        > $output_file"test_C"$z"M$j"
      
      end=`date +%s`
      runtime=$((end-start))
      echo "model testing time:" $runtime sec >> $output_file"test_C"$z"M$j"