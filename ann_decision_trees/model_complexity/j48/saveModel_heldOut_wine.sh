#!/bin/bash
train_file="/home/ubuntu/Desktop/Datasets/wine/wine_train.arff"
test_file="/home/ubuntu/Desktop/Datasets/wine/wine_test.arff"
model_file="/home/ubuntu/Desktop/Datasets/wine/WineResult/model/"
output_file="/home/ubuntu/Desktop/Datasets/wine/WineResult/raw/"
NUMBERS_C="0.1 0.2 0.3 0.4 0.5"
for i in `echo $NUMBERS_C`
do
  for j in $(seq 2 10)
    do
      z=$((i*10))
      java weka.classifiers.trees.J48 -t $train_file \
        -C $i -M $j -x 10 -o \
        -d $model_file"C"$z"M$j"\
        > $output_file"C"$z"M$j"
      
      # test set
      java weka.classifiers.trees.J48 -o \
        -l $model_file"C"$z"M$j" \
        -T $test_file \
         > $output_file"test_C"$z"M$j";
    done
done