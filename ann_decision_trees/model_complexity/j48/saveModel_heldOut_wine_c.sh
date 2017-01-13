#!/bin/bash
train_file="/home/ubuntu/Desktop/Datasets/wine/wine_train.arff"
test_file="/home/ubuntu/Desktop/Datasets/wine/wine_test.arff"
model_file="/home/ubuntu/Desktop/Datasets/wine/WineResult/model/"
output_file="/home/ubuntu/Desktop/Datasets/wine/WineResult/raw/"
NUMBERS_C="0.1 0.2 0.3 0.4 0.5"
      j=2
      java weka.classifiers.trees.J48 -t $train_file \
        -U -M $j -x 10 \
        -d $model_file"UM$j"\
        > $output_file"UM$j"
      
      # test set
      java weka.classifiers.trees.J48 -o \
        -l $model_file"UM$j" \
        -T $test_file \
         > $output_file"test_UM$j"