#!/bin/bash
train_file="/home/ubuntu/Desktop/Datasets/credit/credit_train.arff"
test_file="/home/ubuntu/Desktop/Datasets/credit/credit_test.arff"
model_file="/home/ubuntu/Desktop/Datasets/credit/J48/learningCurve/model/"
output_file="/home/ubuntu/Desktop/Datasets/credit/J48/learningCurve/raw/"
CV=5

#-T "/home/ubuntu/Desktop/Datasets/credit/credit_test.csv"
NUMBERS="100 99.9952 99.95 99.76 99.52 99.05 97.62 95.24 76.19 52.38 30 0"
for i in `echo $NUMBERS`
do
  echo "removeSize = $i"
  java weka.classifiers.meta.FilteredClassifier -t $train_file \
	  -d "$model_file""$i" -x $CV \
	  -o -F  "weka.filters.unsupervised.instance.RemovePercentage -P $i" \
	  -W weka.classifiers.trees.J48 -- -C 0.1 -M 2 \
	  > "$output_file""$i"

  java weka.classifiers.trees.J48 -o \
  	-l "$model_file""$i" \
  	-T $test_file \
    > "$output_file""test_""$i"
done
