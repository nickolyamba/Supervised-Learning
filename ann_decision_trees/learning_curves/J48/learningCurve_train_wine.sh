#!/bin/bash
F_part1='weka.filters.unsupervised.instance.Resample -S 1 -Z '
F_part2=' -no-replacement'

#-T "/home/ubuntu/Desktop/Datasets/wine/wine_test.csv"
NUMBERS="0.01 0.0292 0.29 1.46 2.92 5.83 14.58 29.16 100"
for i in `echo $NUMBERS`
do
  java weka.classifiers.meta.FilteredClassifier -t "/home/ubuntu/Desktop/Datasets/wine/wine_train.arff" \
	  -d "/home/ubuntu/Desktop/Datasets/wine/WineResult/learningCurve/model/wine$i.model" \
	  -no-cv -o -F  "weka.filters.unsupervised.instance.Resample -S 1 -Z $i -no-replacement" \
	  -W weka.classifiers.trees.J48 -- -C 0.1 -M 5 \
	  > "/home/ubuntu/Desktop/Datasets/wine/WineResult/learningCurve/raw/wine$i.out"

  java weka.classifiers.trees.J48 -no-cv -o -l \
  	"/home/ubuntu/Desktop/Datasets/wine/WineResult/learningCurve/model/wine$i.model" \
  	-T "/home/ubuntu/Desktop/Datasets/wine/wine_test.arff" \
     > "/home/ubuntu/Desktop/Datasets/wine/WineResult/learningCurve/raw/test_wine$i.out"
done
