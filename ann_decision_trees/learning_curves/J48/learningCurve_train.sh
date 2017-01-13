#!/bin/bash
F_part1='weka.filters.unsupervised.instance.Resample -S 1 -Z '
F_part2=' -no-replacement'
i=99.5
#-T "/home/nick/Desktop/Datasets/credit/credit_test.csv"

  java weka.classifiers.meta.FilteredClassifier -t "/home/nick/Desktop/Datasets/credit/credit_train.csv" \
	  -d "/home/nick/Desktop/Datasets/credit/CreditResult/learningCurve/model/credit$i" \
	  -no-cv -o -F  "weka.filters.unsupervised.instance.RemovePercentage -P $i" \
	  -W weka.classifiers.trees.J48 -- -C 0.1 -M 5 \
	  > "/home/nick/Desktop/Datasets/credit/CreditResult/learningCurve/raw/credit$i.out"

  java weka.classifiers.trees.J48 -no-cv -o -l \
  	"/home/nick/Desktop/Datasets/credit/CreditResult/learningCurve/model/credit$i" \
  	-T "/home/nick/Desktop/Datasets/credit/credit_test.csv" \
     > "/home/nick/Desktop/Datasets/credit/CreditResult/learningCurve/raw/test_credit$i.out"