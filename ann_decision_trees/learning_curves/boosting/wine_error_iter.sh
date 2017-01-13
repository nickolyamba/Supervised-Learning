#!/bin/bash
train_file="/home/nick/Desktop/Datasets/wine/wine_train.arff"
test_file="/home/nick/Desktop/Datasets/wine/wine_test.arff"
model_file="/home/nick/Desktop/Datasets/wine/boost/model/"
output_file="/home/nick/Desktop/Datasets/wine/boost/raw/"

NUMBERS_C="0.005" #"0.005 0.01 0.1"
NUMBERS_M="2" #"2 5 10"
NUMBERS_I="200 500" #"10 30 50 100"

CV=5

for c in `echo $NUMBERS_C`
  do
    for m in `echo $NUMBERS_M`
      do
        for i in `echo $NUMBERS_I`
          do
            C=$(python -c "print($c*1000)")
            C=$(printf "%.0f" $C)
            M=$m
            I=$i

            echo "C$C""_M$M""_I$I"

            java weka.classifiers.meta.AdaBoostM1 -t $train_file \
              -P 100 -S 1 -I $i -x $CV \
              -d "$model_file""C$C""_M$M""_I$I" \
              -W weka.classifiers.trees.J48 -- -C $c -M $m \
              > "$output_file""C$C""_M$M""_I$I"

            # Test using test set
            start=`date +%s`
            java weka.classifiers.meta.AdaBoostM1 -o \
              -l "$model_file""C$C""_M$M""_I$I" \
              -T $test_file \
              > "$output_file""test_C$C""_M$M""_I$I"
            
            end=`date +%s`
            runtime=$((end-start))
            echo "model testing time:" $runtime sec >> "$output_file""test_C$C""_M$M""_I$I"
          done
    done
done