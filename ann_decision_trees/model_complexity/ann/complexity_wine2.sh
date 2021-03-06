#!/bin/bash
train_file="/home/nick/Desktop/Datasets/wine/wine_train_norm.arff"
test_file="/home/nick/Desktop/Datasets/wine/wine_test_norm.arff"
model_file="/home/nick/Desktop/Datasets/wine/ann/complexity/model/"
output_file="/home/nick/Desktop/Datasets/wine/ann/complexity/raw/"
NUMBERS_N="100 300 500 1000 2000"
NUMBERS_L="0.2 0.3 0.4 0.5" #"0.4 0.5"
NUMBERS_M="0.2 0.3 0.4 0.5" #"0.4 0.5"
LAYERS="a,a"
CV=5

for n in `echo $NUMBERS_N`
  do
    for l in `echo $NUMBERS_L`
      do
        for m in `echo $NUMBERS_M`
          do
            L=$(python -c "print($l*10)")
            M=$(python -c "print($m*10)")
            L=$(printf "%.0f" $L)
            M=$(printf "%.0f" $M)
            echo "L$L""_M$M""_N$n"
            java weka.classifiers.functions.MultilayerPerceptron -t $train_file \
              -L $l -M $m -N $n -V 0 -S 0 -E 20 -H $LAYERS -R -x $CV -o \
              -d "$model_file""L$L""_M$M""_N$n" \
              > "$output_file""L$L""_M$M""_N$n"

            # Test using test set
            start=`date +%s`
            java weka.classifiers.functions.MultilayerPerceptron -o \
              -l "$model_file""L$L""_M$M""_N$n" \
              -T $test_file \
              > "$output_file""test_L$L""_M$M""_N$n"
            
            end=`date +%s`
            runtime=$((end-start))
            echo "model testing time:" $runtime sec >> "$output_file""test_L$L""_M$M""_N$n"
          done
    done
done
