#!/bin/bash
train_file="/home/nick/Desktop/Datasets/wine/wine_train_norm.arff"
test_file="/home/nick/Desktop/Datasets/wine/wine_test_norm.arff"
model_file="/home/nick/Desktop/Datasets/wine/ann/learningCurve/model/"
output_file="/home/nick/Desktop/Datasets/wine/ann/learningCurve/raw_resample/"
n=500
l="0.2" #"0.4 0.5"
m="0.2" #"0.4 0.5"
LAYERS="a"
CV=5

#-T "/home/ubuntu/Desktop/Datasets/wine/wine_test.csv"
SIZES="0.29 1.46 2.92 5.83 14.58 29.16 100"
for i in `echo $SIZES`
do
	L=$(python -c "print($l*10)")
	M=$(python -c "print($m*10)")
	L=$(printf "%.0f" $L)
	M=$(printf "%.0f" $M)

	echo "$i""_L$L""_M$M""_N$n"

	java weka.classifiers.meta.FilteredClassifier -t $train_file \
		-d "$model_file""$i""_L$L""_M$M""_N$n" \
		-x $CV -o -F  "weka.filters.unsupervised.instance.Resample -S 1 -Z $i -no-replacement" \
		-W weka.classifiers.functions.MultilayerPerceptron -- \
		-L $l -M $m -N $n -V 0 -S 0 -E 20 -H $LAYERS \
		> "$output_file""$i""_L$L""_M$M""_N$n"

	# Test using test set
        start=`date +%s`
        java weka.classifiers.functions.MultilayerPerceptron -o \
          -l "$model_file""$i""_L$L""_M$M""_N$n" \
          -T $test_file \
          > "$output_file""test_""$i""_L$L""_M$M""_N$n"
    	
    	end=`date +%s`
    	runtime=$((end-start))
        echo "model testing time:" $runtime sec >> "$output_file""test_""$i""_L$L""_M$M""_N$n"
done