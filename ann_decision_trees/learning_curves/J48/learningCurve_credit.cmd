java weka.experiment.CrossValidationResultProducer -X 10 -O output.zip^
 -W weka.experiment.ClassifierSplitEvaluator -- ^
 -W weka.classifiers.meta.FilteredClassifier ^
 -I 0 -C 1 -- ^
 -F "weka.filters.unsupervised.instance.RemovePercentage -P 50.0" ^
 -W weka.classifiers.trees.J48 -- -C 0.1 -M 5