1) Models computed using WEKA CLI via Linux bash scripts:
 - Decision Tree
 - Boosting
 - ANN
 Bash Scripts require a certain pre-created folder structure for data input and output
 This structure can be seen in bash variables that define I/O files/dirrectories.


2) Models computed using Scikit Learn
All the code for computation was adapted from http://scikit-learn.org/
 - KNN
 Requires for credit files:
 	- "../../../credit/credit_train.csv" and "../../../credit/credit_test.csv" 
 Requires for wine files:
 	- "../../../wine/wine_train.csv" and "../../../wine/wine_test.csv"

 - SVM
Requires for credit files:
 	- "../../../../credit/credit_train.csv" and "../../../../credit/credit_test.csv" 
Requires for wine files:
 	- "../../../../wine/wine_train.csv" and "../../../../wine/wine_test.csv"

The code also requires installed libraries used in the code such as numpy, matplotlib, pandas, sklearn
Datasets can be found on Google Drive: https://goo.gl/Bu56rb


