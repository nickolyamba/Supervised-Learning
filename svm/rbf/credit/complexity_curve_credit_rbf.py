"""
==========================
Plotting Validation Curves
==========================

In this plot you can see the training scores and validation scores of an SVM
for different values of the kernel parameter gamma. For very low values of
gamma, you can see that both the training score and the validation score are
low. This is called underfitting. Medium values of gamma will result in high
values for both scores, i.e. the classifier is performing fairly well. If gamma
is too high, the classifier will overfit, which means that the training score
is good but the validation score is poor.
"""
print(__doc__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.learning_curve import validation_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedShuffleSplit

import os

def get_path(rel_path):
    script_dir = os.path.dirname(__file__) #absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    
    return abs_file_path

#digits = load_digits()
#X, y = digits.data, digits.target

train_df = pd.read_csv(get_path("../../../../credit/credit_train.csv"))
test_df = pd.read_csv(get_path("../../../../credit/credit_test.csv"))

# convert the "quality" label column to numpy arrays
train_Y = train_df.pop('DEFAULT').values
train_X = train_df.values
test_Y = test_df.pop('DEFAULT').values
test_X = test_df.values

# Standartize    
scaler = StandardScaler()
test_X = scaler.fit_transform(test_X)
train_X = scaler.fit_transform(train_X)

cv = StratifiedShuffleSplit(test_Y, n_iter=1, test_size=0.2, random_state=42)
best_C = 1;

param_range = np.logspace(-2, 0, 3)
train_scores, test_scores = validation_curve(
    SVC(C=best_C), test_X, test_Y, param_name="gamma", param_range=param_range,
    cv=cv, scoring="accuracy", n_jobs=-1)
    
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Model Complexity Curve. Wine. RBF Kernel")
plt.xlabel("$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.0)
plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2, color="r")
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="g")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2, color="g")
plt.legend(loc="best")
plt.show()
