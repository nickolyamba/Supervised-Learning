"""
========================
Plotting Learning Curves
========================

On the left side the learning curve of a naive Bayes classifier is shown for
the digits dataset. Note that the training score and the cross-validation score
are both not very good at the end. However, the shape of the curve can be found
in more complex datasets very often: the training score is very high at the
beginning and decreases and the cross-validation score is very low at the
beginning and increases. On the right side we see the learning curve of an SVM
with RBF kernel. We can see clearly that the training score is still around
the maximum and the validation score could be increased with more training
samples.
"""
print(__doc__)

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.svm import SVC
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import cross_val_score
from sklearn import metrics

import time
import os


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 5), heldout_score=None):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    print "train scores\n", train_scores

    print "test_scores \n", test_scores
    
    return plt

def get_path(rel_path):
    script_dir = os.path.dirname(__file__) #absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    
    return abs_file_path

# Function used to print cross-validation scores
def training_score(est, X, y, cv):
    acc = cross_val_score(est, X, y, cv = cv, scoring='accuracy')
    roc = cross_val_score(est, X, y, cv = cv, scoring='roc_auc')
    print '5-fold Train CV | Accuracy:', round(np.mean(acc), 3),'+/-', \
    round(np.std(acc), 3),'| ROC AUC:', round(np.mean(roc), 3), '+/-', round(np.std(roc), 3)
    

if __name__ == "__main__":
     # My Dataset:
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
    
    # Cross-Validation
    cv = cross_validation.StratifiedShuffleSplit(train_Y, n_iter=5,test_size=0.2, random_state=42)
    examples = np.array([10, 50, 100, 500, 1000, 5000, 10000, 16800])
    gamma = 0.3
    C = 1
    
    # timed testing
    t0 = time.time()
    clf = SVC(gamma=gamma, C=C).fit(train_X, train_Y)
    t1 = time.time()
    train_time = t1-t0
    
    heldout_score = clf.score(test_X, test_Y)
    
    title = "Learning Curves (SVM, RBF kernel, $\gamma={0}$, C={1})\n" \
    "held out score={2:.2f}, train time={3:.2f}".format(gamma, C, heldout_score, train_time)
    
    estimator = SVC(gamma=gamma, C=C, cache_size=750)
    plot_learning_curve(estimator, title, train_X, train_Y, (0.7, 1.0), cv=cv, n_jobs=1, 
                        train_sizes = examples, heldout_score=heldout_score)

    plt.show()
    
    # http://www.astroml.org/sklearn_tutorial/classification.html
    pred_Y = clf.predict(test_X)
    print metrics.classification_report(test_Y, pred_Y)
    
    print "\nHold out test set accuracy: ", heldout_score
    print "\nTrain time: ", train_time
    
    
    
    
    
    
