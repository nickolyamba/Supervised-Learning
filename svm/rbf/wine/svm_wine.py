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
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import cross_val_score, StratifiedKFold

import pandas as pd
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
    train_df = pd.read_csv(get_path("../../wine/wine_train.csv"))
    test_df = pd.read_csv(get_path("../../wine/wine_test.csv"))
    
    # convert the "quality" label column to numpy arrays
    train_Y = train_df.pop('quality').values
    train_X = train_df.values
    test_Y = test_df.pop('quality').values
    test_X = test_df.values
    
    # Standartize    
    scaler = StandardScaler()
    test_X = scaler.fit_transform(test_X)
    train_X = scaler.fit_transform(train_X)
    
    cv = cross_validation.StratifiedShuffleSplit(train_Y, n_iter=5,test_size=0.2, random_state=42)
    examples = np.array([5, 10, 50, 100, 200, 500, 1000, 1100, 2740])
    
    ''' 
    # SVC is more expensive so we do a lower number of CV iterations:
    gamma = 0.06
    C = 2
    title = "Learning Curves (SVM, RBF kernel, $\gamma={0}$, C={1})".format(gamma, C)
    cv = cross_validation.StratifiedShuffleSplit(train_Y, n_iter=5,test_size=0.2, random_state=42)
    #cv = cross_validation.StratifiedKFold(train_Y, n_folds=5, shuffle=True, random_state=42)
    #cv = cross_validation.ShuffleSplit(train_X.shape[0], n_iter=5, test_size=0.2, random_state=0)    
    
    examples = np.array([5, 10, 50, 100, 200, 500, 1000, 1100, 2740])
    estimator = SVC(gamma=gamma, C=C)
    plot_learning_curve(estimator, title, train_X, train_Y, (0.0, 1.01), cv=cv, n_jobs=-1, train_sizes = examples)

    plt.show()
    
    clf = SVC(gamma=gamma, C=C).fit(train_X, train_Y)
    print "\n Hold out test set accuracy: ", clf.score(test_X, test_Y)
    '''
    
    gamma = 2
    degree = 3
    
    clf = SVC(kernel='poly', gamma=gamma, degree=degree).fit(train_X, train_Y)
    heldout_score = clf.score(test_X, test_Y)
      
    
    title = "Learning Curves (SVM, Poly kernel, $\gamma={0}$, degree={1})\nheld out score = {2:.2f}".format(gamma, degree, heldout_score)
    estimator = SVC(kernel='poly',gamma=gamma, degree=degree)
    plot_learning_curve(estimator, title, train_X, train_Y, (0.2, 0.9), cv=cv, n_jobs=-1, 
                        train_sizes = examples, heldout_score=heldout_score)

    plt.show() 
    
    print "\n Hold out test set accuracy: ", heldout_score   
    
    
    
    
    
    
