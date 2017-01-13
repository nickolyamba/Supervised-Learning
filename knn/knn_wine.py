"""
================================================
Resources used: http://scikit-learn.org/
================================================

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
from sklearn.neighbors import KNeighborsClassifier

import time
import os

def get_path(rel_path):
    script_dir = os.path.dirname(__file__) #absolute dir the script is in
    abs_file_path = os.path.join(script_dir, rel_path)
    
    return abs_file_path

# Function used to print cross-validation scores
def plot_score(k_range, scores, title, labels, ylim=[0.6, 1.05]):
        plt.figure()
        plt.title(title)
        plt.grid()
        
        plt.plot(k_range, scores[0], label=labels[0])
        plt.plot(k_range, scores[1], label=labels[1])
        #plt.vlines(alpha_optim, plt.ylim()[0], np.max(test_errors), color='k',
        #           linewidth=3, label='Optimum on test')
        plt.legend()
        plt.ylim(ylim)
        plt.xlabel('Number of nearest neighbors K')
        plt.ylabel('Accuracy')
        plt.show()

if __name__ == "__main__":
     # My Dataset:
    train_df = pd.read_csv(get_path("../../../wine/wine_train.csv"))
    test_df = pd.read_csv(get_path("../../../wine/wine_test.csv"))
    
    # convert the "quality" label column to numpy arrays
    train_Y = train_df.pop('quality').values
    train_X = train_df.values
    test_Y = test_df.pop('quality').values
    test_X = test_df.values
    
    # Standartize    
    #scaler = StandardScaler()
    #test_X = scaler.fit_transform(test_X)
    #train_X = scaler.fit_transform(train_X)
    
    # Cross-Validation
    cv = cross_validation.StratifiedShuffleSplit(train_Y, n_iter=3,test_size=0.2, random_state=42)
    #examples = np.array([5, 10, 50, 100, 200, 500, 1000, 1100, 2740])

    weights_list = ['uniform', 'distance']
    test_scores_uni = list()
    test_scores_dist = list()    
    
    for weights in weights_list:
        clf = KNeighborsClassifier(weights=weights).fit(train_X, train_Y)
        k_range = np.array([1, 2, 5, 10, 15, 30, 45, 100, 500])
        
        # emptly lists
        train_scores = list()
        test_scores = list()
        train_time = list()
        
        #iterate over k
        for k in k_range:
            # timed testing
            t0 = time.time()        
            clf.set_params(n_neighbors=k)
            t1 = time.time()
            time_passed = t1-t0
            clf.fit(train_X, train_Y)
            
            # append        
            train_scores.append(clf.score(train_X, train_Y))
            test_scores.append(clf.score(test_X, test_Y))
            train_time.append(time_passed)
        
        if weights == 'uniform': 
            test_scores_uni = test_scores
        else: 
            test_scores_dist = test_scores
        
        scores = [train_scores, test_scores]
        labels = ['Train', 'Test']
        title = "Learning Curves. Wine. KNN, weights='{0}'".format(weights) 
        plot_score(k_range, scores, title, labels, [0.4, 1.01])
        
        print "{0}\n Train: \n {1} \n Test:\n {2} \n Diff:\n {3}".format(weights, train_scores, test_scores, np.subtract(train_scores, test_scores))
    
    scores = [test_scores_uni, test_scores_dist]
    labels = ['Test Uniform','Test Weighted']
    title = "Learning Curves. Wine. KNN. Weighted vs Uniform" 
    plot_score(k_range, scores, title, labels, [0.4, 0.8])

    
    
    '''
    # Show estimated coef_ vs true coef
    plt.subplot(2, 1, 2)
    plt.plot(coef, label='True coef')
    plt.plot(coef_, label='Estimated coef')
    plt.legend()
    plt.subplots_adjust(0.09, 0.04, 0.94, 0.94, 0.26, 0.26)
    plt.show()
    '''
    
    
    
    
    
