"""
Train Different models for Parameters Optimization
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import statsmodels.api as sm
import scipy.stats as stats

import sklearn
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectKBest


class models_methods():

    def model_opt(pipe, params, XX_train, YY_train):
        """Optimize models using nested cross validation."""

        #paramerters for cross-validation
        cv_in = KFold(n_splits = 5, shuffle = True, random_state = 1)
        cv_out = KFold(n_splits = 10, shuffle = True, random_state = 1)
        #apply gridserach with cross_validation
        outer_results = list()

        for train_ix, test_ix in cv_out.split(XX_train):
            # split data
            x_train, x_test = XX_train[train_ix, :], XX_train[test_ix, :]
            y_train, y_test = YY_train[train_ix], YY_train[test_ix]
            # configure the cross-validation procedure
            # define gridsearch
            search = GridSearchCV(pipe, params, scoring='accuracy', cv=cv_in, refit=True,n_jobs=-1)
            result = search.fit(x_train, y_train)
            #get the best performing model fit on the whole training set
            best_model = result.best_estimator_
            #evaluate model on the hold out dataset
            yhat = best_model.predict(x_test)
            #evaluate the model
            acc = accuracy_score(y_test, yhat)
            #store the result
            outer_results.append(acc)
            # report progress
            print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))

        # summarize the estimated performance of the model
        print('Accuracy: %.3f (%.3f)' % (np.mean(outer_results), np.std(outer_results)))