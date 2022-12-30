"""
Outliers removal
balance the dataset
Scalling the dataset
different Feature Selection Approaches
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

from collections import Counter
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from imblearn.under_sampling import ClusterCentroids

class Feat_select():
    """To clean the dataset and select features."""
    def preprocess_dataset(dataset):
        #seperate X and y and drop day,month columns.
        X = dataset.drop(['day','month', 'y'],axis=1)
        y = dataset['y']

        #Encoding the dataset
        X = pd.get_dummies(X)
        le = LabelEncoder()
        y = le.fit_transform(y)

        #z = np.abs(stats.zscore(dataset.iloc[:, [0, 5, 9, 11, 12]]))
        z = np.abs(stats.zscore(X))
        #only keep rows in dataframe with all z-scores less than absolute value of 3
        X_clean = X[(z<3).all(axis=1)]
        y_clean = y[(z<3).all(axis=1)]

        #undersampling to balance dataset
        sm2 = ClusterCentroids()
        X_clean, y_clean = sm2.fit_resample(X_clean, y_clean)
        print('Resampled dataset shape %s' % Counter(y_clean))


        scaling = StandardScaler()
        X_clean = scaling.fit_transform(X_clean)

        return X_clean, y_clean
