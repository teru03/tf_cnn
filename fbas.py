import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
import xgboost as xgb
class FeatureBinarizatorAndScaler:
    """ This class needed for scaling and binarization features
    """
    NUMERICAL_FEATURES = list()
    CATEGORICAL_FEATURES = list()
    BIN_FEATURES = list()
    binarizers = dict()
    scalers = dict()

    def __init__(self, numerical=list(), categorical=list(), binfeatures = list(), binarizers=dict(), scalers=dict()):
        self.NUMERICAL_FEATURES = numerical
        self.CATEGORICAL_FEATURES = categorical
        self.BIN_FEATURES = binfeatures
        self.binarizers = binarizers
        self.scalers = scalers

    def fit(self, train_set):
        for feature in train_set.columns:

            if feature.split('_')[-1] == 'cat':
                self.CATEGORICAL_FEATURES.append(feature)
            elif feature.split('_')[-1] != 'bin':
                self.NUMERICAL_FEATURES.append(feature)

            else:
                self.BIN_FEATURES.append(feature)
        for feature in self.NUMERICAL_FEATURES:
            scaler = StandardScaler()
            self.scalers[feature] = scaler.fit(np.float64(train_set[feature]).reshape((len(train_set[feature]), 1)))
        for feature in self.CATEGORICAL_FEATURES:
            binarizer = LabelBinarizer()
            self.binarizers[feature] = binarizer.fit(train_set[feature])


    def transform(self, data):
        binarizedAndScaledFeatures = np.empty((0, 0))
        for feature in self.NUMERICAL_FEATURES:
            if feature == self.NUMERICAL_FEATURES[0]:
                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape(
                    (len(data[feature]), 1)))
            else:
                binarizedAndScaledFeatures = np.concatenate((
                    binarizedAndScaledFeatures,
                    self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]),
                                                                                       1)))), axis=1)
        for feature in self.CATEGORICAL_FEATURES:

            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,
                                                         self.binarizers[feature].transform(data[feature])), axis=1)

        for feature in self.BIN_FEATURES:
            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((
                len(data[feature]), 1))), axis=1)
        print(binarizedAndScaledFeatures.shape)
        return binarizedAndScaledFeatures
