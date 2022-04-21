"""
An sklearn pipeline compatible estimator that sub-sets
a dataframe to only the desired features.
"""

from sklearn.base import BaseEstimator
import pandas as pd


class FeatureSelector(BaseEstimator):
    def __init__(self, features: list = []):
        self.features = features

    def _check_features(self, X: pd.DataFrame):

        if not all(type(feature) == str for feature in self.features):
            raise TypeError('Expected features to be a list of strings.')

        unknowns = set(self.features) - set(X.columns)
        if len(unknowns) > 0:
            raise ValueError(f'{unknowns} are unknown features.')

        return

    def transform(self, X: pd.DataFrame, y=None):
        self._check_features(X)
        return X[self.features]

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        return self.transform(X, y)
