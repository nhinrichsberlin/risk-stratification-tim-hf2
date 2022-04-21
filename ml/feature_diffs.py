"""
An sklearn pipeline compatible estimator that selects desired features
and adds lags for selected features.
"""


from sklearn.base import BaseEstimator
import pandas as pd

from ml.logger import custom_logger

LOGGER = custom_logger('FeatureDiffs')


class FeatureDiffs(BaseEstimator):
    def __init__(self,
                 features_to_diff: list = [],
                 max_lag: int = None,
                 lags: list = None,
                 features_to_keep: list = [],
                 diff_separator: str = '__d_',
                 column_split_by: str = 'studp_id',
                 column_sort_by: str = 'date',
                 return_columns_split_sort: bool = False,
                 verbose: bool = True):

        self.column_split_by = column_split_by
        self.column_sort_by = column_sort_by
        self.return_columns_split_sort = return_columns_split_sort
        self.features_to_diff = features_to_diff
        self.features_to_keep = features_to_keep
        self.max_lag = max_lag
        self.lags = lags
        self.diff_separator = diff_separator
        self.verbose = verbose

        self.colnames_transformed = None

    def _check_features(self, X: pd.DataFrame):

        if self.column_sort_by not in X.columns:
            raise ValueError(f'Cannot sort by {self.column_sort_by} .')

        if self.column_split_by not in X.columns:
            raise ValueError(f'Cannot split by {self.column_split_by} .')

        if not all(type(feature) == str for feature in self.features_to_diff):
            raise TypeError('Expected features to be a list of strings.')

        if not all(feature in X.columns for feature in self.features_to_diff + self.features_to_keep):
            raise ValueError('Features contain element not in the columns of X.')

        return

    def _check_params(self):
        if not self.lags and not self.max_lag:
            raise ValueError('Please provide either max_lag or lags.')

        if self.lags and self.max_lag:
            LOGGER.warning('Please be aware that specifying lags overwrites the specified max_lag.')

        return

    def _resolve_params(self):
        # remove features listed in both lists from the list of features to keep
        # useful for gridsearching over features_to_diff
        for feature in self.features_to_diff:
            if feature in self.features_to_keep:
                self.features_to_keep.remove(feature)

    @property
    def names_new_cols(self):
        if self.lags:
            return [f'{feature}{self.diff_separator}{lag}'
                    for feature in self.features_to_diff
                    for lag in self.lags]
        else:
            return [f'{feature}{self.diff_separator}{lag}'
                    for feature in self.features_to_diff
                    for lag in range(1, self.max_lag + 1)]

    def transform(self, X: pd.DataFrame, y=None):
        self._check_params()
        self._check_features(X)
        self._resolve_params()

        # make sure the ordering is correct
        # e.g. order by patient and date
        X = X.sort_values([self.column_split_by, self.column_sort_by])

        # define the orders of differencing to create
        diff_orders = range(1, self.max_lag + 1) if not self.lags else range(1, max(self.lags) + 1)

        # as a first step, create all diffs, even if only specific ones are needed
        all_diffs = pd.concat(
            [
                X.groupby(self.column_split_by)[self.features_to_diff]
                .diff(i)
                .rename({feat: feat + f'{self.diff_separator}{i}' for feat in self.features_to_diff}, axis=1)
                for i in diff_orders
            ],
            axis=1
        )

        # fill missing values with the next lower diff
        for col in self.features_to_diff:
            for i in diff_orders:
                if i == 1:
                    all_diffs[f'{col}{self.diff_separator}{i}'].\
                        fillna(0, inplace=True)
                else:
                    all_diffs[f'{col}{self.diff_separator}{i}'].\
                        fillna(all_diffs[f'{col}{self.diff_separator}{i - 1}'], inplace=True)

        if all_diffs.isna().any().any():
            raise ValueError('NAs found.')

        # select only the needed diff orders
        X_diff = all_diffs[self.names_new_cols]

        # we return ONLY
        return_cols = self.features_to_diff + self.features_to_keep
        if self.return_columns_split_sort:
            return_cols += [self.column_split_by, self.column_sort_by]

        X_transformed = pd.concat([X[return_cols], X_diff], axis=1)

        assert set(X_diff.columns) == set(self.names_new_cols), 'New columns not as expected!'

        # we save the column names in case we want to do stuff like estimate feature importance
        self.colnames_transformed = X_transformed.columns

        return X_transformed

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        return self.transform(X, y)
