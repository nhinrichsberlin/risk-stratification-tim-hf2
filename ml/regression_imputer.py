"""
An sklearn pipeline compatible estimator that uses linear or logistic regression
for imputation with user-defined regressors.
"""

from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd

from ml.logger import custom_logger

LOGGER = custom_logger('FeatureImputer')


class FeatureImputer(BaseEstimator):
    def __init__(self,
                 imputation_regressors: dict = {},
                 features_na_allowed: list = [],
                 verbose: bool = True):

        self.imputation_regressors = imputation_regressors
        self.features_na_allowed = features_na_allowed
        self.verbose = verbose

        # variables defined further down
        self.medians = None
        self.minima = None
        self.maxima = None
        self.models = {}

    def _check_imputation_regressors(self, X: pd.DataFrame):
        if type(self.imputation_regressors) is not dict:
            raise TypeError('Expected a dictionary.')

        if not all(type(val) == list for val in self.imputation_regressors.values()):
            raise TypeError('Expected dict values to be lists.')

        for var, regressors in self.imputation_regressors.items():
            if var not in X.columns:
                LOGGER.warning(f'Cannot impute column {var} since it is not a column of X.')
            for col in regressors:
                if col not in X.columns:
                    LOGGER.warning(f'{col} is not a column of X. '
                                   'Dropping it from the list of regressors.')
                    self.imputation_regressors[var].remove(col)

    def _check_output(self, X: pd.DataFrame):
        # check for which columns we no longer want to see any NAs
        features_no_nas = [f for f in X.columns if f not in self.features_na_allowed]
        if not X[features_no_nas].notna().all().all():
            raise ValueError('Still NAs to be found in X.')
        return

    @staticmethod
    def features_with_nas(X: pd.DataFrame):
        nas = X.isna().any()
        return list(nas.index[nas])

    @staticmethod
    def _is_binary_variable(X: pd.DataFrame, col: str):
        return set(X[col].dropna().unique()) == {0, 1}

    def _fit_model(self, X, col, model):
        regressors = self.imputation_regressors[col]
        complete_cases = X[[col] + regressors].notna().all(axis=1)

        X_train = X.loc[complete_cases, regressors]
        y_train = X.loc[complete_cases, col]

        fitted_model = model.fit(X_train, y_train)
        return fitted_model

    def _predict_model(self, X: pd.DataFrame, col: str):

        fitted_model = self.models[col]
        regressors = self.imputation_regressors[col]
        col_to_impute = X[col].copy()

        # impute all cases where the target is missing and regressors are complete
        prediction_possible = (col_to_impute.isna() & X[regressors].notna().all(axis=1))

        if self.verbose:
            LOGGER.info(f'Imputing {prediction_possible.sum()} values of column {col} via fitted regression model.')

        if prediction_possible.sum() > 0:

            imputed_values = fitted_model.predict(X.loc[prediction_possible, regressors])
            # precaution: make sure imputed values are not crazy
            col_to_impute[prediction_possible] = imputed_values.clip(0.75 * self.minima[col], 1.25 * self.maxima[col])

        # fill remaining missing values with the median
        col_to_impute.fillna(self.medians[col], inplace=True)

        return col_to_impute

    def fit(self, X: pd.DataFrame, y=None):
        self._check_imputation_regressors(X)

        self.medians = {col: X[col].median() for col in X.columns
                        if not X[col].dtype in [object, '<M8[ns]']}
        self.minima = {col: X[col].min() for col in X.columns
                       if not X[col].dtype in [object, '<M8[ns]']}
        self.maxima = {col: X[col].min() for col in X.columns
                       if not X[col].dtype in [object, '<M8[ns]']}

        for col, regressors in self.imputation_regressors.items():
            if col not in X.columns:
                LOGGER.warning(f'Skipping Imputation for {col}: unknown column.')
                continue
            if not len(regressors) > 0:
                LOGGER.warning(f'Skipping Imputation for {col}, unknown regressors.')
                continue
            if self._is_binary_variable(X, col):
                model = LogisticRegression(penalty='none', n_jobs=1, random_state=1)
                self.models[col] = self._fit_model(X, col, model)
                if self.verbose:
                    LOGGER.info(f'Fitting a Logistic Regression model for {col} on {regressors} .')
            else:
                model = LinearRegression(n_jobs=1)
                self.models[col] = self._fit_model(X, col, model)
                if self.verbose:
                    LOGGER.info(f'Fitting a Linear Regression model for {col} on {regressors} .')
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X_transformed = X.copy()
        features_to_impute = [f for f in self.features_with_nas(X) if f not in self.features_na_allowed]
        for feature in features_to_impute:
            if feature in self.models.keys():
                X_transformed[feature] = self._predict_model(X, feature)
            else:
                if self.verbose:
                    LOGGER.info(f'Imputing {X[feature].isna().sum()} values of {feature} using the Median value.')
                X_transformed[feature] = X[feature].fillna(self.medians[feature])
        self._check_output(X_transformed)
        return X_transformed

    def fit_transform(self, X: pd.DataFrame, y=None):
        self.fit(X, y)
        return self.transform(X, y)
