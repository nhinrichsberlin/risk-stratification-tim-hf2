"""
Functions to run cross-validated feature importance computation.
"""


import pandas as pd
import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from ml.logger import custom_logger
LOGGER = custom_logger('feature_importance.py')


def permutation_importances(fitted_classifier,
                            X: pd.DataFrame,
                            y: pd.Series,
                            *,
                            scoring='neg_brier_score',
                            n_repeats=100,
                            n_jobs: int = -1) -> dict:
    """
    A wrapper for sklearn's permutation importance
    that returns computed importances in a
    dictionary with feature names as keys.
    """

    perm_imp = permutation_importance(fitted_classifier,
                                      X,
                                      y,
                                      scoring=scoring,
                                      n_jobs=n_jobs,
                                      n_repeats=n_repeats,
                                      random_state=123456)

    return {X.columns[i]: list(perm_imp.importances[i]) for i in range(X.shape[1])}


def permutation_importances_cv(
        X: pd.DataFrame,
        y: pd.Series,
        features: list,
        estimator: Pipeline,
        cv_splits: list,
        n_repeats_per_split: int = 30,
        scoring: str = 'roc_auc',
        n_jobs: int = -1
) -> (pd.Series, pd.Series):
    """
    Calculates the permutation importance of features multiple times on
    different test sets and returns a combined importance.
    """
    LOGGER.info('Estimating feature importances.')

    # perform multiple calculations of the permutation importances using
    # different train and test sets and average the results for robustness
    importances = {col: [] for col in X.columns}

    for i, split in enumerate(cv_splits, start=1):
        LOGGER.info(f'\tIteration {i}/{len(cv_splits)}')

        X_train, y_train = X.iloc[split[0]], y.iloc[split[0]]
        X_test, y_test = X.iloc[split[1]], y.iloc[split[1]]

        # fit the estimator on the training data
        estimator.fit(X_train, y_train)

        # calculate the permutation importances
        perm_imp = permutation_importances(
            fitted_classifier=estimator,
            X=X_test,
            y=y_test,
            scoring=scoring,
            n_repeats=n_repeats_per_split,
            n_jobs=n_jobs
        )

        importances = {col: imps + perm_imp[col] for col, imps in importances.items()}

    assert all(len(imp_values) == n_repeats_per_split * len(cv_splits) for imp_values in importances.values()), \
        'Nr. of entries per column not as expected.'

    # X may contain columns that we don't use as features, but are necessary as part of the pipeline
    importances = {col: imps for col, imps in importances.items() if col in features}

    # final result: the mean and std across all iterations
    importances_median = pd.Series(
        {col: np.median(imps) for col, imps in importances.items()}
    ).sort_values()

    importances_iqr = pd.Series(
        {col: np.percentile(imps, 75) - np.percentile(imps, 25) for col, imps in importances.items()}
    ).reindex(importances_median.index)

    LOGGER.info('Feature Importances: ')
    for feature_name, importance in importances_median.to_dict().items():
        LOGGER.info(f'\t{feature_name}: {np.round(importance, 3)}')

    LOGGER.info(f'Fraction of features with positive impact: '
                f'{(importances_median > 0).mean().round(3)}\n')

    return importances_median, importances_iqr
