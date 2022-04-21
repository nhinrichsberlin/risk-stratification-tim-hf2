"""
A selection wrappers for sklearn functions,
mostly for convenience and nicer display of outputs.
"""


import numpy as np
import pandas as pd
import datetime
from typing import Union, Any

from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


from ml.logger import custom_logger

LOGGER = custom_logger('utils.py')


def balanced_class_weights(y_train):
    # a wrapper for sklearn's compute_class_weight function
    # that returns a dictionary {class_1: weight:1, ...}
    classes = np.unique(y_train)
    class_weight_vector = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    return {classes[i]: class_weight_vector[i] for i in range(len(classes))}


def display_gridsearch_results(gridsearch: RandomizedSearchCV,
                               n_iter: int,
                               scoring: Union[str, dict],
                               start_time: datetime.datetime,
                               refit: str = None):

    end_time = datetime.datetime.now()
    minutes = ((end_time - start_time).seconds / 60)
    LOGGER.info(f'Time: {np.round(minutes, 2)} minutes')

    # show hyperparameters that yielded best results
    cv_results = gridsearch.cv_results_

    if isinstance(scoring, dict):
        scores = {key: cv_results[f'mean_test_{key}'] for key in scoring.keys()}
        scores = pd.DataFrame(
            {**scores, **{'params': cv_results['params']}}
        ).sort_values(refit, ascending=False)
    else:
        scores = pd.DataFrame(
            {'score': cv_results['mean_test_score'], 'params': cv_results['params']}
        ).sort_values('score', ascending=False)

    for k in range(min(n_iter, 5)):
        kth_best_params = scores.iloc[k, scores.shape[1] - 1]
        if isinstance(scoring, dict):
            for num, key in enumerate(scoring.keys()):
                LOGGER.info(f'\t{key}: {scores.iloc[k, num]}')
        else:
            LOGGER.info(f'\t{scoring}: {scores.iloc[k, 0]}')
        for param, val in kth_best_params.items():
            LOGGER.info(f'\t{param}: {val}')
        LOGGER.info('\n')


def get_n_splits_cv(cv: Any):
    if type(cv) == int:
        return cv
    elif hasattr(cv, '__iter__'):
        return len(cv)
    elif hasattr(cv, 'get_n_splits'):
        return cv.get_n_splits()
    else:
        raise KeyError


def tune_model(pipeline: Pipeline,
               hyperparameters: dict,
               X_train: pd.DataFrame,
               y_train: pd.Series,
               *,
               n_iter: int = 50,
               cv: Union[int, list, Any] = 10,
               scoring: Union[str, dict] = 'roc_auc',
               refit: str = None,
               fit_params: dict = {},
               n_jobs: int = -1):

    start_time = datetime.datetime.now()

    gridsearch = RandomizedSearchCV(estimator=pipeline,
                                    param_distributions=hyperparameters,
                                    n_iter=n_iter,
                                    cv=cv,
                                    n_jobs=n_jobs,
                                    scoring=scoring,
                                    refit=refit if refit else True,
                                    random_state=12345)

    LOGGER.info(f'Starting Randomized Search using {get_n_splits_cv(cv)}-fold CV '
                f'and {n_iter} iterations.')

    gridsearch = gridsearch.fit(X_train, y_train, **fit_params)
    LOGGER.info('Randomized Search complete.')

    display_gridsearch_results(
        gridsearch=gridsearch,
        n_iter=n_iter,
        scoring=scoring,
        start_time=start_time,
        refit=refit
    )

    return gridsearch
