"""
An abstract class sklearn-compatible TF Classifier
to be further specified to the desired class of Neural Net (MLP, LSTM etc).
"""


import os
# suppress tensorflow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # noqa: E402

from abc import ABC, abstractmethod
import numpy as np
from typing import Union

from sklearn.base import BaseEstimator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential

from ml.logger import custom_logger
from ml.utils import balanced_class_weights
from ml.tf_utils import make_keras_picklable, TFRandomState

# ensures that we can save and load keras model to/from pickle files
make_keras_picklable()

LOGGER = custom_logger('utilities/ml/nnet_classifier.py')


class NNetClassifier(BaseEstimator, ABC):
    """
    This is an sklearn-compatible wrapper for a
    Keras LSTM Neural Network for binary classification.
    """

    def __init__(self,
                 class_weight: Union[None, dict, str] = 'balanced',
                 loss: str = 'mean_squared_error',
                 batch_size: int = 100,
                 epochs: int = 10,
                 learning_rate: float = 1e-4,
                 early_stopping: bool = False,
                 early_stopping_min_delta: float = 0.0,
                 early_stopping_patience: int = 0,
                 early_stopping_mode: str = 'min',
                 verbose: int = 0,
                 random_state: int = 123):

        self.learning_rate = learning_rate
        self.class_weight = class_weight
        self.loss = loss
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.early_stopping_min_delta = early_stopping_min_delta
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_mode = early_stopping_mode
        self.verbose = verbose
        self.random_state = random_state

        # parameters to be set during fitting or prediction
        self.fitted_model = None
        self.classes_ = None
        self.n_features = None
        self.n_time_steps = None

    @abstractmethod
    def _check_params(self) -> None:
        pass

    @abstractmethod
    def _resolve_params(self) -> None:
        pass

    @abstractmethod
    def nnet_model(self) -> Sequential:
        pass

    def _class_weight(self, y):
        # return equal weights if self.class_weight = None
        # return balanced weights if self.class_weight = 'balanced'
        # else: return whatever dictionary was passed
        if self.class_weight is not None:
            if type(self.class_weight) == dict:
                return self.class_weight
            elif type(self.class_weight) == str:
                if self.class_weight != 'balanced':
                    raise ValueError('Only "balanced", None or passing a dict are supported class_weight options.')
                return balanced_class_weights(y)
        return {cls: 1 for cls in np.unique(y)}

    def fit(self, X, y, validation_data=None, sample_weight=None):
        self._resolve_params()
        self._check_params()

        # extract info on the input shape of X
        self.n_features = X.shape[-1]
        if len(X.shape) == 3:
            self.n_time_steps = X.shape[-2]

        self.classes_ = np.unique(y)
        assert set(self.classes_) == {0, 1}, 'Only (0, 1) are acceptable class labels.'

        fit_params = dict(epochs=self.epochs,
                          batch_size=self.batch_size,
                          verbose=self.verbose,
                          class_weight=self._class_weight(y),
                          validation_data=validation_data)

        if self.early_stopping:
            es = EarlyStopping(monitor='loss',
                               mode=self.early_stopping_mode,
                               min_delta=self.early_stopping_min_delta,
                               patience=self.early_stopping_patience,
                               restore_best_weights=True)
            fit_params['callbacks'] = [es]

        # build and fit model with random state for reproducibility
        with TFRandomState(seed=self.random_state):
            self.fitted_model = self.nnet_model()
            self.fitted_model.fit(X, y, **fit_params)

        if self.verbose > 0:
            LOGGER.info(f'Model Summary: \n {self.fitted_model.summary()} \n')

        return self

    @property
    def nnet_weights(self):
        return self.fitted_model.get_weights()

    def predict_proba(self, X):
        # re-shape to return sklearn-compatible array of shape (n_samples, n_classes=2)
        prob = self.fitted_model.predict(X)
        return np.concatenate((1 - prob, prob), axis=1)

    def predict(self, X):
        return np.squeeze((self.fitted_model.predict(X) > 0.5).astype(int))
