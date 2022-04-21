"""
An sklearn pipeline compatible TF MLP Classifier.
"""

import os
# suppress tensorflow info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # noqa: E402

from typing import Union

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import AUC
from tensorflow.keras.regularizers import l1_l2

from ml.tf_nnet_classifier import NNetClassifier


class MLPBinaryClassifier(NNetClassifier):
    """
    This is an sklearn-compatible wrapper for a
    Keras MLP for binary classification.
    """

    def __init__(self,
                 hidden_layer_sizes: tuple = (100, ),
                 activation: Union[tuple, str] = 'tanh',
                 dropout_rate: Union[tuple, float] = 0.2,
                 l2_penalty: Union[tuple, float] = 0.0,
                 l1_penalty: Union[tuple, float] = 0.0,
                 learning_rate: float = 1e-4,
                 class_weight: Union[None, dict, str] = 'balanced',
                 loss: str = 'mean_squared_error',
                 batch_size: int = 100,
                 epochs: int = 10,
                 early_stopping: bool = False,
                 early_stopping_min_delta: float = 0.0,
                 early_stopping_patience: int = 0,
                 early_stopping_mode: str = 'min',
                 verbose: int = 0,
                 random_state: int = 123):

        super().__init__(
            class_weight=class_weight,
            loss=loss,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping=early_stopping,
            early_stopping_min_delta=early_stopping_min_delta,
            early_stopping_patience=early_stopping_patience,
            early_stopping_mode=early_stopping_mode,
            verbose=verbose,
            random_state=random_state
        )

        # parameters describing the model architecture
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty

    def _check_params(self):

        for param in [self.dropout_rate, self.l2_penalty, self.activation]:
            assert len(param) == len(self.hidden_layer_sizes), \
                f'Length mismatch: {param} vs. {self.hidden_layer_sizes}'

    def _resolve_params(self):

        if isinstance(self.activation, str):
            self.activation = (self.activation, ) * len(self.hidden_layer_sizes)

        if isinstance(self.dropout_rate, float):
            self.dropout_rate = (self.dropout_rate, ) * len(self.hidden_layer_sizes)

        if isinstance(self.l2_penalty, float):
            self.l2_penalty = (self.l2_penalty, ) * len(self.hidden_layer_sizes)

        if isinstance(self.l1_penalty, float):
            self.l1_penalty = (self.l1_penalty, ) * len(self.hidden_layer_sizes)

    def nnet_model(self):

        model = Sequential()

        # 1st layer
        layer = Dense(
            units=self.hidden_layer_sizes[0],
            activation=self.activation[0],
            input_shape=(self.n_features,),
            kernel_regularizer=l1_l2(l1=self.l1_penalty[0], l2=self.l2_penalty[0])
        )
        model.add(layer)

        # dropout layer
        if self.dropout_rate[0] > 0:
            layer = Dropout(rate=self.dropout_rate[0])
            model.add(layer)

        # subsequent layers
        for nr, size in enumerate(self.hidden_layer_sizes[1:], start=2):
            # lstm layer
            layer = Dense(
                units=size,
                activation=self.activation[nr - 1],
                kernel_regularizer=l1_l2(l1=self.l1_penalty[nr - 1], l2=self.l2_penalty[nr - 1])
            )
            model.add(layer)

            # dropout layer
            if self.dropout_rate[nr - 1] > 0:
                layer = Dropout(rate=self.dropout_rate[nr - 1])
                model.add(layer)

        # final layer
        layer = Dense(
            units=1,
            activation='sigmoid'
        )
        model.add(layer)

        # compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(loss=self.loss,
                      optimizer=optimizer,
                      metrics=[AUC(curve='ROC'), AUC(curve='PR')])

        return model
