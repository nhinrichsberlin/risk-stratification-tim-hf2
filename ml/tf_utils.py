"""
Tools to make TF sklearn compatible.
"""


import os
import random
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils


class TFRandomState:
    """
    This will allow us to create reproducible results with keras models.
    Copied from the scikeras package so as not to make the entire
    package a dependency. Source:
    https://github.com/adriangb/scikeras/blob/1283e87f70367722feb4f95949e7e85611f61dbd/scikeras/_utils.py
    """
    def __init__(self, seed):
        self.seed = seed
        self._not_found = object()

    def __enter__(self):
        # warnings.warn(
        #    "Setting the random state for TF involves "
        #    "irreversibly re-setting the random seed. "
        #    "This may have unintended side effects."
        # )

        # Save values
        self.origin_hashseed = os.environ.get("PYTHONHASHSEED", self._not_found)
        self.origin_gpu_det = os.environ.get("TF_DETERMINISTIC_OPS", self._not_found)
        self.orig_random_state = random.getstate()
        self.orig_np_random_state = np.random.get_state()

        # Set values
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def __exit__(self, type, value, traceback):
        if self.origin_hashseed is not self._not_found:
            os.environ["PYTHONHASHSEED"] = self.origin_hashseed
        else:
            os.environ.pop("PYTHONHASHSEED")
        if self.origin_gpu_det is not self._not_found:
            os.environ["TF_DETERMINISTIC_OPS"] = self.origin_gpu_det
        else:
            os.environ.pop("TF_DETERMINISTIC_OPS")
        random.setstate(self.orig_random_state)
        np.random.set_state(self.orig_np_random_state)
        tf.random.set_seed(None)  # TODO: can we revert instead of unset?


# HOTFIX! The following two functions allow saving and loading
# a fitted Keras Model as a pickle file
# Solution found here: https://github.com/tensorflow/tensorflow/issues/34697
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return unpack, (model, training_config, weights)

    cls = Model
    cls.__reduce__ = __reduce__
