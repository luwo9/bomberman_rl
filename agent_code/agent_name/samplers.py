"""
This file contains samplers that sample from the training set for a regression based Q.
"""
from abc import ABC, abstractmethod

import numpy as np

from .regression_models import QRegressionModel
from .training_memory import TrainingMemory

class Sampler(ABC):
    """
    Abstract base class for samplers.
    """

    @abstractmethod
    def sample(self, allowed_coordinates, Q: QRegressionModel, training_memory: TrainingMemory, batch_size: int):
        """
        Returns a sample of the training set.

        :param batch_size: int
        :return: np.ndarray, shape (batch_size, 2)
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the sampler as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the sampler from a dictionary.

        :param state_dict: dict
        """
        pass


class RandomSampler(Sampler):
    """
    Randomly samples from the training set.
    """

    def __init__(self):
        self._rng = np.random.default_rng()

    def sample(self, allowed_coordinates, Q: QRegressionModel, training_memory: TrainingMemory, batch_size: int): #Maybe Q here should be more general, like a Q_func (like QRegressionModel.predict would be)
        """
        Returns a sample of the training set.

        :param batch_size: int
        :return: np.ndarray, shape (batch_size, 2)
        """
        batch_size = min(batch_size, len(allowed_coordinates))
        indices = self._rng.choice(len(allowed_coordinates), batch_size, replace=False)
        return allowed_coordinates[indices]
    
    def state_dict(self):
        """
        Returns the state of the sampler as a dictionary.

        :return: dict
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the sampler from a dictionary.

        :param state_dict: dict
        """
        pass