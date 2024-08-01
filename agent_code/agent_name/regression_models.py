"""
Contains different regression models that can be used to approximate the Q-function.
"""

from abc import ABC, abstractmethod

import numpy as np

# This should maybe be reworked to be a wrapper arround a regression model, that only knows about x and y, prediction and loss (?)
# It will hold on to a transformer in any case
class QRegressionModel(ABC):
    """
    Abstract base class for Q-function regression models.
    """

    @abstractmethod
    def predict(self, states, actions):
        """
        Predicts the value of the Q-function for the given states and actions.

        :param states: dict or list of dicts
        :param actions: int or np.ndarray[int]
        :return: np.ndarray
        """
        pass

    @abstractmethod
    def update(self, states, actions, targets):
        """
        Updates the model based on the given states, actions, and targets.

        :param states: dict or list of dicts
        :param actions: int or np.ndarray[int]
        :param targets: np.ndarray
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the model as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the model from a dictionary.

        :param state_dict: dict
        """
        pass

    @property
    @abstractmethod
    def actions(self):
        """
        Returns the number of actions.

        :return: int
        """
        pass