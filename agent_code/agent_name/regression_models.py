"""
Contains different regression models that can be used to predict the next action.
"""

from abc import ABC, abstractmethod


class RegressionModel(ABC):
    """
    Abstract base class for regression models.
    """

    @abstractmethod
    def predict(self, x):
        """
        Predicts the value of x.

        :param x: np.ndarray
        :return: np.ndarray
        """
        pass

    @abstractmethod
    def update(self, x, y):
        """
        Updates the model based on the observed value y of x.

        :param x: np.ndarray
        :param y: np.ndarray
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