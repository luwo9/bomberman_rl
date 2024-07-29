"""
This file contains the transforms that are applied to the game state before e.g. feeding it into a lookup table or a neural network.
"""

from abc import ABC, abstractmethod


class Transform(ABC):
    """
    Abstract base class for transforms.
    """

    @abstractmethod
    def transform(self, state):
        """
        Transforms the state.

        :param state: dict
        :return: np.ndarray
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the transform as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the transform from a dictionary.

        :param state_dict: dict
        """
        pass

    def __call__(self, state):
        return self.transform(state)