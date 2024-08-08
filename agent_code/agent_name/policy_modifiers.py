"""
Contains things like exploration-exploitation handlers, e.g., epsilon-greedy.
"""
from abc import ABC, abstractmethod

import numpy as np


class ExplorationExploitationHandler(ABC):
    """
    Abstract base class for exploration-exploitation handlers.
    """

    @abstractmethod
    def get_action(self, q_s_a, action_space):
        """
        Returns the action that the agent will take in the current state.

        :param q_s_a: np.ndarray, shape (n_actions,)
        :param action_space: np.ndarray
        :return: int

        q_s_a is array of Q(current_state, action_i)_i.
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the handler as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the handler from a dictionary.

        :param state_dict: dict
        """
        pass


class EpsilonGreedy(ExplorationExploitationHandler):
    """
    Epsilon-greedy exploration-exploitation handler.
    """

    def __init__(self, epsilon):
        """
        Constructor.

        :param epsilon: float
        """
        self._epsilon = epsilon
        self._rng = np.random.default_rng()

    def get_action(self, q_s_a, action_space):
        """
        Returns the action that the agent will take in the current state.

        :param q_s_a: np.ndarray, shape (n_actions,)
        :param action_space: np.ndarray
        :return: int
        """
        if self._rng.uniform() < self._epsilon:
            return self._rng.choice(action_space)
        return action_space[np.argmax(q_s_a)]

    def state_dict(self):
        """
        Returns the state of the handler as a dictionary.

        :return: dict
        """
        return {"epsilon": self._epsilon}

    def load_state_dict(self, state_dict):
        """
        Loads the state of the handler from a dictionary.

        :param state_dict: dict
        """
        self._epsilon = state_dict["epsilon"]