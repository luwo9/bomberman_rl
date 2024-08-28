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
        return {}

    def load_state_dict(self, state_dict):
        """
        Loads the state of the handler from a dictionary.

        :param state_dict: dict
        """
        pass


class ExponentialDecayEpsilonGreedy(ExplorationExploitationHandler):
    """
    Epsilon-greedy exploration-exploitation handler with exponential decay of epsilon.
    """

    def __init__(self, epsilon_start, epsilon_end, half_life):
        """
        Constructor.

        :param epsilon_start: float
        :param epsilon_end: float
        :param half_life: float
        """
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._lifetime = half_life * np.log(2)
        self._rng = np.random.default_rng()
        self._steps = 0

    def get_action(self, q_s_a, action_space):
        """
        Returns the action that the agent will take in the current state.

        :param q_s_a: np.ndarray, shape (n_actions,)
        :param action_space: np.ndarray
        :return: int
        """
        epsilon = self._epsilon_end + (self._epsilon_start - self._epsilon_end) * np.exp(-self._steps / self._lifetime)
        self._steps += 1

        if self._rng.uniform() < epsilon:
            return self._rng.choice(action_space)
        return action_space[np.argmax(q_s_a)]
    
    def state_dict(self):
        """
        Returns the state of the handler as a dictionary.

        :return: dict
        """
        return {"steps": self._steps}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the handler from a dictionary.

        :param state_dict: dict
        """
        self._steps = state_dict["steps"]