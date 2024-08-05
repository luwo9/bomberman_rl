"""
Defines agents that do the Q-learning algorithm.
"""
from abc import ABC, abstractmethod

import numpy as np

from .qhandler import QHandler
from .policy_modifiers import ExplorationExploitationHandler


class QLearningAgent(ABC):
    """
    Abstract base class for Q-learning agents.
    """

    @abstractmethod
    def get_action(self, state, explore=False):
        """
        Returns the action that the agent will take in the current state.

        :param state: dict
        :param explore: bool, optional (default=False), whether to explore or exploit
        :return: int
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state):
        """
        Updates Q based on the observed reward.

        :param state: dict
        :param action: int
        :param reward: int
        :param next_state: dict
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the agent as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the agent from a dictionary.

        :param state_dict: dict
        """
        pass


class SimpleQLearningAgent(QLearningAgent):
    """
    Standard general Q-learning agent.
    """

    def __init__(self, Q_handler: QHandler, ex_ex_handler: ExplorationExploitationHandler):
        """
        Initialize the agent.

        :param Q_handler: QHandler object
        :param eps_handler: EpsHandler object
        """
        self._Q = Q_handler
        self._ex_ex_handler = ex_ex_handler
        self.actions = Q_handler.actions

    def get_action(self, state, explore=False):
        """
        Returns the action that the agent will take in the current state.

        :param state: dict
        :return: int
        """
        # Deterministically choose the action, but let the exploration-exploitation handler decide
        Q_s_a = self._Q.compute(state, self.actions)
        if explore:
            action = self._ex_ex_handler.get_action(Q_s_a, self.actions)
        else:
            action = self.actions[np.argmax(Q_s_a)]
        return action
    
    def update(self, state, action, reward, next_state):
        """
        Updates Q based on what happened.

        :param state: dict
        :param action: int
        :param reward: float
        :param next_state: dict or None
        """
        self._Q.new_step(state, action, reward, next_state)
        if next_state is None:
            self._Q.end_of_episode()

    def state_dict(self):
        """
        Returns the state of the agent as a dictionary.

        :return: dict
        """
        return {
            'Q': self._Q.state_dict(),
            'ex_ex_handler': self._ex_ex_handler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the agent from a dictionary.

        :param state_dict: dict
        """
        self._Q.load_state_dict(state_dict['Q'])
        self._ex_ex_handler.load_state_dict(state_dict['ex_ex_handler'])





