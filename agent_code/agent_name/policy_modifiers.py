"""
Contains things like exploration-exploitation handlers, e.g., epsilon-greedy.
"""

from abc import ABC, abstractmethod


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
