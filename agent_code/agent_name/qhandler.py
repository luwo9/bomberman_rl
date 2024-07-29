"""
This files contains Implementations of the QHandler class. It is responsible for handling the Q-table/function.
"""
import numpy as np

from abc import ABC, abstractmethod


class QHandler(ABC):
    """
    Abstract base class for Q handlers.
    """

    @abstractmethod
    def compute(self, state, action):
        """
        Returns the Q-value of a state-action pair.

        :param state: dict
        :param action: int or np.ndarray[int]
        :return: float
        """
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value of a state-action pair based on the observed reward.

        :param state: dict
        :param action: int or np.ndarray[int]
        :param reward: float
        :param next_state: dict
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the Q-table as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the Q-table from a dictionary.

        :param state_dict: dict
        """
        pass


class RegressionQHandler(QHandler):
    """
    QHandler that uses a regression model to approximate the Q-function.
    """

    def __init__(self, model, number_of_actions: int, type: str = "temporal_difference", discount_factor: float = 0.9, batch_size: int = 32):
        """
        Initializes the QHandler.

        :param model: object
        """
        if type not in ["temporal_difference", "monte_carlo", "k_step_temporal_difference", "SARSA"]:
            raise ValueError(f"Unknown type: {type}")
        self.model = model
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.actions = np.arange(number_of_actions)

        self.training_set = []

        _mapping = {
            "temporal_difference": self._temporal_difference,
            "monte_carlo": self._monte_carlo,
            "k_step_temporal_difference": self._k_step_temporal_difference,
            "SARSA": self._SARSA
        }
        self._get_target_Q_values = _mapping[type]

    def compute(self, state, action):
        """
        Returns the Q-value of a state-action pair.

        :param state: dict
        :param action: int or np.ndarray[int]
        :return: float
        """
        return self.model.predict(state, action)
    
    @abstractmethod
    def _sample(self, batch_size):
        """
        Returns a sample of the training set. to be used for updating the model.

        :param batch_size: int
        :return: list
        """
        pass

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-value of a state-action pair based on the observed reward.

        :param state: dict
        :param action: int or np.ndarray[int]
        :param reward: float
        :param next_state: dict
        """
        self.training_set.append((state, action, reward, next_state))
        batch_size = min(self.batch_size, len(self.training_set))
        batch = self._sample(batch_size)
        target_Q_values = self._get_target_Q_values(batch)
        self.model.update(batch, target_Q_values)

    def state_dict(self):
        """
        Returns the state of the Q-table as a dictionary.

        :return: dict
        """
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """
        Loads the state of the Q-table from a dictionary.

        :param state_dict: dict
        """
        self.model.load_state_dict(state_dict)

    def _temporal_difference(self, batch):
        """
        Updates the model using the temporal difference algorithm.

        :param batch: list
        """
        states, actions, rewards, next_states = zip(*batch)
        next_Q_values = np.max(self.model.predict(next_states, self.actions), axis=1)
        target_Q_values = np.array(rewards) + self.discount_factor * next_Q_values
        return target_Q_values
    
    def _SARSA(self, batch):
        """
        Updates the model using the SARSA algorithm.

        :param batch: list
        """
        raise NotImplementedError
    
    def _k_step_temporal_difference(self, batch):
        """
        Updates the model using the k-step temporal difference algorithm.

        :param batch: list
        """
        raise NotImplementedError
    
    def _monte_carlo(self, batch):
        raise NotImplementedError