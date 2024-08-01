"""
This file contains Implementations of the QHandler class. It is responsible for handling the Q-table/function.
"""
import numpy as np

from abc import ABC, abstractmethod

from .regression_models import RegressionModel
from .training_memory import TrainingMemory
from .samplers import Sampler
from typing import Tuple

N_MAX_STEPS = 400 # See section 3


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
    def new_step(self, state, action, reward, next_state):
        """
        Updates the Q-value of a state-action pair based on the observed reward.

        :param state: dict
        :param action: int or np.ndarray[int]
        :param reward: float
        :param next_state: dict
        """
        pass

    @abstractmethod
    def end_of_episode(self):
        """
        Called at the end of each episode. Tells the Q-handler that the last update had no next state.
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

    # A way to declare a attribute as abstract, such that it must be implemented by the subclass
    @property
    @abstractmethod
    def actions(self):
        """
        Returns the actions that the agent can take.

        :return: np.ndarray
        """
        pass


class RegressionQHandler(QHandler):
    """
    QHandler that uses a regression model to approximate the Q-function.
    """

    def __init__(self, model: RegressionModel, memory: TrainingMemory, number_of_actions: int, updates_step: Tuple[int, int, Sampler], updates_episodes: Tuple[int, int, Sampler],
                 type: str = "k_step_temporal_difference", discount_factor: float = 0.9, k_step: int = 1):
        """
        Initializes the QHandler.

        :param model: RegressionModel
        :param memory: TrainingMemory
        :param number_of_actions: int
        :param updates_step: Tuple[int, int, Sampler], (update_every, batch_size, sampler) for updating the model at the end of each step
        :param updates_episodes: Tuple[int, int, Sampler], (update_every, batch_size, sampler for updating the model at the end of each episode
        :param type: str, type of Q-learning algorithm, one of ["monte_carlo", "k_step_temporal_difference", "SARSA"]
        :param discount_factor: float, discount factor for future rewards
        :param k_step: int, number of steps to look into the future for the k-step temporal difference algorithm
        """

        # Check for correct spelling of type
        if type not in ["monte_carlo", "k_step_temporal_difference", "SARSA"]:
            raise ValueError(f"Unknown type: {type}")
        
        # Make model, memory, and actions available
        self.model = model
        self.discount_factor = discount_factor
        self.actions = np.arange(number_of_actions)

        self.training_set = memory
        self.training_set.set_discount(discount_factor)

        # Get samplers and batch sizes and update frequencies for end of game and end of step
        self._train_every_steps = updates_step[0]
        self._train_every_episodes = updates_episodes[0]

        self._n_trained_steps = 0
        self._n_trained_episodes = 0

        self._sampler_step = updates_step[2]
        self._sampler_episode = updates_episodes[2]

        self._batch_size_step = updates_step[1]
        self._batch_size_episode = updates_episodes[1]

        # Set the type of the Q-learning algorithm
        self._type = type
        if type == "k_step_temporal_difference":
            self._k_step = k_step
        self._type_map = {
            "monte_carlo": self._monte_carlo,
            "k_step_temporal_difference": self._k_step_temporal_difference,
            "SARSA": self._SARSA
        }

    def compute(self, state, action):
        """
        Returns the Q-value of a state-action pair.

        :param state: dict
        :param action: int or np.ndarray[int]
        :return: float
        """
        return self.model.predict(state, action)

    def new_step(self, state, action, reward, next_state):
        """
        Updates the Q-value of a state-action pair based on the observed reward.

        :param state: dict
        :param action: int
        :param reward: float
        :param next_state: dict
        """
        self.training_set.update(state, action, reward, next_state)
        self._n_trained_steps += 1
        if self._n_trained_steps % self._train_every_steps == 0:
            self._update_regression_model(self._sampler_step, self._batch_size_step)

    def end_of_episode(self):
        """
        Called at the end of each episode. Tells the Q-handler that the game has ended.
        """
        self.training_set.new_episode()
        self._n_trained_episodes += 1
        if self._n_trained_episodes % self._train_every_episodes == 0:
            self._update_regression_model(self._sampler_episode, self._batch_size_episode)
        
    def state_dict(self):
        """
        Returns the state of the Q-table as a dictionary.

        :return: dict
        """
        # Include everything needed to reconstruct the QHandler, assuming the objects of the right classes are given
        state_dict = {
            'model': self.model.state_dict(),
            'training_set': self.training_set.state_dict(),
            'number_of_actions': len(self.actions),
            'updates_step': (self._train_every_steps, self._batch_size_step),
            'samplers_step': self._sampler_step.state_dict(),
            'updates_episodes': (self._train_every_episodes, self._batch_size_episode),
            'samplers_episodes': self._sampler_episode.state_dict(),
            'type': self._type,
            'discount_factor': self.discount_factor,
            'k_step': self._k_step,
            'n_trained_steps': self._n_trained_steps,
            'n_trained_episodes': self._n_trained_episodes
        }

    def load_state_dict(self, state_dict):
        """
        Loads the state of the Q-table from a dictionary.

        :param state_dict: dict
        """
        # Load everything needed to reconstruct the QHandler
        self.model.load_state_dict(state_dict['model'])
        self.training_set.load_state_dict(state_dict['training_set'])
        self.actions = np.arange(state_dict['number_of_actions'])
        self._train_every_steps, self._batch_size_step = state_dict['updates_step']
        self._sampler_step.load_state_dict(state_dict['samplers_step'])
        self._train_every_episodes, self._batch_size_episode = state_dict['updates_episodes']
        self._sampler_episode.load_state_dict(state_dict['samplers_episodes'])
        self._type = state_dict['type']
        self.discount_factor = state_dict['discount_factor']
        self._k_step = state_dict['k_step']
        self._n_trained_steps = state_dict['n_trained_steps']
        self._n_trained_episodes = state_dict['n_trained_episodes']

    def _update_regression_model(self, sampler: Sampler, batch_size: int):
        """
        Updates the regression model using a batch of samples from the training set.
        """
        valid_coordinates = self._get_valid_coordinates()
        batch = sampler.sample(valid_coordinates, self.model, self.training_set, batch_size)
        target_Q_values = self._get_target_Q_values(batch)
        actions = self.training_set.get_actions(batch)
        states = self.training_set.get_states(batch)
        self.model.update(states, actions, target_Q_values)

    def _get_valid_coordinates(self):
        """
        Returns the coordinates of the training set that are valid for sampling.

        :return: np.ndarray, shape (batch_size, 2)
        """
        game_steps = self.training_set.number_of_steps
        # Since all methods only require some number of steps in the future, we can just track the number of steps n and then 0...n-1 steps are valid

        N_games = len(game_steps)
        if self._type == "monte_carlo":
            # Leave out the current game, as it is not finished yet
            game_steps[-1] = 0 # No(=0) usable steps here
            # There is no return for the last state of the game (terminal state)
            game_steps -= 1
        elif self._type == "k_step_temporal_difference":
            # Make sure that there are enough steps left in the game
            game_steps -= self._k_step
        elif self._type == "SARSA":
            # Make sure that there are enough steps left in the game
            game_steps -= 1
        
        # Get the coordinates
        games_valid = np.arange(N_games)
        games_valid = np.repeat(games_valid, np.maximum(game_steps, 0)) # Repeat each game number as many times as there are steps in it
        steps_valid = np.concatenate([np.arange(steps) for steps in game_steps]) # The steps are just 0, 1, 2, ..., n-1
        return np.stack([games_valid, steps_valid], axis=1)
    
    def _SARSA(self, batch):
        """
        Updates the model using the SARSA algorithm.

        :param batch: np.ndarray, shape (batch_size, 2)

        :return: np.ndarray, shape (batch_size,)
        """
        rewards = self.training_set.get_rewards(batch)
        next_step_batch = batch + np.array([0, 1])
        next_states = self.training_set.get_states(next_step_batch)
        next_actions = self.training_set.get_actions(next_step_batch)
        next_q_values = self.model.predict(next_states, next_actions)
        target_Q_values = rewards + self.discount_factor * next_q_values
        return target_Q_values
    
    def _k_step_temporal_difference(self, batch):
        """
        Updates the model using the k-step temporal difference algorithm.

        :param batch: np.ndarray, shape (batch_size, 2)
        """
        # Formula: sum_{i=0}^{k-1} gamma^i * r_{t+i} + gamma^k * max_a Q(s_{t+k}, a)
        next_step_batch = batch
        target_Q_values = 0
        for k in range(self._k_step): # could be done more efficiently
            rewards = self.training_set.get_rewards(next_step_batch)
            target_Q_values += rewards * self.discount_factor ** k
            next_step_batch = next_step_batch + np.array([0, 1])
        
        next_states = self.training_set.get_states(next_step_batch)
        all_actions = np.repeat(self.actions.reshape(1, -1), len(batch), axis=0)
        next_q_values = self.model.predict(next_states, all_actions)
        target_Q_values += self.discount_factor ** self._k_step * np.max(next_q_values, axis=1)
        return target_Q_values

    def _monte_carlo(self, batch):
        """
        Updates the model using the Monte Carlo algorithm.

        :param batch: np.ndarray, shape (batch_size, 2)
        """
        returns = self.training_set.get_returns(batch)
        return returns
    
    def _get_target_Q_values(self, batch):
        """
        Returns the target Q-values for a batch of samples.

        :param batch: np.ndarray, shape (batch_size, 2)
        """
        return self._type_map[self._type](batch)