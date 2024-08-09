"""
This module provides the memory buffer for training a regression-based Q-function.
"""
from collections import deque

import numpy as np

from .regression_models import QRegressionModel

N_MAX_STEPS = 400 # See section 3

class TrainingMemory:
    """
    Provides the memory buffer for training the Q-function.
    """
    def __init__(self, capacity: int):
        """
        Initializes the memory buffer.

        :param capacity: int
        """
        self.capacity = capacity
        self._memory = deque(maxlen=capacity) #[] # List of games
        
        self._first_episode = True
        self.new_episode()
        
        self._discount_was_set = False

    def update(self, state, action, reward, next_state):
        """
        Updates the memory buffer.

        :param state: dict
        :param action: int
        :param reward: float
        :param next_state: dict or None
        """
        # If first step, add "state" to the memory buffer
        if self._first_step:
            self._current_game[0].append(state)
            self._first_step = False
        # Add "next_state" to the memory buffer, the "state" is already there from the previous step
        self._current_game[0].append(next_state)
        self._current_game[1].append(action)
        self._current_game[2].append(reward)

    def new_episode(self):
        """
        Starts a new game in the memory buffer.
        """
        self._memory.append([[], [], [], None, None]) # states, actions, rewards, returns, q_values
        self._first_step = True
        if not self._first_episode:
            self._compute_and_store_returns()
        else:
            self._first_episode = False

    def set_discount(self, discount):
        """
        Sets the discount factor for the memory buffer. This will update all returns in the memory buffer.

        :param discount: float
        """
        self._discount = discount
        self._discount_weights = discount ** np.arange(N_MAX_STEPS)

        if not self._discount_was_set:
            self._discount_was_set = True
        else:
            # Update all returns
            for i in range(len(self._memory)-1):
                self._compute_returns(i)

    def _compute_and_store_returns(self):
        """
        Computes the returns for the last (completed) game in the memory buffer and stores them in the memory buffer.
        """
        self._compute_returns(-2) # -1 is the current game that is not finished yet by definition

    def compute_and_store_q_values(self, Q: QRegressionModel):
        """
        Computes the Q-values and write them to the memory buffer.

        :param Q: RegressionModel
        """
        for i in range(len(self._memory)):
            self._compute_q_values(Q, i)

    def get_states(self, coordinates):
        """
        Returns the states from the memory buffer.

        :param coordinates: np.ndarray, shape (batch_size, 2)
        :return: list
        """
        return self._get_from_memory(coordinates, 0)
    
    def get_actions(self, coordinates):
        """
        Returns the actions from the memory buffer.

        :param coordinates: np.ndarray, shape (batch_size, 2)
        :return: np.ndarray
        """
        return self._get_from_memory(coordinates, 1)
    
    def get_rewards(self, coordinates):
        """
        Returns the rewards from the memory buffer.

        :param coordinates: np.ndarray, shape (batch_size, 2)
        :return: np.ndarray
        """
        return self._get_from_memory(coordinates, 2)
    
    def get_returns(self, coordinates):
        """
        Returns the returns from the memory buffer.

        :param coordinates: np.ndarray, shape (batch_size, 2)
        :return: np.ndarray
        """
        return self._get_from_memory(coordinates, 3)
    
    def get_q_values(self, coordinates):
        """
        Returns the Q-values from the memory buffer.

        :param coordinates: np.ndarray, shape (batch_size, 2)
        :return: np.ndarray
        """
        return self._get_from_memory(coordinates, 4)
    
    @property
    def number_of_steps(self):
        """
        Returns the number of steps of each game in the memory buffer. Precisely, it returns the number of states in each game (including the terminal state).

        :return: np.ndarray
        """
        return np.array([len(game[0]) for game in self._memory])
    
    def state_dict(self):
        """
        Returns the state of the memory buffer as a dictionary.

        :return: dict
        """
        return {
            'memory': self._memory,
            'first_step': self._first_step,
            'discount': self._discount,
        }
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the memory buffer from a dictionary.

        :param state_dict: dict
        """ 
        self._memory = state_dict['memory']
        self._first_step = state_dict['first_step']
        self._discount_was_set = False # To avoid updating the returns in the memory buffer
        self.set_discount(state_dict['discount'])

    
    def _get_from_memory(self, coordinates, index):
        """
        Returns the data from the memory buffer.

        :param coordinates: np.ndarray, shape (batch_size, 2)
        :param index: int, index of the data to get in the memory buffer
        :return: list or np.ndarray
        """
        result =  [self._memory[game][index][step] for game, step in coordinates] # Make sure never a reference is returned only a copy
        if index == 0: # states are not array-able, as they are dicts
            return result
        return np.array(result)
        
    def _compute_returns(self, game_number):
        """
        Computes the returns for a game in the memory buffer. The returns are stored in the memory buffer.

        :param game_number: int
        """
        rewards = self._memory[game_number][2]
        returns = np.convolve(rewards, self._discount_weights[::-1], mode="full")[-len(rewards):]
        self._memory[game_number][3] = returns

    def _compute_q_values(self, Q, game_number):
        """
        Computes the Q-values for a game in the memory buffer. The Q-values are stored in the memory buffer.

        :param Q: RegressionModel
        :param game_number: int
        """
        states = self._memory[game_number][0]
        actions = self._memory[game_number][1]
        q_values = Q.predict(states, actions)
        self._memory[game_number][4] = q_values

    @property
    def _current_game(self):
        """
        Returns the current game in the memory buffer.

        :return: list
        """
        return self._memory[-1]