"""
This file contains the transforms that are applied to the game state before e.g. feeding it into a lookup table or a neural network.
"""
from abc import ABC, abstractmethod

import numpy as np

class Transform(ABC):
    """
    Abstract base class for transforms.
    """

    @abstractmethod
    def transform(self, states, keep_bijection=True):
        """
        Transforms the state.

        :param state: list of dicts
        :param keep_bijection: bool, if True the transform simply retain the one-to-one mapping between states and transformed states,
        if False the transform can change the number of states (e.g. augmenting the data)
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
    

class AllFields(Transform):
    """
    Transforms all data into an image-like format.
    """

    def __init__(self, width, height):
        """
        Constructor.

        :param width: int
        :param height: int
        """
        self._width = width
        self._height = height

    def transform(self, states, keep_bijection=True):
        """
        Transforms the state.

        :param state: list of dicts
        :param keep_bijection: bool, if True the transform simply retain the one-to-one mapping between states and transformed states,
        if False the transform can change the number of states (e.g. augmenting the data)
        :return: np.ndarray
        """
        num_states = len(states)

        N_FIELDS = 8 # field, bombs, explosion_map, coins, self_score, self_bomb, others_score, others_bomb
        # Maybe make something like own_cooldown at some point aswell
        all_fields = np.zeros((num_states, N_FIELDS, self._width, self._height))

        for i, state in enumerate(states):
            field = state['field']
            all_fields[i, 0, :, :] = field

            bombs = state['bombs']
            for (x, y), t in bombs:
                all_fields[i, 1, x, y] = t/3

            explosion_map = state['explosion_map']
            all_fields[i, 2, :, :] = explosion_map

            coins = state['coins']
            for x, y in coins:
                all_fields[i, 3, x, y] = 1

            _, self_score, self_bomb, (x, y) = state['self']
            all_fields[i, 4, x, y] = self_score/5
            all_fields[i, 5, x, y] = self_bomb # 1 if self_bomb else 0

            for _, score, bomb, (x, y) in state['others']:
                all_fields[i, 6, x, y] = score/5
                all_fields[i, 7, x, y] = bomb

        return all_fields
    
    def state_dict(self):
        """
        Returns the state of the transform as a dictionary.

        :return: dict
        """
        return {
            'width': self._width,
            'height': self._height
        }
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the transform from a dictionary.

        :param state_dict: dict
        """
        self._width = state_dict['width']
        self._height = state_dict['height']


class AllfieldsFlat(AllFields):
    """
    Transforms all data into a flat format.
    """

    def transform(self, states, keep_bijection=True):
        """
        Transforms the state.

        :param state: list of dicts
        :param keep_bijection: bool, if True the transform simply retain the one-to-one mapping between states and transformed states,
        if False the transform can change the number of states (e.g. augmenting the data)
        :return: np.ndarray
        """
        all_fields = super().transform(states, keep_bijection)
        return all_fields.reshape(all_fields.shape[0], -1)