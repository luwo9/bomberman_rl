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
    def transform(self, states):
        """
        Transforms the state.

        :param state: list of dicts
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

    def transform(self, states):
        """
        Transforms the state.

        :param state: list of dicts
        :return: np.ndarray
        """
        num_states = len(states)

        N_FIELDS = 8 # field, bombs, explosion_map, coins, self_score, self_bomb, others_score, others_bomb
        # Maybe make something like own_cooldown at some point aswell
        all_fields = np.zeros((num_states, N_FIELDS, self._width, self._height))

        for i, state in enumerate(states):
            # For the terminal state, set to nan
            if state is None:
                all_fields[i][:] = np.nan
                continue

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
    
    def augment_fields_and_actions(self, all_fields, actions):

        new_shape_fields = (8,) + all_fields.shape
        augmented_fields = np.empty(new_shape_fields)

        augmented_fields[0] = all_fields
        augmented_fields[1] = np.array([np.flip(matrix, axis=0) for matrix in all_fields])
        augmented_fields[2] = np.array([np.rot90(matrix, k=1) for matrix in all_fields])
        augmented_fields[3] = np.array([np.rot90(matrix, k=2) for matrix in all_fields])
        augmented_fields[4] = np.array([np.rot90(matrix, k=3) for matrix in all_fields])
        augmented_fields[5] = np.array([np.flip(np.rot90(matrix, k=1), axis=0) for matrix in all_fields])
        augmented_fields[6] = np.array([np.flip(np.rot90(matrix, k=2), axis=0) for matrix in all_fields])
        augmented_fields[7] = np.array([np.flip(np.rot90(matrix, k=3), axis=0) for matrix in all_fields])


        augmented_actions = np.tile(actions, (8, 1))

        augmented_actions[0] = actions
        augmented_actions[1] = np.where(augmented_actions[1] == 0, 2, np.where(augmented_actions[1] == 2, 0, augmented_actions[1]))
        augmented_actions[2][augmented_actions[2]<4]+=1
        augmented_actions[2][augmented_actions[2]<4]%4
        augmented_actions[3][augmented_actions[3]<4]+=2
        augmented_actions[3][augmented_actions[3]<4]%4
        augmented_actions[4][augmented_actions[4]<4]+=3
        augmented_actions[4][augmented_actions[4]<4]%4
        augmented_actions[5] = np.where(augmented_actions[5] == 0, 1, np.where(augmented_actions[5] == 1, 0, augmented_actions[5]))
        augmented_actions[5] = np.where(augmented_actions[5] == 2, 3, np.where(augmented_actions[5] == 3, 2, augmented_actions[5]))
        augmented_actions[6] = np.where(augmented_actions[6] == 1, 3, np.where(augmented_actions[6] == 3, 1, augmented_actions[6]))
        augmented_actions[7] = np.where(augmented_actions[7] == 1, 2, np.where(augmented_actions[7] == 2, 1, augmented_actions[7]))
        augmented_actions[7] = np.where(augmented_actions[7] == 3, 0, np.where(augmented_actions[7] == 0, 3, augmented_actions[7]))



        


    def state_dict(self):
        """
        Returns the state of the transform as a dictionary.

        :return: dict
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the transform from a dictionary.

        :param state_dict: dict
        """
        pass


class AllfieldsFlat(AllFields):
    """
    Transforms all data into a flat format.
    """

    def transform(self, states):
        """
        Transforms the state.

        :param state: list of dicts
        :return: np.ndarray
        """
        all_fields = super().transform(states)
        return all_fields.reshape(all_fields.shape[0], -1)