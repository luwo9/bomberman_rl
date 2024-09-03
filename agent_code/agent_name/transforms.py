"""
This file contains the transforms that are applied to the game state before e.g. feeding it into a lookup table or a neural network.
"""
from abc import ABC, abstractmethod

import numpy as np

from .qsettings import ACTIONS_INV_MAP
from .bomberman_base import get_blast_coords

# Define action integers
# NOTE: The actual actions here are transposed:
# - Rotating a field 90 degrees is not the same as rotating the array representing it 90 degrees
# - Image coordinates are used (would need to transpose)
# - Instead, the actions are transposed here for more intuitive use
# - Now the array orientation fits the intuitive action orientation
# - "UP" moves a row up, "RIGHT" moves a column to the right, etc.
LEFT = ACTIONS_INV_MAP['UP']
DOWN = ACTIONS_INV_MAP['RIGHT']
RIGHT = ACTIONS_INV_MAP['DOWN']
UP = ACTIONS_INV_MAP['LEFT']


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

    def augment(self, transformed_states, actions):
        """
        Augments the transformed states and actions.

        Returns two numpy arrays (states, actions) where a new, 0th axis is the augmentation axis.

        :param transformed_states: np.ndarray
        :param actions: np.ndarray
        :return: np.ndarray, np.ndarray
        """
        return transformed_states.reshape(1, *transformed_states.shape), actions.reshape(1, *actions.shape) # Just identity as default

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
    

def augment_fields(fields):
        """
        Augments image-like fields to states that are equivalent.

        .. note:: Returns 4 rotations (counter-clockwise) and 4 reflections (vertical axis, horizontal axis, matrix diagonal, anti-diagonal) in this order.

        :param fields: np.ndarray (Assumes shape (..., width, height))
        :return: np.ndarray
        """

        # Augment by using the dihedral group of the square
        # Contains 4 rotations and 4 reflections

        # Note: The code below could be optimized
        # (flips and transpositions seem faster than rot90
        # by using group theory one could find a more efficient way to obtain all elements)
        # but this is not a bottleneck for now and the difference is small
        new_shape_fields = (8,) + fields.shape
        augmented_fields = np.empty(new_shape_fields)

        # Rotations:
        augmented_fields[0] = fields
        augmented_fields[1] = np.rot90(fields, k=1, axes=(-2,-1))
        rot180 = np.rot90(fields, k=2, axes=(-2,-1))
        augmented_fields[2] = rot180
        augmented_fields[3] = np.rot90(fields, k=3, axes=(-2,-1))

        # Flips
        augmented_fields[4] = np.flip(fields, axis=-1)
        augmented_fields[5] = np.flip(fields, axis=-2)
        # Diagonal flip: Transpose and anti-transpose
        augmented_fields[6] = np.swapaxes(fields, -2, -1)
        augmented_fields[7] = np.swapaxes(rot180, -2, -1)

        return augmented_fields


def augment_actions(actions):
        """
        Augments the actions to actions that are equivalent.

        .. note:: Returns 4 rotations (counter-clockwise) and 4 reflections (vertical axis, horizontal axis, matrix diagonal, anti-diagonal) in this order.

        :param actions: np.ndarray, shape (n,)
        :return: np.ndarray
        """
        # Augment by using the dihedral group of the square
        # Contains 4 rotations and 4 reflections

        # The implementation below is not perfectly efficient (and somewhat codeheavy)
        # But its failsafe-ness and readability should be worth it
        # (+small performance difference, not a bottleneck)

        augmented_actions = np.tile(actions, (8, 1))

        # Masks for the different actions
        mask_up = actions == UP
        mask_right = actions == RIGHT
        mask_down = actions == DOWN
        mask_left = actions == LEFT

        # Rotations
        # Identity is already done
        # Rotate in mathematically positive direction (counter-clockwise)
        augmented_actions[1][mask_up] = LEFT
        augmented_actions[1][mask_right] = UP
        augmented_actions[1][mask_down] = RIGHT
        augmented_actions[1][mask_left] = DOWN

        augmented_actions[2][mask_up] = DOWN
        augmented_actions[2][mask_right] = LEFT
        augmented_actions[2][mask_down] = UP
        augmented_actions[2][mask_left] = RIGHT

        augmented_actions[3][mask_up] = RIGHT
        augmented_actions[3][mask_right] = DOWN
        augmented_actions[3][mask_down] = LEFT
        augmented_actions[3][mask_left] = UP

        # Flips
        # Along vertical axis
        augmented_actions[4][mask_right] = LEFT
        augmented_actions[4][mask_left] = RIGHT

        # Along horizontal axis
        augmented_actions[5][mask_up] = DOWN
        augmented_actions[5][mask_down] = UP

        # Along diagonal
        augmented_actions[6][mask_up] = LEFT
        augmented_actions[6][mask_left] = UP
        augmented_actions[6][mask_right] = DOWN
        augmented_actions[6][mask_down] = RIGHT

        # Along anti-diagonal
        augmented_actions[7][mask_up] = RIGHT
        augmented_actions[7][mask_right] = UP
        augmented_actions[7][mask_left] = DOWN
        augmented_actions[7][mask_down] = LEFT
        
        return augmented_actions
    

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
                all_fields[i, 1, x, y] = t/3+1

            explosion_map = state['explosion_map']
            all_fields[i, 2, :, :] = explosion_map

            coins = state['coins']
            for x, y in coins:
                all_fields[i, 3, x, y] = 1

            _, self_score, self_bomb, (x, y) = state['self']
            all_fields[i, 4, x, y] = self_score/5+1
            all_fields[i, 5, x, y] = self_bomb # 1 if self_bomb else 0

            for _, score, bomb, (x, y) in state['others']:
                all_fields[i, 6, x, y] = score/5+1
                all_fields[i, 7, x, y] = bomb

        return all_fields
    
    def augment(self, transformed_states, actions):
        """
        Augments the transformed states and actions.

        Returns two numpy arrays (states, actions) where a new, 0th axis is the augmentation axis.

        :param transformed_states: np.ndarray
        :param actions: np.ndarray
        :return: np.ndarray, np.ndarray
        """

        augmented_states = augment_fields(transformed_states)
        augmented_actions = augment_actions(actions)

        return augmented_states, augmented_actions

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
        self._field_shape = all_fields.shape[1:]
        return all_fields.reshape(all_fields.shape[0], -1)

    def augment(self, transformed_states, actions):
        """
        Augments the transformed states and actions.

        Returns two numpy arrays (states, actions) where a new, 0th axis is the augmentation axis.

        :param transformed_states: np.ndarray
        :param actions: np.ndarray
        :return: np.ndarray, np.ndarray
        """
        # Reshape back to original shape
        transformed_states = transformed_states.reshape(-1, *self._field_shape)
        augmented_fields, augmented_actions =  super().augment(transformed_states, actions)
        augmented_flat =  augmented_fields.reshape(augmented_fields.shape[0], augmented_fields.shape[1], -1)
        return augmented_flat, augmented_actions
    

class AllFieldsBombPreview(AllFields):
    """
    Transforms all data into an image-like format but encoding bombs as a preview of their blast coverage.
    """

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
                blast_coords = get_blast_coords(x, y)
                x_blast, y_blast = blast_coords.T
                all_fields[i, 1, x_blast, y_blast] = t/3+1

            explosion_map = state['explosion_map']
            all_fields[i, 2, :, :] = explosion_map

            coins = state['coins']
            for x, y in coins:
                all_fields[i, 3, x, y] = 1

            _, self_score, self_bomb, (x, y) = state['self']
            all_fields[i, 4, x, y] = self_score/5+1
            all_fields[i, 5, x, y] = self_bomb # 1 if self_bomb else 0

            for _, score, bomb, (x, y) in state['others']:
                all_fields[i, 6, x, y] = score/5+1
                all_fields[i, 7, x, y] = bomb

        return all_fields