"""
Code for computing (shaped) rewards.
"""
from abc import ABC, abstractmethod
from typing import List

import events as e

class Rewarder(ABC):
    """
    Abstract base class for rewarders.
    """

    def compute_reward_from_events(self, events: List[str]) -> int:
        """
        Returns the reward the agent receives

        :return: int 
        """
        reward_sum = 0
        for event in events:
            reward_sum += self._rewards_map.get(event, 0)
        return reward_sum/10

    @property
    @abstractmethod
    def _rewards_map(self) -> dict:
        """
        Returns the rewards map.

        :return: dict
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the rewarder as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the rewarder from a dictionary.

        :param state_dict: dict
        """
        pass

# Additional methods computing proximity to other bombs/agents could be added, moving in the same place, surviving bomb explosion, ... (additional input to current event inoput is necessary for this)

class TemplateRewarder(Rewarder):
    """
    Collection of possible rewards and how they are scaled.
    """

    def __init__(self) -> None:
        self._game_rewards = {   
            e.SURVIVED_ROUND: 0.5,  #difference to (not) got killed?
            e.COIN_COLLECTED: 2,
            e.OPPONENT_ELIMINATED: 4,
            e.KILLED_OPPONENT: 40,

            e.GOT_KILLED: -20,
            e.KILLED_SELF: -40,
            
            e.CRATE_DESTROYED: 0,
            e.COIN_FOUND: 0,
            e.BOMB_DROPPED: 0, #combine negatively with nobody killed and positively with crate_destroyed and coin_found - currently not possible because enemy doesn't die in same timestep
            #winning the game still missing, needs to be in event input
        }

    @property
    def _rewards_map(self):
        return self._game_rewards
    
    def state_dict(self):
        """
        Returns the state of the rewarder as a dictionary.

        :return: dict
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the rewarder from a dictionary.

        :param state_dict: dict
        """
        pass


class CoinsSurvives(Rewarder):
    """
    Collection of possible rewards and how they are scaled
    """
    def __init__(self) -> None:
        self._game_rewards = {
            e.COIN_COLLECTED: 2,
            e.INVALID_ACTION: -0.8,
            e.NO_COIN: -2,

            e.WAITED: -0.1,
            
            e.KILLED_SELF: -40,
            e.IN_BOMB_RANGE_1: -5,
            e.IN_BOMB_RANGE_0: -10,
        }
    
    @property
    def _rewards_map(self):
        return self._game_rewards
    
    def state_dict(self):
        """
        Returns the state of the rewarder as a dictionary.

        :return: dict
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the rewarder from a dictionary.

        :param state_dict: dict
        """
        pass


class SurviveCratesCoins(Rewarder):
    """
    Collection of possible rewards and how they are scaled
    """
    def __init__(self) -> None:
        self._game_rewards = {
            e.COIN_COLLECTED: 6,
            e.INVALID_ACTION: -0.7,
            e.NO_COIN: -2,
            # e.CRATE_DESTROYED: 8,
            # e.NO_CRATE: -0.3,
            e.BOMB_DROPPED_NEXT_TO_CRATE_1: 3,
            e.BOMB_DROPPED_NEXT_TO_CRATE_2: 3,
            e.BOMB_DROPPED_NEXT_TO_CRATE_4: 4,
            e.BOMB_DROPPED_NEXT_TO_CRATE_8: 5,
            e.BOMB_POSSIBLE_BUT_NO_CRATE_IN_RANGE: -1,

            e.WAITED: -0.5,
            # e.SURVIVED_ROUND: -0.4,
            
            e.KILLED_SELF: -20,
            e.IN_BOMB_RANGE_1: -3,
            e.IN_BOMB_RANGE_0: -5,
            # e.IN_BOMB_RANGE_3: -2,

            e.BOMB_DROPPED: 3, # Dont penalize for dropping a bomb
            e.BOMB_DISTANCE_0: -3, # Only if not dropped, but still on same tile
            # e.BOMB_DISTANCE_1: -2,
        }
    
    @property
    def _rewards_map(self):
        return self._game_rewards
    
    def state_dict(self):
        """
        Returns the state of the rewarder as a dictionary.

        :return: dict
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the rewarder from a dictionary.

        :param state_dict: dict
        """
        pass


class SurviveCratesCoinsKills(Rewarder):
    """
    Collection of possible rewards and how they are scaled
    """
    def __init__(self) -> None:
        self._game_rewards = {
            e.COIN_COLLECTED: 6,
            e.INVALID_ACTION: -7,
            e.NO_COIN: -1,
            
            e.BOMB_DROPPED_NEXT_TO_CRATE_PER_CRATE: 1.3,
            e.BOMB_POSSIBLE_BUT_NO_CRATE_IN_RANGE: -1,
            
            e.GOT_KILLED: -100,
            e.IN_BOMB_RANGE_1: -10,
            e.IN_BOMB_RANGE_0: -15,

            e.BOMB_DROPPED: 3, # Dont penalize for dropping a bomb:
            e.BOMB_DISTANCE_0: -3, # Only if not dropped, but still on same tile
            e.BOMB_DISTANCE_1: -2,
            e.KILLED_OPPONENT: 30,

            e.BOMB_DROPPED_NEXT_TO_OPPONENTS_1: 8,
            e.BOMB_DROPPED_NEXT_TO_OPPONENTS_2: 9,

            e.CLOSEST_ENEMY_CLOSER: 0.5,
            e.CLOSEST_ENEMY_FURTHER: -0.5,
        }
    
    @property
    def _rewards_map(self):
        return self._game_rewards
    
    def state_dict(self):
        """
        Returns the state of the rewarder as a dictionary.

        :return: dict
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the rewarder from a dictionary.

        :param state_dict: dict
        """
        pass