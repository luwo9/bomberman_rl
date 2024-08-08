"""
Code for computing (shaped) rewards.
"""
from abc import ABC, abstractmethod
from typing import List

import numpy as np

import events as e

class Rewarder(ABC):
    """
    Abstract base class for rewarders.
    """

    @abstractmethod
    def compute_reward_from_events(self, events: List[str]) -> int:
        """
        Returns the reward the agent receives

        :
        :return: int 
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

    #additional methods computing proximity to other bombs/agents could be added, moving in the same place, surviving bomb explosion, ... (additional input to current event inoput is necessary for this)

class TemplateRewarder(Rewarder):
    """
    Collection of possible rewards and how they are scaled
    """

    def __init__(self) -> None:
        self.game_rewards = {   #als self definieren in _init_
            e.SURVIVED_ROUND: 0.5,  #difference to (not) got killed?
            e.COIN_COLLECTED: 2,
            e.OPPONENT_ELIMINATED: 4,
            e.KILLED_OPPONENT: 40,

            e.GOT_KILLED: -20,
            e.KILLED_SELF: -40,
            
            e.CRATE_DESTROYED: 0, #combine with coin_found
            e.COIN_FOUND: 0,
            e.BOMB_DROPPED: 0, #combine negatively with nobody killed and positively with crate_destroyed and coin_found - currently not possible because enemy doesn't die in same timestep
            #winning the game still missing, needs to be in event input
        }


    def compute_reward_from_events(self, events: List[str]) -> int:
        reward_sum = 0
        for event in events:
            if event in self.game_rewards:
                reward_sum += self.game_rewards[event]
            #simple example for combining events so that they do not count on their own
            if (e.CRATE_DESTROYED in events) & (e.COIN_FOUND in events):
                    reward_sum += 1
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}") #raus, auch bei anderem
        return reward_sum
    
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

class CoinCollectRewarder(Rewarder):
    """
    Collection of possible rewards and how they are scaled
    """
    def __init__(self) -> None:
        self.game__rewards = {
            e.SURVIVED_ROUND: 0.5,
            e.COIN_COLLECTED: 2,
            
            e.GOT_KILLED: -20,
            e.KILLED_SELF: -40, 
        }
    def compute_reward_from_events(self, events: List[str]) -> int:
        reward_sum = 0
        for event in events:
            if event in self.game_rewards:
                reward_sum += self.game_rewards[event]
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        return reward_sum
    
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
    




class SimpleRewarder(Rewarder):
    """
    Simple test rewarder
    """
    def __init__(self) -> None:
        #events rewards similar to the template in tpl_agent, modify those for testing
        self.game_rewards = {
            e.COIN_COLLECTED: 1,
            e.KILLED_OPPONENT: 5,
            #insert further rewards 
        }

    def compute_reward_from_events(self, events: List[str]) -> int:
        
        reward_sum = 0
        for event in events:
            if event in self.game_rewards:
                reward_sum += self.game_rewards[event]
        self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
        return reward_sum
    
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