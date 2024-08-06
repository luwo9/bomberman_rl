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


class SimpleRewarder(Rewarder):
    """
    Simple test rewarder
    """


    def compute_reward_from_events(self, events: List[str]) -> int:
        #events rewards similar to the template in tpl_agent, modify those for testing
        game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        #insert further rewards 
        }

        reward_sum = 0
        for event in events:
            if event in game_rewards:
                reward_sum += game_rewards[event]
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