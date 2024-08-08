"""
Contains classes that package together a QAgent with all it's necessary components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from . import rewarders
from . import qagents
from . import policy_modifiers
from . import qhandler
from . import regression_models
from . import networks
from . import transforms
from . import training_memory
from . import samplers

WIDTH, HEIGHT = 17, 17

N_ACTIONS = 6

ACTINS_MAP = {
    0: 'UP',
    1: 'RIGHT',
    2: 'DOWN',
    3: 'LEFT',
    4: 'WAIT',
    5: 'BOMB'
}

ACTIONS_INV_MAP = {v: k for k, v in ACTINS_MAP.items()}


class BombermanBundle(ABC):
    """
    Abstract base class for Bombermans agents.

    Subclasses should bundle together all components and must construct a QLearningAgent and a Rewarder that are then used to play the game.
    """

    def __init__(self, training: bool = False):
        """
        Initializes all components.

        :param training: bool, optional (default=False), whether the agent is used for training or playing
        """
        self._training_mode = training

    @property
    @abstractmethod
    def _q_agent(self) -> qagents.QLearningAgent:
        """
        Returns the Q-learning agent.

        :return: qagents.QLearningAgent
        """
        pass

    @property
    @abstractmethod
    def _rewarder(self) -> rewarders.Rewarder:
        """
        Returns the rewarder.

        :return: rewarders.Rewarder
        """
        pass

    def act(self, game_state: dict) -> str:
        """
        Returns the action that the agent will take in the current state.

        :param game_state: dict
        :return: str
        """
        return ACTINS_MAP[self._q_agent.get_action(game_state, self._training_mode)]
    
    def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict|None, events: list):
        """
        Updates Q based on the observed reward.

        :param old_game_state: dict
        :param self_action: str
        :param new_game_state: dict
        :param events: list
        """
        self_action = ACTIONS_INV_MAP[self_action]
        reward = self._rewarder.compute_reward_from_events(events)
        self._q_agent.update(old_game_state, self_action, reward, new_game_state)

    def save(self, path: str):
        """
        Saves the agent to a file.

        :param path: str
        """
        with open(path, 'wb') as f:
            pickle.dump(self._state_dict(), f)

    @classmethod
    def load(cls, path: str, training: bool = False) -> BombermanBundle:
        """
        Loads the agent from a file.

        :param path: str
        :param training: bool, optional (default=False), whether the agent is used for training or playing
        :return: BombermanBundle
        """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)
        
        agent = cls(training)
        agent._load_state_dict(state_dict)
        return agent
    
    def _state_dict(self) -> dict:
        """
        Returns the state of the agent as a dictionary.

        :return: dict
        """
        return {
            "q_agent": self._q_agent.state_dict(),
            "rewarder": self._rewarder.state_dict()
        }
    
    def _load_state_dict(self, state_dict: dict):
        """
        Loads the state of the agent from a dictionary.

        :param state_dict: dict
        """
        self._q_agent.load_state_dict(state_dict["q_agent"])
        self._rewarder.load_state_dict(state_dict["rewarder"])


class VectorMLPSimple(BombermanBundle):
    """
    Uses:
    - Standard regression Q-learning
    - Vector neural network
    - MLP
    - Epsilon greedy
    - Random sampling
    - Simple rewards
    """

    def __init__(self, training: bool = False):
        """
        Initializes all components.

        :param training: bool, optional (default=False), whether the agent is used for training or playing
        """
        super().__init__(training)

        # Exploration and exploitation
        ex_ex_handler = policy_modifiers.EpsilonGreedy(0.3)

        # Q-handler

        # Regression model
        transformer = transforms.AllfieldsFlat(WIDTH, HEIGHT)
        neural_net = networks.NeuralNetwork(WIDTH*HEIGHT*8, [512, 256, 128], N_ACTIONS)
        loss = nn.SmoothL1Loss()
        optimizer = optim.RAdam(neural_net.parameters(), lr=0.0001)

        regression_model = regression_models.NeuralNetworkVectorQRM(neural_net, transformer, loss, optimizer, N_ACTIONS, "cpu")

        training_memory_use = training_memory.TrainingMemory(100)

        sampler = samplers.RandomSampler()

        sample_step = (3, 64, sampler)
        sample_round = (1, 256, sampler)

        q_handler = qhandler.RegressionQHandler(regression_model, training_memory_use, N_ACTIONS, sample_step, sample_round)

        # Q Agent
        self._q_agent_ = qagents.SimpleQLearningAgent(q_handler, ex_ex_handler)

        # Rewarder
        self._rewarder_ = rewarders.SimpleRewarder()

    @property
    def _q_agent(self) -> qagents.QLearningAgent:
        return self._q_agent_
    
    @property
    def _rewarder(self) -> rewarders.Rewarder:
        return self._rewarder_