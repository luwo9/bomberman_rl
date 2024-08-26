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
from settings import COLS as WIDTH, ROWS as HEIGHT
from .qsettings import N_ACTIONS, ACTIONS_MAP, ACTIONS_INV_MAP


class BombermanBundle(ABC):
    """
    Abstract base class for Bombermans agents.

    Intended to bundle an agent together with all its necessary components.

    The bundle creates the components and is responsible for:
    - Saving and loading the components (methods _state_dict and _load_state_dict)
    - Assembling them to a q-agent and a rewarder (property _q_agent and _rewarder)

    The q-agent and the rewarder are automatically used to provide the act and game_events_occurred methods to play the game.

    The bundle also keeps track of whether it is in training mode or not. A newly created bundle is always in training mode.
    When loading a bundle one can specify whether loading for training or playing and the mode will be set accordingly.
    This boolean flag is also passed on to _load_state_dict such that only components that are needed for training can be loaded.

    Only when loaded in training mode:
    - The agent will explore in its actions.
    - The save() method will save the state of the agent to a file.
    - A call new_run() indicates that a new training run starts.

    Starting a new run increments the counter _n_training_runs.

    This counter can be used in _load_state_dict to make changes to the components.
    This is useful if, e.g. a training run is completed but now training should be succeded in a changed environment,
    where e.g. the rewarder needs change.

    Finally, the save() method can be called with the final=True flag to indicate that this is the final save that can never be retrained.
    This is passed to _state_dict(), that can then throw away everything only needed for training to rduce file size.
    After this was done the agent is always loaded with training=False.
    """

    def __init__(self):
        """
        Initializes all components.
        """
        self._training_mode = True
        self._n_training_runs = 0
        self._was_finalized = False

    @abstractmethod
    def _state_dict(self, final: bool) -> dict:
        """
        Returns the state of the agent as a dictionary.

        :param final: bool, whether this is the final save that can never be retrained
        :return: dict
        """
        pass

    @abstractmethod
    def _load_state_dict(self, state_dict: dict, training: bool):
        """
        Loads the state of the agent from a dictionary.

        :param state_dict: dict
        :param training: bool, whether the agent is used for training or playing
        """
        pass

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

    def new_run(self):
        """
        If in training mode, indicate a new training run starts.
        """
        if self._training_mode:
            self._n_training_runs += 1

    def act(self, game_state: dict) -> str:
        """
        Returns the action that the agent will take in the current state.

        :param game_state: dict
        :return: str
        """
        return ACTIONS_MAP[self._q_agent.get_action(game_state, self._training_mode)]
    
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

    def save(self, path: str, final: bool = False):
        """
        Saves the agent to a file, if in training mode.

        :param path: str
        :param final: bool, optional (default=False), whether this is the final save that can never be retrained
        """
        if self._training_mode:
            with open(path, 'wb') as f:
                pickle.dump(self._state_dict(final), f)
                self._was_finalized = final

    def load(self, path: str, training: bool = False):
        """
        Loads the agent from a file.

        :param path: str
        :param training: bool, optional (default=False), whether the agent is used for training or playing
        :return: BombermanBundle
        """
        with open(path, 'rb') as f:
            state_dict = pickle.load(f)

        self._training_mode = training
        training = self._training_mode and not self._was_finalized
        self._load_state_dict(state_dict, training)


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
        super().__init__()

        # Exploration and exploitation
        self._ex_ex_handler = policy_modifiers.EpsilonGreedy(0.3)

        # Q-handler

        # Regression model
        self._transformer = transforms.AllfieldsFlat(WIDTH, HEIGHT)
        self._neural_net = networks.NeuralNetwork(WIDTH*HEIGHT*8, [512, 256, 128], N_ACTIONS)
        self._loss = nn.SmoothL1Loss()
        self._optimizer = optim.RAdam(self._neural_net.parameters(), lr=0.0001)

        device_use = "cuda" if torch.cuda.is_available() else "cpu"
        self._regression_model = regression_models.NeuralNetworkVectorQRM(self._neural_net, self._transformer, self._loss, self._optimizer, N_ACTIONS, device_use)

        self._training_memory_use = training_memory.TrainingMemory(100)

        self._sampler = samplers.RandomSampler()

        sample_step = (3, 64, self._sampler)
        sample_round = (1, 256, self._sampler)

        self._q_handler = qhandler.RegressionQHandler(self._regression_model, self._training_memory_use, N_ACTIONS, sample_step, sample_round)

        # Q Agent
        self._q_agent_ = qagents.SimpleQLearningAgent(self._q_handler, self._ex_ex_handler, N_ACTIONS)

        # Rewarder
        self._rewarder_ = rewarders.SimpleRewarder()

    @property
    def _q_agent(self) -> qagents.QLearningAgent:
        return self._q_agent_
    
    @property
    def _rewarder(self) -> rewarders.Rewarder:
        return self._rewarder_
    
    def _state_dict(self, final: bool) -> dict:
        # Save all components manually
        out_dict = {
                "training_mode": self._training_mode,
                "n_training_runs": self._n_training_runs,
                "was_finalized": self._was_finalized,
                "ex_ex_handler": self._ex_ex_handler.state_dict(),
                "transformer": self._transformer.state_dict(),
                "neural_net": self._neural_net.state_dict(),
                "regression_model": self._regression_model.state_dict(),
                "q_agent": self._q_agent_.state_dict(),
                "q_handler": self._q_handler.state_dict(),
                "rewarder": self._rewarder_.state_dict()
            }
        
        if not final:
            out_dict.update({
                "loss": self._loss.state_dict(),
                "optimizer": self._optimizer.state_dict(),
                "training_memory_use": self._training_memory_use.state_dict(),
                "sampler": self._sampler.state_dict()
            })

        return out_dict
        
    def _load_state_dict(self, state_dict: dict, training: bool):
        # Load all components manually
        self._training_mode = state_dict["training_mode"]
        self._n_training_runs = state_dict["n_training_runs"]
        self._was_finalized = state_dict["was_finalized"]
        if training:
            self._loss.load_state_dict(state_dict["loss"])
            self._optimizer.load_state_dict(state_dict["optimizer"])
            self._training_memory_use.load_state_dict(state_dict["training_memory_use"])
            self._sampler.load_state_dict(state_dict["sampler"])
            self._rewarder_.load_state_dict(state_dict["rewarder"])
        
        # Always loaded
        self._ex_ex_handler.load_state_dict(state_dict["ex_ex_handler"])
        self._transformer.load_state_dict(state_dict["transformer"])
        self._neural_net.load_state_dict(state_dict["neural_net"])
        self._regression_model.load_state_dict(state_dict["regression_model"])
        self._q_agent_.load_state_dict(state_dict["q_agent"])
        self._q_handler.load_state_dict(state_dict["q_handler"])

        # if self._n_training_runs > 0:
        #     self._rewarder_ = rewarders.TemplateRewarder()