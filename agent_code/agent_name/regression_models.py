"""
Contains different regression models that can be used to approximate the Q-function.
"""
from abc import ABC, abstractmethod
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .transforms import Transform

# This should maybe be reworked to be a wrapper arround a regression model, that only knows about x and y, prediction and loss (?)
# It will hold on to a transformer in any case
class QRegressionModel(ABC):
    """
    Abstract base class for Q-function regression models.
    """

    @abstractmethod
    def predict(self, states, actions):
        """
        Predicts the value of the Q-function for the given states and actions.

        :param states: dict or list of dicts
        :param actions: int or np.ndarray[int]
        :return: np.ndarray

        .. Note::
            Different input formats are supported for convenience. If states is a dict, actions can be int or np.ndarray[int].
            In case of int the Q-value for that action is returned. In case of np.ndarray[int] the Q-values for all actions are returned.
            If states is a list of dicts, actions must be an np.ndarray[int]. The array length must be the same as the number of states.
            If the array is 1D an array of pairwise Q-values is returned. Otherwise a list of 1D arrays must be given, where each array contains the actions for the corresponding state.
            A list is then returned with the Q-values for each state.
            Also a 2D array can be given, if the actions for all states are of equal length. In this case a 2D array is returned with the Q-values for all states.
        """
        pass

    @abstractmethod
    def update(self, states, actions, targets):
        """
        Updates the model based on the given states, actions, and targets.

        :param states: dict or list of dicts
        :param actions: int or np.ndarray[int]
        :param targets: np.ndarray

        .. Note::
            See predict for the input formats. However, several actions per state are not supported.
        """
        pass

    @abstractmethod
    def state_dict(self):
        """
        Returns the state of the model as a dictionary.

        :return: dict
        """
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """
        Loads the state of the model from a dictionary.

        :param state_dict: dict
        """
        pass

    @property
    @abstractmethod
    def actions(self):
        """
        Returns the number of actions.

        :return: int
        """
        pass


class DoubleQRegressionModel(QRegressionModel):
    """
    Uses two different Q-function regression models internally.
    """

    @abstractmethod
    def switch(self):
        """
        Switches between the active and inactive model.
        """
        pass
    
    @abstractmethod
    def sync(self):
        """
        Synchronizes/updates the inactive model with the active model.
        """
        pass


def batched_nn_eval(neural_network, x, batch_size, eval_device):
    """
    Evaluates the neural network in batches. The main purpose is to avoid gpu memory errors.

    :param neural_network: callable
    :param x: torch.Tensor
    :param batch_size: int, batch size the neural network should be evaluated with
    :param eval_device: str, device on which to evaluate the neural network
    """
    # This should be safe to use for gradient computation
    if len(x) <= batch_size:
        return neural_network(x.to(device=eval_device)).to(device=x.device)
    
    batches = torch.split(x, batch_size)
    idx_at = 0
    for i, batch in enumerate(batches):

        batch = batch.to(device=eval_device)
        out = neural_network(batch).to(device=x.device)

        # Allow for lazy determination of the output shape
        if i == 0:
            y = torch.zeros(len(x), *out.shape[1:], device=x.device)

        y[idx_at:idx_at + len(batch)] = out
        idx_at += len(batch)

    return y


class NeuralNetworkVectorQRM(QRegressionModel):
    """
    Uses a neural network to approximate the Q-function.
    It uses the 'vector' approach, where the input is the state and the ouput is a vector of Q-values for each action.
    """

    def __init__(self, neural_network: nn.Module, transformer: Transform, loss_function: nn.Module, optimizer: optim.Optimizer, number_of_actions: int,
                  device: str, chunk_size: int = 10**7, lr_scheduler: optim.lr_scheduler.LRScheduler = None):
        """
        Initialize the model.

        :param neural_network: torch.nn.Module
        :param transformer: Transform
        :param loss_function: torch.nn.Module
        :param optimizer: torch.optim.Optimizer
        :param number_of_actions: int
        :param device: str
        :param chunk_size: int, default: 10**7, maximum number of points that can be evaluated at once

        """
        self._neural_network = neural_network
        self._transformer = transformer

        self._loss_function = loss_function
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler

        self._device = device
        self._chunk_size = chunk_size

        self._actions = np.arange(number_of_actions)

        self._neural_network.to(device=device)

    def predict(self, states, actions):
        """
        Predicts the value of the Q-function for the given states and actions.

        :param states: dict or list of dicts
        :param actions: int or np.ndarray[int]
        :return: np.ndarray

        .. Note::
            Different input formats are supported for convenience. If states is a dict, actions can be int or np.ndarray[int].
            In case of int the Q-value for that action is returned. In case of np.ndarray[int] the Q-values for all actions are returned.
            If states is a list of dicts, actions must be an np.ndarray[int]. The array length must be the same as the number of states.
            If the array is 1D an array of pairwise Q-values is returned. Otherwise a list of 1D arrays must be given, where each array contains the actions for the corresponding state.
            A list is then returned with the Q-values for each state.
            Also a 2D array can be given, if the actions for all states are of equal length. In this case a 2D array is returned with the Q-values for all states.
        """
        with torch.inference_mode():
            all_Q = self._compute_Q(states)
        
        # Match the input format
        if isinstance(states, dict):
            if isinstance(actions, int):
                return all_Q[0, actions].item()
            return all_Q[0, actions].numpy()
        
        if isinstance(actions, list):
            all_Q = all_Q.numpy()
            out = [all_Q[i, actions_i] for i, actions_i in enumerate(actions)]
            return out
        
        actions = torch.tensor(actions, dtype=torch.int64)
        if actions.ndim == 1:
            actions = actions.unsqueeze(1)
        return torch.gather(all_Q, 1, actions).squeeze().numpy()
    
    def update(self, states, actions, targets):
        """
        Updates the model based on the given states, actions, and targets.

        :param states: dict or list of dicts
        :param actions: int or np.ndarray[int]
        :param targets: np.ndarray

        .. Note::
        See predict for the input formats. However, several actions per state are not supported.
        """
        actions = torch.tensor(actions, dtype=torch.int64)
        targets = torch.tensor(targets, dtype=torch.float32)
        targets = targets.reshape(-1, 1)

        # Predict and Compute the loss
        all_Q = self._compute_Q(states)
        q_s_a = torch.gather(all_Q, 1, actions.reshape(-1, 1))
        loss = self._loss_function(q_s_a, targets)

        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()


    def state_dict(self):
        """
        Returns the state of the model as a dictionary.

        :return: dict
        """
        return {}
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the model from a dictionary.

        :param state_dict: dict
        """
        pass

    @property
    def actions(self):
        """
        Returns the actions.

        :return: int
        """
        return self._actions

    def _compute_Q(self, states, keep_bijection=True):
        """
        Predicts the value of the Q-function for the given states for all actions.

        :param states: dict or list of dicts
        :param actions: int or np.ndarray[int]
        :keep_bijection: bool, default: True, as passed to the transformer
        :return: np.ndarray

        .. Note::
            
        """
        if isinstance(states, dict):
            states = [states]
            
        states = self._transformer.transform(states, keep_bijection=keep_bijection)
        states = torch.tensor(states, dtype=torch.float32)

        # Handle terminal states
        check_axis = tuple(range(1, states.ndim))
        is_terminal = torch.isnan(states).all(dim=check_axis) # Maybe find some better way to handle this as nans can occur otherwise
        non_terminal_states = states[~is_terminal]

        # shape: (batch_size, number_of_actions) (it is a vector model)
        y_non_terminal = batched_nn_eval(self._neural_network, non_terminal_states, self._chunk_size, self._device)

        # Handle terminal states
        y_full = torch.zeros(len(states), y_non_terminal.shape[1])
        y_full[~is_terminal] = y_non_terminal

        return y_full
    

class DoubleNeuralNetworkVectorQRM(DoubleQRegressionModel, NeuralNetworkVectorQRM):
    """
    Same as NeuralNetworkVectorQRM, but uses two neural networks internally, see DoubleQRegressionModel.
    """

    def __init__(self, neural_network: nn.Module, transformer: Transform, loss_function: nn.Module, optimizer: optim.Optimizer, number_of_actions: int,
                  device: str, chunk_size: int = 10**7, lr_scheduler: optim.lr_scheduler.LRScheduler = None, tau: float = 1):
        """
        Initialize the model.

        :param neural_network1: torch.nn.Module
        :param transformer: Transform
        :param loss_function: torch.nn.Module
        :param optimizer: torch.optim.Optimizer
        :param number_of_actions: int
        :param device: str
        :param chunk_size: int, default: 10**7, maximum number of points that can be evaluated at once

        """
        super().__init__(neural_network, transformer, loss_function, optimizer, number_of_actions, device, chunk_size, lr_scheduler)
        self._tau = tau
        # Copy the neural network
        self._inactive_neural_network = copy.deepcopy(neural_network)

    def switch(self):
        """
        Switches between the active and inactive model.
        """
        self._neural_network, self._inactive_neural_network = self._inactive_neural_network, self._neural_network

    def sync(self):
        """
        Synchronizes/updates the inactive model with the active model.
        """
        # Soft update
        inactive_state_dict = self._inactive_neural_network.state_dict()
        active_state_dict = self._neural_network.state_dict()
        for key in active_state_dict:
            inactive_state_dict[key] = self._tau * active_state_dict[key] + (1 - self._tau) * inactive_state_dict[key]
        self._inactive_neural_network.load_state_dict(inactive_state_dict)

    def state_dict(self):
        """
        Returns the state of the model as a dictionary.

        :return: dict
        """
        state_dict = super().state_dict()
        # The inactive network is safed, as it is an internal state and thus the responsibility of this class
        state_dict['inactive_neural_network'] = self._inactive_neural_network.state_dict()
        return state_dict
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the model from a dictionary.

        :param state_dict: dict
        """
        super().load_state_dict(state_dict)
        self._inactive_neural_network.load_state_dict(state_dict['inactive_neural_network'])