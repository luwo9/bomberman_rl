"""
Contains different neural network architectures that can be used to approximate the Q-function.
"""
import itertools
from typing import Mapping
from typing import Tuple

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    A simple feedforward neural network.
    """

    def __init__(self, input_size, hidden_sizes: list[int], output_size):
        """
        Initialize the neural network.

        :param input_size: int
        :param hidden_sizes: list of int
        :param output_size: int
        """
        super().__init__()

        self.register_buffer('_input_size', torch.tensor(input_size))
        self.register_buffer('_output_size', torch.tensor(output_size))
        self.register_buffer('_hidden_sizes', torch.tensor(hidden_sizes, dtype=torch.int))

        self._layers = []
        for hidden_size in hidden_sizes:
            self._layers.append(nn.Linear(input_size, hidden_size))
            self._layers.append(nn.ReLU())
            input_size = hidden_size
        self._layers.append(nn.Linear(input_size, output_size))

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        """
        Forward pass.

        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return self._layers(x)
    
    @property
    def device(self):
        return next(self.parameters()).device
    

def compute_convolution_output_size(H, W, kernel_size, stride, padding, dilation):
    """
    Computes the output size of a convolutional layer.

    :param H: int
    :param W: int
    :param kernel_size: int
    :param stride: int
    :param padding: int
    :param dilation: int
    :return: tuple[int]
    """
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
        stride = (stride, stride)
        padding = (padding, padding)
        dilation = (dilation, dilation)

    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1

    return H_out, W_out


class SimpleCNN(nn.Module):
    """
    Convolutional neural network.
    """

    def __init__(self, kernel_sizes: list[int], paddings: list[int], channels: list[int], input_size: Tuple[int, int] = None):
        """
        Initialize the neural network.

        :param kernel_sizes: list of int
        :param paddings: list of int
        :param channels: list of int
        :param input_size: tuple of int, H x W if supplied the output size will be computed
        """
        super().__init__()

        self.register_buffer('_kernel_sizes', torch.tensor(kernel_sizes, dtype=torch.int))
        self._paddings = paddings # As this might be e.g. a string
        self.register_buffer('_channels', torch.tensor(channels, dtype=torch.int))
        self.register_buffer('_input_size', torch.tensor(input_size, dtype=torch.int) if input_size is not None else None)

        if input_size is not None:
            H, W = input_size
        else:
            H, W = None, None

        channels_out = channels[1:]

        self._layers = []
        for kernel_size, padding, in_channels, out_channels in zip(kernel_sizes, paddings, channels, channels_out):
            if H is not None:
                H, W = compute_convolution_output_size(H, W, kernel_size, stride=1, padding=padding, dilation=1)
            self._layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding))
            self._layers.append(nn.ReLU())

        self._out_size = (H, W, channels_out[-1])

        self._layers = nn.Sequential(*self._layers)

    def forward(self, x):
        """
        Forward pass.

        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return self._layers(x)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def output_size(self):
        return self._out_size
    
    def state_dict(self):
        """
        Returns the state of the model as a dictionary.

        :return: dict
        """
        super_dict = super().state_dict()
        super_dict["paddings"] = self._paddings
        return super_dict
    
    def load_state_dict(self, state_dict):
        """
        Loads the state of the model from a dictionary.

        :param state_dict: dict
        """
        self._paddings = state_dict.pop("paddings")
        super().load_state_dict(state_dict)