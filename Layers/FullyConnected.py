import Optimization.Optimizers
from Layers.Base import BaseLayer
import numpy as np
import random
from Layers.Initializers import *


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self._optimizer = None
        self._input = None
        self._gradient_weights = None
        self.input_size = input_size
        self.output_size = output_size
        #self.weights = np.random.uniform(0, 1, (self.input_size + 1, self.output_size))
        self.weights = np.ones((self.input_size + 1, self.output_size))
        self.initialize(UniformRandom(), UniformRandom())

    @property
    def optimizer(self):
        """optimizer property"""
        return self._optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    def forward(self, input_tensor):
        # add column of ones to input_tensor
        input_tensor = np.append(input_tensor, np.ones(input_tensor.shape[0], dtype=int)[np.newaxis].T, axis=1)
        # save input for backward()
        self.input = input_tensor

        # W * X = Y
        # (batch_size x input_size) * (input_size x output_size) = batch_size x output_size
        return np.dot(input_tensor, self.weights)

    def backward(self, error_tensor):
        # gradient with respect to X
        output = np.dot(error_tensor, self.weights.T)
        output = np.delete(output, -1, 1)

        # gradient with respect to W
        self.gradient_weights = np.dot(self.input.T, error_tensor)

        if self.optimizer:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)

        return output

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(self.weights.shape, self.input_size, self.output_size)
        # since bias is part of weights variable, bias.shape = weights[-1, :].shape
        self.weights[-1, :] = bias_initializer.initialize(self.weights[-1, :].shape, 1, self.output_size)
