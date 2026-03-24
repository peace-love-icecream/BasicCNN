from Layers.Base import BaseLayer
import numpy as np
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.bias = None
        self.weights = None
        self.mean = None
        self.sigma = None
        self.input = None
        self._moving_mean = 0
        self._moving_sigma = 0
        self.alpha = 0.8
        self._gradient_weights = 0
        self._gradient_bias = 0
        self.normalized_input = None
        self.initialize()

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    @property
    def moving_mean(self):
        return self._moving_mean

    @moving_mean.setter
    def moving_mean(self, value):
        self._moving_mean = value

    @property
    def moving_sigma(self):
        return self._moving_sigma

    @moving_sigma.setter
    def moving_sigma(self, value):
        self._moving_sigma = value

    def forward(self, input_tensor):
        self.input = input_tensor
        mean_b = np.mean(input_tensor, axis=0)
        sigma_b = np.var(input_tensor, axis=0)
        self.moving_mean = mean_b
        self.moving_sigma = sigma_b

        if self.testing_phase:
            normalized_input_tensor = (input_tensor - self.moving_mean) / (np.sqrt(self.moving_sigma + np.finfo(float).eps))
            return self.weights * normalized_input_tensor + self.bias

        self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * mean_b
        self.moving_sigma = self.alpha * self.moving_sigma + (1 - self.alpha) * sigma_b
        self.mean = mean_b
        self.sigma = sigma_b

        normalized_input_tensor = (input_tensor - mean_b) / (np.sqrt(sigma_b + np.finfo(float).eps))
        self.normalized_input = normalized_input_tensor
        return self.weights * normalized_input_tensor + self.bias

    def backward(self, error_tensor):
        # gradient with respect to W
        for batch in range(0, error_tensor.shape[1] + 1):
            self.gradient_weights += error_tensor[batch] * self.normalized_input[batch]
            self.gradient_bias += error_tensor[batch]


        # gradient with respect to X
        return compute_bn_gradients(error_tensor, self.input, self.weights, self.mean, self.sigma)

    def initialize(self):
        self.bias = np.zeros(self.channels)
        self.weights = np.ones(self.channels)
