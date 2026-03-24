import numpy as np


class Constant:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.ones(weights_shape) * self.constant


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, weights_shape)

# eventuell weights_shape mit (fan_in, fan_out) ersetzen
class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, sigma, weights_shape)


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2 / fan_in)
        return np.random.normal(0, sigma, weights_shape)
