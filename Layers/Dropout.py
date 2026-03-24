import numpy.random
from Layers.Base import BaseLayer
import numpy as np

class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.act_binary = None

    def forward(self, input_tensor):
        if not self.testing_phase:
            act_bin = np.random.rand(input_tensor.shape[0], input_tensor.shape[1]) < self.probability
            self.act_binary = act_bin
            res = input_tensor * act_bin
            return res / self.probability
        return input_tensor

    def backward(self, error_tensor):
        return (error_tensor * self.act_binary) / self.probability
