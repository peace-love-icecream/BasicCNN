from Layers.Base import BaseLayer
import numpy as np


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self._output = None

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    def softmax(self, x):
        # shift x for numerical stability
        x = x - np.max(x)
        return np.exp(x) / np.exp(x).sum(axis=0)

    def forward(self, input_tensor):
        # calculate softmax probabilities row wise (batch wise)
        output = np.apply_along_axis(self.softmax, 1, input_tensor)
        # safe output
        self.output = output
        return output

    def backward(self, error_tensor):
        error = self.output * (error_tensor.T - (error_tensor * self.output).sum(axis=1)).T
        return error
