from Layers.Base import BaseLayer


class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self._input = None

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    def forward(self, input_tensor):
        self.input = input_tensor
        output_tensor = input_tensor.copy()
        output_tensor[output_tensor < 0] = 0
        return output_tensor

    def backward(self, error_tensor):
        gradient = error_tensor.copy()
        gradient[self.input <= 0] = 0
        return gradient
