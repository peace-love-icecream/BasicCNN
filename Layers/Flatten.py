from Layers.Base import BaseLayer


class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.prev_shape = None

    def forward(self, input_tensor):
        self.prev_shape = input_tensor.shape
        return input_tensor.reshape((input_tensor.shape[0], -1))

    def backward(self, error_tensor):
        return error_tensor.reshape(self.prev_shape)
