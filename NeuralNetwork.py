import copy
import numpy as np


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self._tmp_label_tensor = None
        self.phase = None

    @property
    def phase(self):
        return self.phase

    @phase.setter
    def phase(self, value):
        self.phase = value

    @property
    def tmp_label_tensor(self):
        return self._tmp_label_tensor

    @tmp_label_tensor.setter
    def tmp_label_tensor(self, value):
        self._tmp_label_tensor = value

    def forward(self):
        reg_loss = 0
        input_tensor, label_tensor = self.data_layer.next()
        self.tmp_label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
            if self.optimizer.regularizer:
                if layer.trainable:
                    reg_loss += self.optimizer.regularizer.norm(layer.weights)

        output = self.loss_layer.forward(input_tensor, label_tensor)
        #if self.optimizer.regularizer:
        #    reg_loss += self.optimizer.regularizer.norm(self.loss_layer.weights)

        return output + reg_loss

    def backward(self):
        error_tensor = self.loss_layer.backward(self.tmp_label_tensor)
        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            optimizer_copy = copy.deepcopy(self.optimizer)
            layer.optimizer = optimizer_copy
            layer.weights = layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(0, iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self, input_tensor):
        input_tensor_copy = np.copy(input_tensor)
        for layer in self.layers:
            input_tensor_copy = layer.forward(input_tensor_copy)
        return input_tensor_copy
