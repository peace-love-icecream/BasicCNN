import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self._prediction = None

    @property
    def prediction(self):
        return self._prediction

    @prediction.setter
    def prediction(self, value):
        self._prediction = value

    def forward(self, prediction_tensor, label_tensor):
        loss = (np.log(prediction_tensor[label_tensor == 1] + np.finfo(float).eps) * (-1)).sum()
        self.prediction = prediction_tensor
        return loss

    def backward(self, label_tensor):
        return label_tensor * (-1) / (self.prediction + np.finfo(float).eps)