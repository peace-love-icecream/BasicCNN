import numpy as np


class Optimizer:
    """Optimizer"""
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Optimizer):
    """Optimizer SGD"""
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer:
            return weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) - self.learning_rate * gradient_tensor

        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.old_momentum = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        current_momentum = self.momentum_rate * self.old_momentum - self.learning_rate * gradient_tensor
        self.old_momentum = current_momentum

        if self.regularizer:
            return weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) + current_momentum

        return weight_tensor + current_momentum


class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.old_momentum1 = 0
        self.old_momentum2 = 0
        self.iteration = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        current_momentum1 = self.mu * self.old_momentum1 + (1 - self.mu) * gradient_tensor
        self.old_momentum1 = current_momentum1

        current_momentum2 = self.rho * self.old_momentum2 + (1 - self.rho) * np.square(gradient_tensor)
        self.old_momentum2 = current_momentum2

        cur_mom1_biascor = current_momentum1 / (1 - np.power(self.mu, self.iteration))
        cur_mom2_biascor = current_momentum2 / (1 - np.power(self.rho, self.iteration))

        self.iteration += 1

        if self.regularizer:
            return weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor)\
                   - self.learning_rate * cur_mom1_biascor / (np.sqrt(cur_mom2_biascor) + np.finfo(float).eps)

        return weight_tensor - self.learning_rate * cur_mom1_biascor / (np.sqrt(cur_mom2_biascor) + np.finfo(float).eps)