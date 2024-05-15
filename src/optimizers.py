from abc import ABC, abstractmethod
from typing import override

import numpy as np


class Optimizer(ABC):
    """Abstract base class for optimizers."""

    def __init__(self, oracle: object, N: int, C: float):
        """
        Parameters:
        oracle: Oracle object to compute gradients.
        N (int): Number of iterations.
        C (float): Learning rate, controlling the iteration step.
        """
        self.oracle = oracle
        self.N = N
        self.C = C
        self.p = oracle.initialize_prices()
        self.S = oracle.S
        self.D = oracle.D

    @abstractmethod
    def update(self):
        """
        Updates prices of products. This method should be overridden by subclasses.
        Returns:
            p_mean (float): Mean of prices.
        """
        pass

    def get_prices(self):
        """
        Returns optimized prices of products.
        """
        return self.p


class SGD(Optimizer):
    """Stochastic Gradient Descent."""

    @override
    def update(self):
        """
        Updates prices of products using SGD.
        Returns:
            p_mean (float): Mean of prices.
        """
        p_mean = self.p
        for t in range(self.N):
            index = np.random.randint(1, self.S + self.D)
            self.p = self.p - (self.C / np.sqrt(t + 1)) * self.oracle.compute_gradient(
                self.p, index
            ).clip(min=0)
            p_mean += self.p
        return p_mean / self.N


class AdaGrad(Optimizer):
    """Adaptive Gradient."""

    @override
    def update(self):
        """
        Updates prices of products using AdaGrad.
        Returns:
            p_mean (float): Mean of prices.
        """
        p_mean = self.p
        H = np.zeros(self.S + self.D)
        for _ in range(self.N):
            index = np.random.randint(1, self.S + self.D)
            g = self.oracle.compute_gradient(self.p, index)
            H += g**2
            self.p = (self.p - self.C / np.sqrt(H + 1e-7) * g).clip(min=0)
            p_mean += self.p
        return p_mean / self.N


class Momentum(Optimizer):
    """Momentum."""

    def __init__(self, oracle: object, N: int, C: float, gamma: float = 0.9):
        """
        Parameters:
        gamma (float): Momentum parameter.
        """
        super().__init__(oracle, N, C)
        self.gamma = gamma

    @override
    def update(self):
        """
        Updates prices of products using Momentum.
        Returns:
            p_mean (float): Mean of prices.
        """
        p_mean = self.p
        v = np.zeros(self.S + self.D)
        for _ in range(self.N):
            index = np.random.randint(1, self.S + self.D)
            g = self.oracle.compute_gradient(self.p, index)
            v = self.gamma * v + self.C * g
            self.p = (self.p - v).clip(min=0)
            p_mean += self.p
        return p_mean / self.N


class RMSprop(Optimizer):
    """Root Mean Square Propagation."""

    def __init__(self, oracle: object, N: int, C: float, decay_rate: float = 0.9):
        """
        Parameters:
        decay_rate (float): Decay rate parameter for RMSprop.
        """
        super().__init__(oracle, N, C)
        self.decay_rate = decay_rate

    @override
    def update(self):
        """
        Updates prices of products using RMSprop.
        Returns:
            p_mean (float): Mean of prices.
        """
        p_mean = self.p
        H = np.zeros(self.S + self.D)
        for _ in range(self.N):
            index = np.random.randint(1, self.S + self.D)
            g = self.oracle.compute_gradient(self.p, index)
            H = self.decay_rate * H + (1 - self.decay_rate) * g**2
            self.p = (self.p - self.C / np.sqrt(H + 1e-7) * g).clip(min=0)
            p_mean += self.p
        return p_mean / self.N


class ADAM:
    """Adaptive Moment Estimation."""

    def __init__(
        self, oracle: object, N: int, C: float, beta1: float = 0.9, beta2: float = 0.999
    ):
        """
        Parameters:
        beta1 (float): Decay rate of the first moment.
        beta2 (float): Decay rate of the second moment.
        """
        self.oracle = oracle
        self.N = N
        self.C = C
        self.p = oracle.initialize_prices()
        self.S = oracle.S
        self.D = oracle.D
        self.beta1 = beta1
        self.beta2 = beta2

    @override
    def update(self):
        """
        Updates prices of products using ADAM.
        Returns:
            p_mean (float): Mean of prices.
        """
        p_mean = self.p
        m = np.zeros(self.S + self.D)
        v = np.zeros(self.S + self.D)
        for t in range(self.N):
            index = np.random.randint(1, self.S + self.D)
            g = self.oracle.compute_gradient(self.p, index)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1 ** (t + 1))
            v_hat = v / (1 - self.beta2 ** (t + 1))
            self.p = (self.p - self.C / np.sqrt(v_hat + 1e-7) * m_hat).clip(min=0)
            p_mean += self.p
        return p_mean / self.N
