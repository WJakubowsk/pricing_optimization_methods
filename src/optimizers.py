from abc import ABC, abstractmethod
from typing_extensions import override

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
        self.S = oracle.s
        self.D = oracle.d

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
        results = []
        p_mean = self.p
        for t in range(self.N):
            index = np.random.randint(0, self.S + self.D)
            self.p = (
                self.p - (self.C / np.sqrt(t + 1)) * self.oracle.compute_gradient(index)
            ).clip(min=0)
            self.oracle.prices = self.p
            print("prices: ", self.p)
            print("cost function: ", self.oracle.compute_cost_func(self.oracle.prices))
            results.append(self.oracle.compute_cost_func(self.oracle.prices))
            print("gradient: ", self.oracle.compute_gradient(index))
            p_mean += self.p
        return results


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
        H = np.zeros(self.oracle.n)
        results = []
        for _ in range(self.N):
            index = np.random.randint(0, self.S + self.D)
            g = self.oracle.compute_gradient(index)
            H += g**2
            self.p = (self.p - self.C / np.sqrt(H + 1e-7) * g).clip(min=0)
            self.oracle.prices = self.p
            print("prices: ", self.p)
            print("cost function: ", self.oracle.compute_cost_func(self.oracle.prices))
            results.append(self.oracle.compute_cost_func(self.oracle.prices))
            print("gradient: ", self.oracle.compute_gradient(index))
            p_mean += self.p
        return results


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
        results = []
        p_mean = self.p
        v = np.zeros(self.oracle.n)
        for _ in range(self.N):
            index = np.random.randint(0, self.S + self.D)
            g = self.oracle.compute_gradient(index)
            v = self.gamma * v + self.C * g
            self.p = (self.p - v).clip(min=0)
            self.oracle.prices = self.p
            print("prices: ", self.p)
            print("cost function: ", self.oracle.compute_cost_func(self.oracle.prices))
            results.append(self.oracle.compute_cost_func(self.oracle.prices))
            print("gradient: ", self.oracle.compute_gradient(index))
            p_mean += self.p
        return results


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
        results = []
        p_mean = self.p
        H = np.zeros(self.oracle.n)
        for _ in range(self.N):
            index = np.random.randint(0, self.S + self.D)
            g = self.oracle.compute_gradient(index)
            H = self.decay_rate * H + (1 - self.decay_rate) * g**2
            self.p = (self.p - self.C / np.sqrt(H + 1e-7) * g).clip(min=0)
            self.oracle.prices = self.p
            print("prices: ", self.p)
            print("cost function: ", self.oracle.compute_cost_func(self.oracle.prices))
            results.append(self.oracle.compute_cost_func(self.oracle.prices))
            print("gradient: ", self.oracle.compute_gradient(index))
            p_mean += self.p
        return results


class ADAM(Optimizer):
    """Adaptive Moment Estimation."""

    def __init__(
        self, oracle: object, N: int, C: float, beta1: float = 0.9, beta2: float = 0.999
    ):
        """
        Parameters:
        beta1 (float): Decay rate of the first moment.
        beta2 (float): Decay rate of the second moment.
        """
        super().__init__(oracle, N, C)
        self.beta1 = beta1
        self.beta2 = beta2

    @override
    def update(self):
        """
        Updates prices of products using ADAM.
        Returns:
            p_mean (float): Mean of prices.
        """
        results = []
        p_mean = self.p
        m = np.zeros(self.oracle.n)
        v = np.zeros(self.oracle.n)
        for t in range(self.N):
            index = np.random.randint(0, self.S + self.D)
            g = self.oracle.compute_gradient(index)
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2
            m_hat = m / (1 - self.beta1 ** (t + 1))
            v_hat = v / (1 - self.beta2 ** (t + 1))
            self.p = (self.p - self.C / np.sqrt(v_hat + 1e-7) * m_hat).clip(min=0)
            self.oracle.prices = self.p
            print("prices: ", self.p)
            print("cost function: ", self.oracle.compute_cost_func(self.oracle.prices))
            results.append(self.oracle.compute_cost_func(self.oracle.prices))
            print("gradient: ", self.oracle.compute_gradient(index))
            p_mean += self.p
        return results
