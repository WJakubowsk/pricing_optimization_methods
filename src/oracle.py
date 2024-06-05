import numpy as np

class Oracle:
    def __init__(self, s = 5, d = 10, n = 20, m = 5, gamma = 0.0001):
        self.s = s
        self.d = d
        self.n = n
        self.m = m
        self.gamma = gamma

    def initialize_prices(self):
        self.prices = np.random.uniform(0.01, 5, self.n)
        self.prices = self.prices / np.max(self.prices)
        return self.prices
    
    def initialize_customers(self):
        self.customers = np.zeros((self.n, self.d))
        for i in range(self.n):
            for j in range(self.d):
                segment = i % self.m
                self.customers[i,j] = np.random.uniform(segment/self.m, (segment+1)/self.m)
        return self.customers
    
    def compute_y(self):
        self.y_dash = np.random.uniform(0.01, 2, (self.s, self.n))
        self.y = (self.prices + 2*self.gamma*self.y_dash) / (2+2*self.gamma)
        return self.y

    def compute_cost_func(self):
        # generate s vectors y_^_s of size n from uniform distribution [0.01, 2]
        self.compute_y()
        # c is squared second norm of y 
        self.c = np.sum(self.y**2, axis=1)
        self.pi = np.sum(self.y * self.prices, axis=1) - self.c - self.gamma*np.sum((self.y-self.y_dash)**2, axis=1)
        # compute expected surplus
        self.mu = np.random.uniform(0.1, 1, self.m)
        # define segments G
        self.G = [[] for _ in range(self.m)]
        for i in range(self.n):
            segment = i % self.m
            self.G[segment].append(i)
        self.expected_surplus = np.zeros(self.d)
        for d in range(self.d):
            result = 0
            for j in range(self.m):
                inner_sum = 0
                for i in self.G[j]:
                    exp_term = np.exp((self.customers[i] - self.prices[i]) / self.mu[j])
                    inner_sum += np.sum(exp_term) ** self.mu[j]
                result += inner_sum
            self.expected_surplus[d] = np.log(result)
        # compute target function as sum of s 
        self.cost_func = np.sum(self.pi) + np.sum(self.expected_surplus)
        return self.cost_func
    
    def compute_probabilities(self):
        self.probabilities = np.zeros((self.n, self.d))

        denominator = np.zeros(self.d)
        for i in range(self.d):
            outer_sum = 0
            for h in range(self.m):
                inner_sum = 0
                for k in self.G[h]:
                    inner_sum += np.exp((self.customers[k, i] - self.prices[k]) / self.mu[h])
                outer_sum += inner_sum ** self.mu[h]
            denominator[i] = outer_sum

        # numerator = np.zeros((self.n, self.d))
        for i in range(self.n):
            for j in range(self.d):
                sum = 0
                for k in self.G[i % self.m]:
                    sum += np.exp((self.customers[k, j] - self.prices[k]) / self.mu[i % self.m])
                # numerator[i, j] = np.exp((self.customers[i, j] - self.prices[i]) / self.mu[i % self.m]) * sum ** self.mu[i % self.m]
                numerator = np.exp((self.customers[i, j] - self.prices[i]) / self.mu[i % self.m]) * (sum ** (self.mu[i % self.m] - 1))
                self.probabilities[i, j] = numerator / denominator[j]
        # self.probabilities = numerator / denominator
        return self.probabilities

    def compute_gradient(self, index):
        if index < self.s:
            self.compute_y()
            return self.y[index]
        if index < self.s + self.d:
            self.compute_probabilities()
            return np.array([-1 if np.random.uniform() < self.probabilities[index, index - self.s] else 0 for _ in range(self.n)])
    

