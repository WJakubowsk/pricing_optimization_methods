import numpy as np


class Oracle:
    def __init__(self, s=5, d=10, n=20, m=5, gamma=0.0001):
        self.s = s
        self.d = d
        self.n = n
        self.m = m
        self.gamma = gamma
        self.y_dash = np.random.uniform(0.01, 2, (self.s, self.n))
        # compute expected surplus
        self.mu = np.random.uniform(0.1, 1, self.m)
        self.initialize_prices()
        self.initialize_customers()
        # define segments G
        self.G = [[] for _ in range(self.m)]
        for i in range(self.n):
            segment = i % self.m
            self.G[segment].append(i)
        self.final_price = self.epku()

    def epku2(self, p, h, N):
        sum_y = np.sum(
            (p + 2 * self.gamma * self.y_dash) / (2 + 2 * self.gamma), axis=0
        )
        # sum_x = np.sum([N[i] * self.compute_probabilities() for i in range(self.d)], axis=1)

        sum_x = self.compute_probabilities()
        for i in range(self.d):
            sum_x[:, i] += N[i] * self.compute_probabilities()[:, i]
        sum_x = np.sum(sum_x, axis=1)
        p_new = p - h * (sum_y - sum_x)
        return np.maximum(p_new, 0)

    def epku(self):
        p = self.prices
        N = np.random.randint(1, 10, self.d)  # Random N values for consumers
        beta = np.random.uniform(0.1, 1, self.d)  # Random beta values for consumers
        sum_N_over_beta = np.sum(N / beta)
        sum_1_over_Gamma = np.sum([1 / self.gamma for _ in range(self.s)])
        h = 1 / (sum_N_over_beta + sum_1_over_Gamma)

        prices_over_time = [p]

        t = 0
        while True:
            prices_over_time.append(None)
            prices_over_time[t + 1] = self.epku2(prices_over_time[t], h, N)
            t += 1
            if np.linalg.norm(prices_over_time[t] - prices_over_time[t - 1]) < 1e-6:
                break
        return np.array(prices_over_time)[len(prices_over_time) - 1]

    def initialize_prices(self):
        self.prices = np.random.uniform(0.01, 5, self.n)
        self.prices = self.prices / np.max(self.prices)
        return self.prices

    def initialize_customers(self):
        self.customers = np.zeros((self.n, self.d))
        for i in range(self.n):
            for j in range(self.d):
                segment = i % self.m
                self.customers[i, j] = np.random.uniform(
                    segment / self.m, (segment + 1) / self.m
                )
        return self.customers

    def compute_y(self):
        self.y = (self.prices + 2 * self.gamma * self.y_dash) / (2 + 2 * self.gamma)
        return self.y

    def compute_cost_func(self, p):
        # generate s vectors y_^_s of size n from uniform distribution [0.01, 2]
        self.compute_y()
        # c is squared second norm of y
        self.c = np.sum(self.y**2, axis=1)
        self.pi = (
            np.sum(self.y * p, axis=1)
            - self.c
            - self.gamma * np.sum((self.y - self.y_dash) ** 2, axis=1)
        )

        self.expected_surplus = np.zeros(self.d)
        for d in range(self.d):
            result = 0
            for j in range(self.m):
                inner_sum = 0
                for i in self.G[j]:
                    exp_term = np.exp((self.customers[i, d] - p[i]) / self.mu[j])
                    inner_sum += np.sum(exp_term) ** self.mu[j]
                result += inner_sum
            self.expected_surplus[d] = np.log(result)
        # compute target function as sum of s
        self.cost_func = np.sum(self.pi) + np.sum(self.expected_surplus)
        return self.cost_func

    def compute_probabilities(self):
        self.probabilities = np.zeros((self.n, self.d))
        # for d in range(self.d):
        #     for j, group in enumerate(self.G):
        #         for product_idx in group:
        #             expp = np.exp((self.customers[product_idx, d] - self.prices[product_idx])/self.mu[j])
        #             expp2 = 0
        #             for k in group:
        #                 expp2 += np.exp((self.customers[k, d] - self.prices[k])/self.mu[j])
        #             expp2 = expp2 ** (self.mu[j] - 1)
        #             expp3 = 0
        #             for h, group2 in enumerate(self.G):
        #                 for k in group2:
        #                     expp3 += np.exp((self.customers[k, d] - self.prices[k])/self.mu[h])
        #                 expp3 = expp3 ** self.mu[h]
        #     self.probabilities[product_idx, d] = expp * expp2 / expp3

        # return self.probabilities
        denominator = np.zeros(self.d)
        for i in range(self.d):
            outer_sum = 0
            for h in range(self.m):
                inner_sum = 0
                for k in self.G[h]:
                    inner_sum += np.exp(
                        (self.customers[k, i] - self.prices[k]) / self.mu[h]
                    )
                outer_sum += inner_sum ** self.mu[h]
            denominator[i] = outer_sum

        # numerator = np.zeros((self.n, self.d))
        for i in range(self.n):
            for j in range(self.d):
                sum = 0
                for k in self.G[i % self.m]:
                    sum += np.exp(
                        (self.customers[k, j] - self.prices[k]) / self.mu[i % self.m]
                    )
                # numerator[i, j] = np.exp((self.customers[i, j] - self.prices[i]) / self.mu[i % self.m]) * sum ** self.mu[i % self.m]
                numerator = np.exp(
                    (self.customers[i, j] - self.prices[i]) / self.mu[i % self.m]
                ) * (sum ** (self.mu[i % self.m] - 1))
                self.probabilities[i, j] = numerator / denominator[j]
        # self.probabilities = numerator / denominator
        return self.probabilities

    def compute_gradient(self, index):
        self.compute_y()
        if index < self.s:
            return self.y[index]
        if index < self.s + self.d:
            self.compute_probabilities()
            return np.array(
                [
                    (
                        -1
                        if np.random.uniform()
                        < self.probabilities[index, index - self.s]
                        else 0
                    )
                    for _ in range(self.n)
                ]
            )
