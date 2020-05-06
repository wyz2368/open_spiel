import numpy as np

class Exp3(object):
    """
    EXP3 algorithm for adversarial bandit.
    """
    def __init__(self,
                 num_arms,
                 gamma=0.0):
        self.weights = np.ones(num_arms)
        self.num_arms = num_arms
        self.gamma = gamma
        self.arm_pulled = 0

    def sample(self, temerature=None):
        """
        Sample a new arm to pull.
        :return: int, index of arms.
        """
        weight_sum = np.sum(self.weights)
        self.probability_distribution = [(1.0 - self.gamma) * (w / weight_sum) + (self.gamma / len(self.weights)) for w in self.weights]
        self.arm_pulled = np.random.choice(range(len(self.probability_distribution)), p=self.probability_distribution)
        return self.arm_pulled

    def update_weights(self, reward):
        rewards = np.zeros(self.num_arms)
        rewards[self.arm_pulled] = reward/self.probability_distribution[self.arm_pulled]
        self.weights *= np.exp(rewards * self.gamma / self.num_arms)

def softmax(x, temperature=1/1.3):
    return np.exp(x / temperature)/np.sum(np.exp(x / temperature))

class pure_exp(object):
    def __init__(self,
                 num_arms,
                 gamma=0.0):
        self.weights = np.ones(num_arms) * 100
        self.num_arms = num_arms
        self.gamma = gamma


    def sample(self, num_iters):
        temperature = self.temperature_scheme(num_iters)
        self.probability_distribution = softmax(self.weights, temperature=temperature)
        self.arm_pulled = np.random.choice(range(len(self.probability_distribution)), p=self.probability_distribution)
        return self.arm_pulled

    def update_weights(self, reward):
        self.weights[self.arm_pulled] = (1 - self.gamma) * reward + self.gamma * self.weights[self.arm_pulled]

    def temperature_scheme(self, num_iters):
        # Numbers are hyperparameters.
        if num_iters < 20:
            return 1
        elif num_iters < 35:
            return 5
        else:
            return 10
