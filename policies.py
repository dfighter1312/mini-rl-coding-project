import numpy as np


class BasePolicy:
    
    def __init__(self):
        raise NotImplementedError
    
    def get_next_action(self, actions, state=None, values=None):
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    
    def __init__(self):
        pass
    
    def get_action(self, actions):
        return np.random.choice(actions)


class EpsilonGreedyPolicy(BasePolicy):
    
    def __init__(self, epsilon):
        self.epsilon = epsilon
    
    def get_action(self, actions, state=None, values=None):
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(actions)
        else:
            # Exploitation
            return actions[np.argmax(values[state])]
        
policy_mapping = {
    "Random": RandomPolicy,
    "EpsilonGreedy": EpsilonGreedyPolicy
}