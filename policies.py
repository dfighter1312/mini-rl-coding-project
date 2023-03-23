import numpy as np


class BasePolicy:
    
    def __init__(self):
        raise NotImplementedError
    
    def get_next_action(self, actions, state=None, values=None):
        raise NotImplementedError
    
    def get_probs(self, actions):
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    
    def __init__(self):
        pass
    
    def get_action(self, actions):
        return np.random.choice(actions)
    
    def get_probs(self, actions):
        return np.full(len(actions), 1 / len(actions))


class EpsilonGreedyPolicy(BasePolicy):
    
    def __init__(self, epsilon):
        if epsilon > 1:
            raise ValueError("Epsilon in e-greedy policy cannot be greater than 1")
        self.epsilon = epsilon
    
    def get_action(self, actions, state=None, values=None):
        if np.random.random() < self.epsilon:
            # Exploration
            return np.random.choice(actions)
        else:
            # Exploitation
            return actions[np.argmax(values[state])]
        
    def get_probs(self, actions, state=None, values=None):
        probs = np.full(len(actions), self.epsilon / len(actions))
        max_value_state = np.argmax[values[state]]
        probs[max_value_state] += 1 - self.epsilon
        return probs
        
policy_mapping = {
    "Random": RandomPolicy,
    "EpsilonGreedy": EpsilonGreedyPolicy
}