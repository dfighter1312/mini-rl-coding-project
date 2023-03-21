import numpy as np

from environment import BaseEnvironment
from policies import BasePolicy


class MonteCarloEvaluator:
    
    def __init__(
        self,
        env: BaseEnvironment,
        policy: BasePolicy,
        discount_factor: float = 1.0,
        episode_count: int = 100,
        first_visit: bool = True
    ):
        self.env = env
        self.policy = policy
        self.discount_factor = discount_factor
        self.episode_count = episode_count
        self.first_visit = first_visit
        
    
    def eval(self):
        V = np.zeros(self.env.get_observation_space())
        N = np.zeros(self.env.get_observation_space())
        i = 0
        
        possible_actions = self.env.get_actions()
        
        # Run multiple episodes
        while i < self.episode_count:
            episode_states = []
            episode_returns = []
            state = self.env.reset()
            episode_states.append(state)
            
            while True:
                action = self.policy.get_action(possible_actions)
                (state, reward, done, _) = self.env.step(action)
                episode_returns.append(reward)
                if not done:
                    episode_states.append(state)
                else:
                    break
            
            # Update state values
            G = 0
            count = len(episode_states)
            
            # Go backward
            for t in range(count - 1, -1, -1):
                s, r = episode_states[t], episode_returns[t]
                G = self.discount_factor * G + r
                
                if s not in episode_states[:t] or not self.first_visit:
                    N[s] += 1
                    V[s] = V[s] + 1 / N[s] * (G - V[s])
            
            i += 1
        if hasattr(self.env, "shape"):
            return V.reshape(self.env.shape)
        else:
            return V


evaluator_mapping = {
    "MonteCarlo": MonteCarloEvaluator    
}