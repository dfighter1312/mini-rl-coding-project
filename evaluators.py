import numpy as np

from environment import BaseEnvironment
from policies import BasePolicy


class DynamicPolicyEvaluator:
    
    def __init__(
        self,
        env: BaseEnvironment,
        policy: BasePolicy,
        discount_factor: float = 1.0,
        theta: float = 0.00001
    ):
        self.env = env
        self.policy = policy
        self.discount_factor = discount_factor
        self.theta = theta
        
    def eval(self):
        V = np.zeros(self.env.get_observation_space())
        V_new = np.copy(V)
        
        while True:
            delta = 0
            for s in range(self.env.get_observation_space()):
                v = 0
                # Get possible next actions
                possible_actions = self.env.get_possible_actions(s)
                # Get the possibility for possible actions under the policy
                policy_probs = self.policy.get_probs(possible_actions)
                
                for a, pi_a in zip(possible_actions, policy_probs):
                    # Get transition info for possible actions under the environment
                    [(prob, next_state, reward, done)] = self.env.simulate_step(s, a)
                    v += pi_a * prob * (reward + self.discount_factor * V[next_state])
                
                # How much our value function changed (across any states)
                V_new[s] = v
                delta = max(delta, np.abs(V_new[s] - V[s]))
            V = np.copy(V_new)
        
            # Stop if change is below a threshold
            if delta < self.theta:
                break
        
        if hasattr(self.env, "shape"):
            return V.reshape(self.env.shape)
        else:
            return V


class DynamicValueEvaluator:
    
    def __init__(
        self,
        env: BaseEnvironment,
        policy: BasePolicy,
        discount_factor: float = 1.0,
        theta: float = 0.00001
    ):
        self.env = env
        self.policy = policy
        self.discount_factor = discount_factor
        self.theta = theta

    def eval(self):
        V = np.zeros(self.env.get_observation_space())
        V_new = np.copy(V)
        
        while True:
            delta = 0
            # For each state, perform a greedy backup
            for s in range(self.env.get_observation_space()):
                
                # Get possible next actions
                possible_actions = self.env.get_possible_actions(s)
                # Initialize q-value
                q = np.zeros(len(possible_actions))
                
                # Look at the possible next actions
                for a in range(len(possible_actions)):
                    
                    # Get transition info for possible actions under the environment
                    [(prob, next_state, reward, done)] = self.env.simulate_step(s, a)
                    if not done:
                        q[a] += prob * (reward + self.discount_factor * V[next_state])
                    else:
                        q[a] += prob * reward
                
                V_new[s] = q.max()
                delta = max(delta, np.abs(V_new[s] - V[s]))
            
            V = np.copy(V_new)
            
            # Stop if change is below a threshold
            if delta < self.theta:
                break
                
        if hasattr(self.env, "shape"):
            return V.reshape(self.env.shape)
        else:
            return V


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


class TemporalDifferenceEvaluator:
    
    def __init__(
        self,
        env: BaseEnvironment,
        policy: BasePolicy,
        discount_factor: float = 1.0,
        step_size: float = 0.25,
        episode_count: int = 100
    ):
        self.env = env
        self.policy = policy
        self.discount_factor = discount_factor
        self.episode_count = episode_count
        self.step_size = step_size

    def eval(self):
        V = np.zeros(self.env.get_observation_space())
        possible_actions = self.env.get_actions()
        i = 0
        
        # Run multiple episodes
        while i < self.episode_count:
            state = self.env.reset()
            
            while True:
                action = self.policy.get_action(possible_actions)
                (next_state, reward, done, _) = self.env.step(action)
                
                # Update equation
                if not done:
                    V[state] = V[state] + self.step_size * (reward + self.discount_factor * V[next_state] - V[state])
                else:
                    V[state] = V[state] + self.step_size * (reward - V[state])
                    break

            i += 1
        if hasattr(self.env, "shape"):
            return V.reshape(self.env.shape)
        else:
            return V


evaluator_mapping = {
    "MonteCarlo": MonteCarloEvaluator,
    "DynamicPolicy": DynamicPolicyEvaluator,
    "DynamicValue": DynamicValueEvaluator,
    "TemporalDifference": TemporalDifferenceEvaluator
}