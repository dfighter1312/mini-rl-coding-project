import sys
import gym
import numpy as np

from io import StringIO

from contextlib import closing

from gym.envs.toy_text import discrete


class BaseEnvironment():
    """Initial setup of an environment."""
    
    def __init__(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError
    
    def get_n_actions(self):
        raise NotImplementedError
    
    def get_actions(self):
        raise NotImplementedError
    
    def get_observation_space(self):
        raise NotImplementedError
    
    def get_possible_actions(self, state):
        raise NotImplementedError
    
##############################################

class GymEnvironment(BaseEnvironment):
    
    def __init__(self, env_name, render_mode="human", seed=42):
        self.env = gym.make(env_name, render_mode=render_mode)
        self.seed = seed
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info
        
    def reset(self):
        self.env.reset(seed=self.seed)
        
    def get_n_actions(self):
        return self.env.action_space.n
    
    def get_actions(self):
        return list(range(self.get_n_actions()))
    
    def get_observation_space(self):
        return self.env.observation_space.shape[0]
    
    def render(self, mode="ansi"):
        return self.env.render(mode=mode)
    
##############################################

class GridWorldEnvironment(discrete.DiscreteEnv):
    """
    A 4x4 Grid World environment from Sutton's Reinforcement
    Learning book chapter 4. Termial states are top left and
    the bottom right corner.
    Actions are (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave agent in current state.
    Reward of -1 at each step until agent reachs a terminal state.
    
    Args:
    ------
        P: transition dynamics of the environment.
        P[s][a]: [(prob, next_state, reward, done)]
            prob: Chance of moving to the next_state.
            next_state: State after taking the action.
            reward: Reward returned
            done: Agent reaches the terminate state
        n_states: number of states in the environment
        n_actions: number of actions in the environment
        discount_factor: gamma
    """
    
    metadata = {
        "render.modes": ['human', 'ansi']
    }
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    
    def __init__(self, shape=(4, 4)):
        self.shape = shape
        self.n_states = np.prod(self.shape)
        self.n_actions = 4
        P = {}
        
        for s in range(self.n_states):
            # Unravel index return the index on shape if
            # reshape an 1D array to array with `shape`
            # In this example:
            #            0        1         2          3
            # --------|-----------------------------------
            #    0    |  0        1         2          3
            #.   1    |  4        5         6          7
            #.   2.   |. 8        9.       10         11
            #.   3.   |.12       13        14         15
            # If you call np.unravel_index(6, (4, 4))
            # mapping is from 6 to (1, 2) ~ (row, column)
            position = np.unravel_index(s, self.shape)
            P[s] = {a: [] for a in range(self.n_actions)}
            P[s][self.UP] = self._transition_prob(position, [-1, 0])
            P[s][self.RIGHT] = self._transition_prob(position, [0, 1])
            P[s][self.DOWN] = self._transition_prob(position, [1, 0])
            P[s][self.LEFT] = self._transition_prob(position, [0, -1])
        
        # Initial state distribution is uniform
        isd = np.ones(self.n_states) / self.n_states
        
        self.P = P
        super(GridWorldEnvironment, self).__init__(self.n_states, self.n_actions, P, isd)
        
    def _limit_coordinates(self, coord):
        """Prevent the agent from falling out of the grid world"""
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord
    
    def _transition_prob(self, current, delta):
        # If stuck in terminal state
        current_state = np.ravel_multi_index(
            tuple(current), self.shape
        )
        if current_state == 0 or current_state == self.n_states - 1:
            return [(1.0, current_state, 0, True)]
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(
            tuple(new_position), self.shape
        )
        is_done = (new_state == 0) or (new_state == self.n_states - 1)
        return [(1.0, new_state, -1, is_done)]

    def render(self, mode="human"):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        for s in range(self.n_states):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif s == 0 or s == self.n_states - 1:
                output = " T "
            else:
                output = " o "
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += '\n'
            outfile.write(output)
        outfile.write('\n')
        # No need to return anything for human
        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()
            
    def get_actions(self):
        return list(range(self.n_actions))

    def get_possible_actions(self, state):
        """Get possible actions under a state.
        
        Returns: A tuple of (probability to state, next state, reward, is terminate)
        """
        return list(range(self.n_actions))
    
    def simulate_step(self, state, action):
        """Simulate a step but not actually step on it. Get the necessary information for
        policy evaluation

        Args:
            state (int): State label
            action (int): Action taken
        """
        return self.P[state][action]
            
    def get_n_actions(self):
        return self.n_actions
    
    def get_observation_space(self):
        return self.n_states
    

env_mapping = {
    "GridWorld": GridWorldEnvironment
}