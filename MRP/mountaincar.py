from typing import Tuple

import numpy as np

from MRP.mrp import MRP
from utils import compute_steady_dist


class MountainCarEnvironment(MRP):
    """Credit : https://github.com/openai/gym/blob/master/gym/envs/classic_control/mountain_car.py"""

    def __init__(self, dim, bins_per_feature: int = 2, gamma: float = 0.9, weight: np.ndarray = None, X: np.ndarray = None):

        self.gravity = np.random.uniform(low=0.002, high=0.003) # gravity is uniformly distributed
        self.force = np.random.uniform(low=0.0005, high=0.0015) # force is uniformly distributed 

        # Position bounds
        self._pos_lims = [-1.2, 0.5]
        # Speed bounds
        self._vel_lims = [-0.07, 0.07]
        # Terminal state position
        self._pos_terminal = self._pos_lims[1]

        # Generate action distribution
        self.action_dist = np.random.rand(3)
        self.action_dist = self.action_dist / np.sum(self.action_dist)

        # Number of bins per dimension
        self.s_bins = bins_per_feature
        # Define the observation boundaries for each state variable
        # state = (x, x_dot)
        self.obs_bounds = [[-1.2, 0.5], [-0.07, 0.07]]
        # Create bins for each dimension
        self.bins = [
            np.linspace(low, high, self.s_bins + 1)[1:-1]  # Exclude the first and last bin edges
            for low, high in self.obs_bounds
        ]

        self.dim = dim
        self.total_states = self.s_bins ** dim

        # Action space
        self._all_actions = [-1, 0, 1] # accelerate left, don't accelerate, accelerate right

        self.gamma = gamma
        self.w = None
        self.X = None
        self.rewards = np.random.uniform(low=-1, high=1, size=self.total_states)

        # Compute steady distribution
        self.P = self.compute_P()
        self.steady_d = compute_steady_dist(self.P)

        # Compute representable rewards
        if weight is not None:
            assert X is not None, 'feature matrix X must be provided if weight is given'
            self.w = weight
            self.X = X
            self.rewards = (np.eye(self.total_states) - gamma * self.P).dot(X.dot(weight)).squeeze()

    def is_state_valid(self, state):
        x, x_dot = state
        return x in self.pos_lims and x_dot in self.vel_lims
    
    def reset(self):
        """Get a random starting position."""
        s = np.random.uniform(low=-0.6, high=-0.4)
        return np.array([s, 0.]) # starting velocity is always 0

    def step(self, state):
        x, x_dot = state
        action = np.random.choice(self._all_actions, p=self.action_dist)

        x_dot_next = x_dot + self.force * action - self.gravity * np.cos(3 * x)
        x_dot_next = np.clip(x_dot_next, a_min=self._vel_lims[0], a_max=self._vel_lims[1])

        x_next = x + x_dot_next
        x_next = np.clip(x_next, a_min=self._pos_lims[0], a_max=self._pos_lims[1])

        if x_next == self._pos_lims[0]:
            x_dot_next = 0. # left border : reset speed

        next_state = (x_next, x_dot_next)

        discretized_state_idx = self.get_discretized_feature_idx(next_state)  
        reward = self.rewards[discretized_state_idx]
        if (x_next == self._pos_terminal): reward = 1

        return next_state, reward

    def discretize_state(self, state: Tuple[float, float]) -> Tuple[int, int]:
        """Discretize the continuous state into bins for each dimension."""
        discretized_state = []
        for i, s in enumerate(state):
            # For each dimension, find which bin s belongs to
            bin_indices = np.digitize(s, self.bins[i])
            discretized_state.append(bin_indices)
        return discretized_state
    
    def get_discretized_feature_idx(self, state: Tuple[float, float]) -> int:
        """Get the state index for the state."""
        discretized_state = self.discretize_state(state)
        # Calculate a unique index for the discretized state
        feature_index = 0
        num_bins = self.s_bins
        for i, bin_idx in enumerate(discretized_state):
            feature_index *= num_bins
            feature_index += bin_idx
        return feature_index
    
    def make_distribution(self, matrix) -> np.ndarray:
        denom = np.sum(matrix, axis=1, keepdims=True)
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.where(denom != 0, matrix / denom, 0)

    def compute_P(self, steps=50000) -> np.ndarray:
        P = np.ones((self.total_states, self.total_states))
        state = self.reset()
        for _ in range(steps):
            next_state, _ = self.step(state)
            P[self.get_discretized_feature_idx(state), self.get_discretized_feature_idx(next_state)] += 1
            state = next_state
        return self.make_distribution(P)
    
    def copy(self) -> 'MountainCarEnvironment':
        mc = MountainCarEnvironment(self.dim, self.s_bins, self.gamma, self.w, self.X)
        mc.gravity = self.gravity
        mc.force = self.force
        mc.action_dist = self.action_dist.copy()
        mc.rewards = self.rewards.copy()
        mc.P = self.P.copy()
        mc.steady_d = self.steady_d.copy()
        return mc