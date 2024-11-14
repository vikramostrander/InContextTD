from typing import Tuple

import numpy as np

from MRP.mrp import MRP
from utils import compute_steady_dist

import gymnasium as gym
import copy


class GymAgent:
    def __init__(self, env, epsilon=0.1, alpha=0.5, gamma=0.9) -> None:
        self.q_values = np.zeros((env.observation_space.n, env.action_space.n))
        self.env = env
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
    
    def act(self, state) -> int:
        if np.random.rand() < self.epsilon: action = self.env.action_space.sample()
        else: action = np.random.choice(np.where(self.q_values[state, :] == np.max(self.q_values[state, :]))[0])
        return action
    
    def learn(self, state, action, reward, next_state) -> None:
        self.q_values[state, action] += self.alpha * (reward + self.gamma * np.max(self.q_values[next_state, :]) - self.q_values[state, action])

    def make_distribution(self, matrix) -> np.ndarray:
        denom = np.sum(matrix, axis=1, keepdims=True)
        with np.errstate(divide='ignore',invalid='ignore'):
            return np.where(denom != 0, matrix / denom, 0)

    def train(self, episodes=16000) -> np.ndarray:
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, term, trunc, _ = self.env.step(action)
                self.learn(state, action, reward, next_state)
                state = next_state
                done = term or trunc
        self.policy = self.make_distribution(self.q_values)
        return self.policy
    
    def compute_P(self, n_states, episodes=4000) -> np.ndarray:
        P = np.zeros((n_states, n_states))
        # assert self.policy is not None
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = np.random.choice(self.env.action_space.n, p=self.policy[state, :])
                # action = np.random.choice(self.env.action_space.n)
                next_state, _, term, trunc, _ = self.env.step(action)
                done = term or trunc
                if done: 
                    P[state, 0] += 1
                else:
                    P[state, next_state] += 1
                    state = next_state
        self.P = self.make_distribution(P)
        return self.P


class FrozenLake(MRP):
    def __init__(self,
                 n_states: int,
                 gamma: float = 0.9) -> None:
        super().__init__(n_states)
        self.gamma = gamma

        # create gym environment
        self.env = gym.make("FrozenLake-v1")

        # create policy and transition matricies
        agent = GymAgent(self.env, gamma=self.gamma)
        self.policy = agent.train()
        self.P = agent.compute_P(n_states)
        self.steady_d = compute_steady_dist(self.P)

    def reset(self) -> int:
        state, _ = self.env.reset()
        return state

    def step(self, state: int) -> Tuple[int, float]:
        action = np.random.choice(self.env.action_space.n, p=self.policy[state, :])
        # action = np.random.choice(self.env.action_space.n)
        next_state, reward, term, trunc, _ = self.env.step(action)
        if term or trunc:
            next_state = self.reset()
            reward = (reward * 30) - 10
        return next_state, reward - 1
    
    def sample_stationary(self) -> int:
        return np.random.choice(self.n_states, p=self.steady_d)
    
    def copy(self) -> 'FrozenLake':
        fl = FrozenLake(self.n_states, self.gamma)
        fl.env = copy.deepcopy(self.env)
        # fl.policy = self.policy.copy()
        fl.P = self.P.copy()
        fl.steady_d = self.steady_d.copy()
        return fl
    