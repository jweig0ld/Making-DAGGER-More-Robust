import gym
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from gym.spaces import Discrete, Dict, Box


class CartPoleWrapper:
    def __init__(self, config=None):
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.action_space = Discrete(2)
        self.observation_space = self.env.observation_space
      
    def reset(self):
        return self.env.reset()
      
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def set_state(self, state):
        self.env = deepcopy(state)
        obs = np.array(list(self.env.unwrapped.state))
        return obs

    def get_state(self):
        return deepcopy(self.env)

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


env = CartPoleWrapper()
observation, info = env.reset()

original_obs = observation
final_state = None

for i in range(100):
    a = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(a)

    if i == 99:
        final_state = env.get_state()

final_obs = observation

print(f'Original Obs: {original_obs}. Final Obs: {final_obs}.')

new_env = CartPoleWrapper()
new_env.set_state(final_state)
a = env.action_space.sample()
new_obs, reward, terminated, truncated, info = env.step(a)
print(f'New Obs: {new_obs}.')