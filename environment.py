import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TradingEnvironment(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.current_step = 0
        self.profit = 0

        # Define action and observation space
        self.action_space = spaces.Discrete(3) # 0: Buy, 1: Sell, 2: Hold
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,))

    def reset(self):
        self.current_step = 0
        self.profit = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.data[self.current_step]
        return obs

    def step(self, action):
        reward = 0
        if action == 0: # Buy
            self.profit -= self.data[self.current_step]
        elif action == 1: # Sell
            self.profit += self.data[self.current_step]
        else: # Hold
            pass

        self.current_step += 1

        done = self.current_step == len(self.data)
        if done:
            reward = self.profit

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        print(f"Profit: {self.profit}")