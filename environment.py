import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data
        self.current_step = 0
        self.profit = 0
        self.position = None
        self.buy_counter = 0
        self.sell_counter = 0

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 0: Buy, 1: Sell, 2: Hold
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,))

    def reset(self):
        self.current_step = 0
        self.profit = 0
        self.position = None
        self.buy_counter = 0
        self.sell_counter = 0

        return self._next_observation()

    def _next_observation(self):
        price = self.data[self.current_step]
        return np.array([price, self.position])

    def step(self, action):
        reward = 0
        if action == 0:  # Buy
            if self.position == None:
                self.position = "Bought"
                self.profit -= self.data[self.current_step]
                self.buy_counter += 1
            if self.position == "Bought":
                pass
        elif action == 1:  # Sell
            if self.position == "Bought":
                self.position = None
                self.profit += self.data[self.current_step]
                self.sell_counter -= 1
            if self.position == None:
                pass
        else:  # Hold
            pass

        self.current_step += 1

        done = self.current_step == len(self.data) - 1
        if done:
            if self.position == "Bought":
                self.profit += self.data[self.current_step]
            reward = self.profit

        return self._next_observation(), reward, done, {}

    def render(self, mode='human', close=False):
        print(f"Profit: {self.profit}")
