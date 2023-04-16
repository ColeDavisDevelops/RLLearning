from environment import TradingEnvironment
from data import data

env = TradingEnvironment(data)

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()