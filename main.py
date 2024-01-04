import numpy as np
from environment import TradingEnv
import random


def data(length): return [random.randint(0, 10000) for _ in range(length)]


env = TradingEnv(data(100))

# Initialize the Q-table
q_table = np.zeros((env.observation_space.shape[0], env.action_space.n))
# (3,3)
# [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


# Set the hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate

# Run the Q-learning algorithm
for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        # Choose an action using an epsilon-greedy policy
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[0])

        # Take the chosen action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # Update the Q-value of the current state-action pair
        q_table[0, action] = (1 - alpha) * q_table[0, action] + alpha * (
            reward + gamma * np.max(q_table[0]))

# Evaluate the performance of the agent
env.reset()
done = False
total_reward = 0

while not done:
    action = np.argmax(q_table[0])
    state, reward, done, _ = env.step(action)
    total_reward += reward

print(f"buys: {env.buy_counter}")
print(f"sells: {env.sell_counter}")
print("Total profit: {:.2f}".format(total_reward))
