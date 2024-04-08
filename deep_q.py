
import numpy as np
import random
import tensorflow as tf

# Simple Trading Environment


class SimpleTradingEnv:
    def __init__(self):
        self.account_value = 10000
        self.current_step = 0
        self.current_price = 100  # Starting price
        self.done = False
        self.position = 0

    def reset(self):
        self.account_value = 10000
        self.current_price = 100
        self.current_step = 0
        self.done = False
        return self.current_price

    def step(self, action):
        # Simulate price change
        self.current_price += np.random.randn()
        self.current_step += 1

        if (action == 0):  # Buy
            if (self.position == 0):
                self.position = 1
                self.account_value -= self.current_price
        elif (action == 1):  # Sell
            if (self.position == 1):
                self.position = 0
                self.account_value += self.current_price
        else:
            pass

        self.account_value = self.account_value + \
            (self.position * self.current_price)

        # Reward function (placeholder, needs a proper strategy)
        reward = self.account_value

        if self.current_step > 5:
            self.done = True

        return self.current_price, reward, self.done, {}

# Create the DQN Model


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(3, activation='linear')  # 2 actions: Buy, Sell
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# Replay Buffer


class ReplayBuffer:
    def __init__(self):
        self.buffer = []

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Training Function


def train(replay_buffer, model, target_model, batch_size):
    minibatch = replay_buffer.sample(batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + 0.95 * \
                np.amax(target_model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

# Epsilon-Greedy Action Selection


def select_action(state, model, epsilon):
    if np.random.rand() <= epsilon:
        return np.random.randint(0, action_size)
    else:
        q_values = model.predict(state)
        return np.argmax(q_values[0])


# Main Training Loop
env = SimpleTradingEnv()
state_size = 1  # Current price
action_size = 3  # Buy, Sell, Hold
batch_size = 32

model = create_model()
target_model = create_model()
target_model.set_weights(model.get_weights())

replay_buffer = ReplayBuffer()
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5  # Adjust as needed
episode_rewards = []  # To track rewards for each episode

for episode in range(episodes):
    state = np.reshape(env.reset(), [1, state_size])
    total_reward = 0
    steps = 0
    while True:
        action = select_action(state, model, epsilon)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        steps += 1

        if len(replay_buffer.buffer) > batch_size:
            train(replay_buffer, model, target_model, batch_size)

        if done:
            episode_rewards.append(total_reward)
            print(
                f"Episode: {episode + 1}, Total Reward: {total_reward}, Steps: {steps}")
            break

    epsilon = max(epsilon_min, epsilon_decay * epsilon)


try:
    import matplotlib.pyplot as plt
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('results/figure.png')  # Specify your path here
    print("Plot saved to /results/figure.png.")
except ImportError:
    print("matplotlib is not installed. Can't plot the rewards")
