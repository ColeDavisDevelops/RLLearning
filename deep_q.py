import numpy as np
import random
import tensorflow as tf

# Simple Trading Environment


class SimpleTradingEnv:
    def __init__(self):
        self.current_step = 0
        self.current_price = 100  # Starting price
        self.done = False

    def reset(self):
        self.current_price = 100
        self.current_step = 0
        self.done = False
        return self.current_price

    def step(self, action):
        # Simulate price change
        self.current_price += np.random.randn()
        self.current_step += 1

        # Reward function (placeholder, needs a proper strategy)
        reward = self.current_price - 100 if action == 1 else 100 - self.current_price
        print(self.current_step)

        if self.current_step > 100:
            self.done = True

        return self.current_price, reward, self.done, {}

# Create the DQN Model


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(24, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(2, activation='linear')  # 2 actions: Buy, Sell
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


# Main Training Loop
env = SimpleTradingEnv()
state_size = 1  # Current price
action_size = 2  # Buy, Sell
batch_size = 32

model = create_model()
target_model = create_model()
target_model.set_weights(model.get_weights())

replay_buffer = ReplayBuffer()
for episode in range(1000):
    state = np.reshape(env.reset(), [1, state_size])
    for step in range(200):
        # Random action for simplicity
        action = np.random.randint(0, action_size)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state
        if done:
            break

        if len(replay_buffer.buffer) > batch_size:
            train(replay_buffer, model, target_model, batch_size)
