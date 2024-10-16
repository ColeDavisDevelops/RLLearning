import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import pandas as pd
import os  # To handle directory creation
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Parameters
gamma = 0.95                  # Discount factor for future rewards
epsilon = 1.0                 # Initial exploration rate
epsilon_decay = 0.995         # Decay rate for exploration probability
epsilon_min = 0.01            # Minimum exploration probability
batch_size = 32               # Batch size for training
episodes = 5                   # Number of episodes to train
target_update_freq = 5        # Frequency to update the target network

# Ensure the results directory exists
if not os.path.exists('results'):
    os.makedirs('results')

# Read the CSV file containing historical stock prices
data = pd.read_csv('AAPL.csv')  # Replace with your CSV file name

# Ensure that the 'Close' column exists
if 'Close' not in data.columns:
    raise ValueError("The CSV file must contain a 'Close' column with closing prices.")

# Simple Trading Environment
class SimpleTradingEnv:
    def __init__(self, data):
        self.data = data.reset_index(drop=True)
        self.data_length = len(self.data)
        self.initial_account_value = 10000
        self.reset()

    def reset(self):
        """Resets the environment to the initial state."""
        self.account_value = self.initial_account_value
        # Randomize the starting point
        self.current_step = random.randint(0, self.data_length - 2)  # -2 to prevent index out of range
        self.done = False
        self.position = 0        # 0: No position, 1: Long position
        self.entry_price = None  # Price at which the position was opened
        self.current_price = self.data.loc[self.current_step, 'Close']
        return self._get_observation()

    def _get_observation(self):
        # Include all past prices up to the current step
        return self.data['Close'].iloc[:self.current_step + 1].values

    def step(self, action):
        """
        Takes an action (Buy, Sell, Hold) and updates the environment state.
        Returns the next state, reward, done flag, and info dictionary.
        """
        # Record whether a trade was executed
        trade_executed = False
        trade_action = None  # Will be 'Buy' or 'Sell' if a trade occurs

        reward = 0  # Default reward

        # Define actions
        # 0: Buy
        # 1: Sell
        # 2: Hold
        if action == 0:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = self.current_price
                self.account_value -= self.current_price  # Deduct the cost
                trade_executed = True
                trade_action = 'Buy'
                # No immediate reward on buying
        elif action == 1:  # Sell
            if self.position == 1:
                self.position = 0
                self.account_value += self.current_price  # Add the proceeds
                trade_executed = True
                trade_action = 'Sell'
                # Calculate profit or loss from the trade
                profit_loss = self.current_price - self.entry_price
                reward = profit_loss
                self.entry_price = None  # Reset entry price
        elif action == 2:  # Hold
            pass  # Do nothing

        # Move to the next step
        self.current_step += 1

        # Check if the episode is done
        if self.current_step >= self.data_length - 1:
            self.done = True
        else:
            self.current_price = self.data.loc[self.current_step, 'Close']

        # Update net liquidation value
        self.net_liq_val = self.account_value + (self.position * self.current_price)

        # If the episode is done and we still hold a position, close it
        if self.done and self.position == 1:
            # Close the position
            self.position = 0
            self.account_value += self.current_price
            trade_executed = True
            trade_action = 'Sell'
            # Calculate profit or loss from the trade
            profit_loss = self.current_price - self.entry_price
            reward += profit_loss
            self.entry_price = None

        # Get the next observation
        if not self.done:
            observation = self._get_observation()
        else:
            observation = np.zeros(self.current_step + 1)  # Return zeros if done

        # Include trade information in info dictionary
        info = {'trade_executed': trade_executed, 'trade_action': trade_action}

        return observation, reward, self.done, info

# Create the DQN Model with LSTM
def create_model(input_shape, action_size):
    """
    Builds a Deep Q-Network model using Keras with LSTM layers.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0., input_shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(action_size, activation='linear')
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# Replay Buffer
class ReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        """
        Adds a new experience to the buffer.
        If the buffer is full, it removes the oldest experience.
        """
        self.buffer.append(experience)
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.
        """
        return random.sample(self.buffer, batch_size)

# Training Function
def train(replay_buffer, model, target_model, batch_size, max_length):
    """
    Trains the model using experiences from the replay buffer.
    """
    if len(replay_buffer.buffer) < batch_size:
        return None  # Return None if not enough samples

    minibatch = replay_buffer.sample(batch_size)
    states = [exp[0] for exp in minibatch]
    actions = np.array([exp[1] for exp in minibatch])
    rewards = np.array([exp[2] for exp in minibatch])
    next_states = [exp[3] for exp in minibatch]
    dones = np.array([exp[4] for exp in minibatch])

    # Pad sequences
    states_padded = pad_sequences(states, maxlen=max_length, dtype='float32', padding='pre', truncating='pre')
    next_states_padded = pad_sequences(next_states, maxlen=max_length, dtype='float32', padding='pre', truncating='pre')

    # Reshape for LSTM input
    states_padded = states_padded.reshape(len(minibatch), max_length, 1)
    next_states_padded = next_states_padded.reshape(len(minibatch), max_length, 1)

    # Predict Q-values for current states and next states
    q_values = model.predict(states_padded)
    target_q_values = target_model.predict(next_states_padded)

    # Update Q-values using the Bellman equation
    for i in range(len(minibatch)):
        if dones[i]:
            q_values[i][actions[i]] = rewards[i]
        else:
            q_values[i][actions[i]] = rewards[i] + gamma * np.amax(target_q_values[i])

    # Train the model and capture the loss
    history = model.fit(states_padded, q_values, epochs=1, verbose=0)
    loss = history.history['loss'][0]
    return loss

# Epsilon-Greedy Action Selection
def select_action(state, model, epsilon, max_length):
    """
    Selects an action using an epsilon-greedy policy.
    """
    if np.random.rand() <= epsilon:
        return np.random.randint(0, 3)  # Random action (Buy, Sell, Hold)
    else:
        # Pad the state
        padded_state = pad_sequences([state], maxlen=max_length, dtype='float32', padding='pre', truncating='pre')
        padded_state = padded_state.reshape(1, max_length, 1)
        q_values = model.predict(padded_state)
        return np.argmax(q_values[0])

# Main Training Loop
env = SimpleTradingEnv(data)
action_size = 3  # Buy, Sell, Hold

max_length = env.data_length  # Maximum possible sequence length

input_shape = (max_length, 1)

model = create_model(input_shape, action_size)
target_model = create_model(input_shape, action_size)
target_model.set_weights(model.get_weights())

replay_buffer = ReplayBuffer()
episode_rewards = []          # To track rewards for each episode
losses = []                   # To track loss values during training
avg_q_values = []             # To track average Q-values
epsilon_values = []           # To track epsilon values over episodes

for episode in range(episodes):
    state = env.reset()
    total_reward = 0
    steps = 0

    # Initialize lists to collect data for plotting
    step_numbers = []
    prices = []
    positions = []
    trade_indices = []    # Indices where trades were executed
    trade_actions = []    # Actions that were executed ('Buy' or 'Sell')

    while True:
        action = select_action(state, model, epsilon, max_length)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        # Collect data for plotting
        step_numbers.append(env.current_step)
        prices.append(env.current_price)
        positions.append(env.position)

        # Record actions only if a trade was executed
        if info['trade_executed']:
            trade_indices.append(len(step_numbers) - 1)  # Current index
            trade_actions.append(info['trade_action'])

        # Store experience in replay buffer
        replay_buffer.add((state, action, reward, next_state, done))
        state = next_state

        # Train the model if enough samples are available
        loss = train(replay_buffer, model, target_model, batch_size, max_length)
        if loss is not None:
            losses.append(loss)

        # Update target network periodically
        if steps % target_update_freq == 0:
            target_model.set_weights(model.get_weights())

        # Track average Q-values
        # Pad the state
        padded_state = pad_sequences([state], maxlen=max_length, dtype='float32', padding='pre', truncating='pre')
        padded_state = padded_state.reshape(1, max_length, 1)
        q_values = model.predict(padded_state)
        avg_q = np.mean(q_values[0])
        avg_q_values.append(avg_q)

        if done:
            episode_rewards.append(total_reward)
            epsilon_values.append(epsilon)
            print(f"Episode: {episode + 1}, Total Reward: {total_reward:.2f}, Steps: {steps}, Epsilon: {epsilon:.2f}")
            print(f"Final Net Liquidation Value: {env.net_liq_val:.2f}")

            # Decay epsilon after each episode
            epsilon = max(epsilon_min, epsilon_decay * epsilon)

            # Plotting trading results
            plt.figure(figsize=(12, 6))
            plt.plot(step_numbers, prices, label='Price')

            # Plot executed trades
            for idx, trade_action in zip(trade_indices, trade_actions):
                if trade_action == 'Buy':
                    plt.scatter(step_numbers[idx], prices[idx],
                                color='green', marker='^', s=100, label='Buy' if 'Buy' not in plt.gca().get_legend_handles_labels()[1] else "")
                elif trade_action == 'Sell':
                    plt.scatter(step_numbers[idx], prices[idx],
                                color='red', marker='v', s=100, label='Sell' if 'Sell' not in plt.gca().get_legend_handles_labels()[1] else "")

            plt.xlabel('Step')
            plt.ylabel('Price')
            plt.title(f'Episode {episode + 1} Trading Results')
            plt.legend()

            # Add a text box with the final net liquidation value
            textstr = f'Final Net Liquidation Value: {env.net_liq_val:.2f}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.5)
            plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top', bbox=props)


            plt.savefig(f'results/trades_episode_{episode + 1}.png')
            plt.close()
            print(f"Trading plot saved to results/trades_episode_{episode + 1}.png.")

            break

# Plot of rewards over episodes
plt.figure()
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Total Rewards over Episodes')
plt.savefig('results/total_rewards.png')
plt.close()
print("Total Rewards plot saved to results/total_rewards.png.")

# Plot of training loss over time
plt.figure()
plt.plot(losses)
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.savefig('results/loss.png')
plt.close()
print("Loss plot saved to results/loss.png.")

# Plot of average Q-values over time
plt.figure()
plt.plot(avg_q_values)
plt.xlabel('Training Steps')
plt.ylabel('Average Q-Value')
plt.title('Average Q-Values Over Time')
plt.savefig('results/avg_q_values.png')
plt.close()
print("Average Q-Values plot saved to results/avg_q_values.png.")

# Plot of epsilon over episodes
plt.figure()
plt.plot(range(1, episodes + 1), epsilon_values)
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.title('Epsilon Values over Episodes')
plt.savefig('results/epsilon_values.png')
plt.close()
print("Epsilon plot saved to results/epsilon_values.png.")
