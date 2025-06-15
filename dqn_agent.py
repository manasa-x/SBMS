import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # [SoC, Energy Demand, Renewable Supply]
        self.action_size = action_size  # Actions: Idle, Charge, Discharge

        # Memory buffer for experience replay
        self.memory = deque(maxlen=2000)

        # Hyperparameters
        self.gamma = 0.95  # Discount factor (future rewards)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01  # Minimum exploration rate
        self.epsilon_decay = 0.999  # Decay factor for exploration
        self.learning_rate = 0.001  # Learning rate

        # Build the Q-network
        self.model = self._build_model()

    def _build_model(self):
        """
        Builds a neural network with three layers to approximate the Q-values.
        """
        model = Sequential([
            Dense(24, input_dim=self.state_size, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')  # Output Q-values for each action
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """
        Stores experience (state, action, reward, next_state, done) for replay.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Selects an action using an epsilon-greedy policy.
        """
        if np.random.rand() <= self.epsilon:  # Exploration (random action)
            return random.randrange(self.action_size)
        
        # Exploitation (choose best action from Q-network)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Action with highest predicted Q-value

    def replay(self, batch_size):
        """
        Trains the Q-network using experience replay.
        """
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward  # Default target is the immediate reward

            if not done:  # If episode is not finished, update with future rewards
                target += self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])

            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target  # Update Q-value of the chosen action

            self.model.fit(np.array([state]), target_f, epochs=1, verbose=0)

        # Reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
