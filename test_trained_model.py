from battery_env import BatteryEnv
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# Explicitly define loss function
model = tf.keras.models.load_model("battery_dqn_model.h5", custom_objects={"mse": MeanSquaredError()})

action_counts = {0: 0, 1: 0, 2: 0}

# Create environment
env = BatteryEnv()

# Reset environment and get initial state
state = env.reset()
state = np.array(state)

total_reward = 0
steps = 10  # Number of test steps
soc_values = []


print("Testing trained model...\n")

for step in range(steps):
    # Select action using the trained model (Exploitation)
    q_values = model.predict(np.array([state]), verbose=0)
    action = np.argmax(q_values[0])  # Best action from Q-values

    # Apply action in environment
    next_state, reward, done, _ = env.step(action)
    next_state = np.array(next_state)
    action_counts[action] += 1  # Track actions
    state = next_state
    soc_values.append(state[0])
    # Update total reward
    total_reward += reward
    state = next_state

    # Print step details
    print(f"Step {step+1}: Action = {action}, State = {state}, Reward = {reward}")

    if done:
        print("Battery reached critical limit. Stopping test.")
        break

# Plot Action Distribution
plt.bar(['Idle', 'Charge', 'Discharge'], [action_counts[0], action_counts[1], action_counts[2]])
plt.xlabel('Actions')
plt.ylabel('Count')
plt.title('Action Distribution')
plt.show()

# Plot SoC Progression
plt.plot(soc_values, marker='o', linestyle='-')
plt.xlabel('Time Steps')
plt.ylabel('State of Charge (SoC)')
plt.title('Battery SoC Over Time')
plt.show()