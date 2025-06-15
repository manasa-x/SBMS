from battery_env import BatteryEnv
from dqn_agent import DQNAgent
import numpy as np

# Create environment and agent
env = BatteryEnv()
agent = DQNAgent(state_size=3, action_size=3)

# Reset environment and get initial state
state = env.reset()
state = np.array(state)

# Run agent for 5 steps
for _ in range(5):
    action = agent.act(state)  # Get action from agent
    next_state, reward, done, _ = env.step(action)  # Apply action in environment
    next_state = np.array(next_state)

    agent.remember(state, action, reward, next_state, done)  # Store experience
    state = next_state

    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")

    if done:
        break  # Stop if battery reaches critical limits
