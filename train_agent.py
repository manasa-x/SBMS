from battery_env import BatteryEnv
from dqn_agent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt


# Create environment and agent
env = BatteryEnv()
agent = DQNAgent(state_size=3, action_size=3)

# Training parameters
episodes = 2000  # Number of episodes to train
batch_size = 32  # Mini-batch size for replay
rewards_per_episode = []  # Track rewards per episode

# Training loop
for e in range(episodes):
    state = env.reset()
    state = np.array(state)
    total_reward = 0

    for time in range(100):  # Max steps per episode
        action = agent.act(state)  # Get action from agent
        next_state, reward, done, _ = env.step(action)  # Apply action in environment
        next_state = np.array(next_state)

        agent.remember(state, action, reward, next_state, done)  # Store experience
        state = next_state
        total_reward += reward

        if done:
            break  # Stop if battery reaches critical limits

    rewards_per_episode.append(total_reward)  # Track rewards

    # Train agent using experience replay
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    # Print episode stats
    print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.4f}")

# Save trained model
agent.model.save("battery_dqn_model.h5")
print("Training complete. Model saved as battery_dqn_model.h5")


# Plot Rewards Per Episode
plt.plot(rewards_per_episode)
plt.xlabel('Episodes')
plt.ylabel('Total Reward')
plt.title('Training Performance')
plt.show()
