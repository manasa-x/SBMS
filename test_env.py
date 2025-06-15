from battery_env import BatteryEnv

# Create environment instance
env = BatteryEnv()

# Reset the environment to get the initial state
state = env.reset()
print("Initial State:", state)

# Take some random actions
for _ in range(5):
    action = env.action_space.sample()  # Random action
    next_state, reward, done, _ = env.step(action)
    print(f"Action: {action}, Next State: {next_state}, Reward: {reward}, Done: {done}")
    
    if done:
        break  # End simulation if battery is at limits
