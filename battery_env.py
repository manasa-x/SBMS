import numpy as np
import gym  #openai module
from gym import spaces

class BatteryEnv(gym.Env):
    def __init__(self):
        super(BatteryEnv, self).__init__()

        # Battery state variables
        self.soc = 0.5  # Initial State of Charge (50%)
        self.energy_demand = np.random.uniform(0.2, 0.6)  # Random energy demand
        self.renewable_supply = np.random.uniform(0.1, 0.5)  # Random renewable energy supply

        # Action space: 0 = Idle, 1 = Charge, 2 = Discharge
        self.action_space = spaces.Discrete(3)

        # Observation space: [SoC, Energy Demand, Renewable Supply]
        self.observation_space = spaces.Box(low=np.array([0.0, 0.0, 0.0]), 
                                            high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

        # Battery properties
        self.max_soc = 1.0  # Full charge
        self.min_soc = 0.2  # Minimum safe SoC
        self.charge_efficiency = 0.95  # 95% efficiency
        self.discharge_efficiency = 0.90  # 90% efficiency

    def step(self, action):
        """
        Take an action (Charge, Discharge, or Idle) and update the battery state.
        """
        reward = 0

        # Randomize energy demand and renewable supply
        self.energy_demand = np.random.uniform(0.2, 0.6)
        self.renewable_supply = np.random.uniform(0.1, 0.5)

        if action == 1:  # Charge
            if self.renewable_supply > 0:
                charge_amount = min(self.max_soc - self.soc, self.renewable_supply * self.charge_efficiency)
                self.soc += charge_amount
                reward += 10  # Reward for using renewable energy
            else:
                reward -= 5  # Penalty for attempting to charge with no energy available

        elif action == 2:  # Discharge
            if self.soc > self.min_soc:
                discharge_amount = min(self.soc - self.min_soc, self.energy_demand * self.discharge_efficiency)
                self.soc -= discharge_amount
                reward += 5  # Reward for meeting demand
            else:
                reward -= 10  # Penalty for over-discharge

        else:  # Idle
            reward -= 2  # Penalize idle behavior to encourage action

        # Check if battery is at critical limits
        done = self.soc <= self.min_soc or self.soc >= self.max_soc

        # Create new state
        state = np.array([self.soc, self.energy_demand, self.renewable_supply])
        return state, reward, done, {}

    def reset(self):
        """
        Reset the environment to an initial state.
        """
        self.soc = 0.5  # Reset to 50% SoC
        self.energy_demand = np.random.uniform(0.2, 0.6)
        self.renewable_supply = np.random.uniform(0.1, 0.5)
        return np.array([self.soc, self.energy_demand, self.renewable_supply])
