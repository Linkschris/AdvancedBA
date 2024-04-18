import numpy as np
import gym
from gym import spaces

class BatteryManagementEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple battery charge/discharge system model where
    - Action space is [0, 1, 2] corresponding to [Charge, Discharge, Neither].
    - Observation space is the state of charge of the battery.
    - The battery has a capacity (100 units in this example) and starts at half capacity.
    - The goal is to keep the battery optimally utilized across a 24-hour cycle.
    """
    metadata = {'render.modes': ['console']}

    def __init__(self):
        super(BatteryManagementEnv, self).__init__()
        self.action_space = spaces.Discrete(3)  # Charge, Discharge, Neither
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)
        self.state = 50 + np.random.rand(1)  # Starting state
        self.day_length = 24  # Simulate decisions for 24 hours

    def step(self, action):
        assert self.action_space.contains(action), "Invalid Action"
        charge_level = self.state[0]
        done = False
        
        # Implement the effect of the action
        if action == 0:  # Charge
            charge_level = min(charge_level + 10, 100)
        elif action == 1:  # Discharge
            charge_level = max(charge_level - 10, 0)
        
        # Update the state
        self.state = np.array([charge_level])
        
        # Calculate reward
        reward = self._calculate_reward(charge_level)
        
        # Check if we are at the end of the day
        self.day_length -= 1
        if self.day_length == 0:
            done = True
        
        return self.state, reward, done, {}

    def reset(self):
        self.state = 50 + np.random.rand(1)  # Reset to a new initial state
        self.day_length = 24
        return self.state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(f'Charge level: {self.state[0]}')

    def _calculate_reward(self, charge_level):
        # Reward is highest at medium charge levels; this is a simple example
        return -abs(charge_level - 50)

    def close(self):
        pass
