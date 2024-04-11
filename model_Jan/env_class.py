import numpy as np
from gym import Env, spaces
from gym.utils import seeding

class BatteryManagementEnv(Env):

    def __init__(self):
        self.charge_level = 50  # Initial battery charge level
        self.action_space = spaces.MultiDiscrete([3] * 24)  # 24-hour action vector with actions -1, 0, or 1
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([100]), dtype=np.float32)

        # Set other needed variables like the charge rate, discharge rate, etc.
        self.charge_rate = 10  # Amount of charge to increase per step when charging
        self.discharge_rate = 10  # Amount of charge to decrease per step when discharging
        self.max_charge = 100
        self.min_charge = 0

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Reset the state of the environment to an initial state
        self.charge_level = 50  # Or some other logic to determine initial charge level
        return np.array([self.charge_level]).astype(np.float32)

    def step(self, action_vector):
        total_reward = 0

        for hour_action in action_vector:
            reward = 0
            if hour_action == 2:  # charge
                new_charge_level = self.charge_level + self.charge_rate
                if new_charge_level < self.max_charge:
                    self.charge_level = new_charge_level
                    reward += 20
                else:
                    reward += -100  # Penalty for trying to overcharge

            elif hour_action == 0:  # discharge
                new_charge_level = self.charge_level - self.discharge_rate
                if new_charge_level > self.min_charge:
                    self.charge_level = new_charge_level
                    reward += 10
                else:
                    reward += -100

            elif hour_action == 1:  # wait
                reward += 0

            total_reward += reward

        info = {}

        return np.array([self.charge_level]).astype(np.float32), action_vector, total_reward, info
    
from gym.envs.registration import register

register(
     id='BatteryManagementEnv',
     entry_point='C:/Users/janle/Desktop/Master_local/2/AdvancedBA/model_Jan/env_class:BatteryManagementEnv',)
