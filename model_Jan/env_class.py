import numpy as np
from gym import Env, spaces
from gym.utils import seeding

class BatteryManagementEnv(Env):

    def __init__(self):
        self.charge_level = 0  # Initial battery charge level
        self.action_space = 1  # 24-hour action vector with actions -1, 0, or 1
        self.observation_space = 1  # Battery charge level as a float

        # Set other needed variables like the charge rate, discharge rate, etc.
        self.charge_rate = 10  # Amount of charge to increase per step when charging
        self.discharge_rate = 10  # Amount of charge to decrease per step when discharging
        self.max_charge = 100
        self.min_charge = 0

        self.nA = 2

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Reset the state of the environment to an initial state
        self.charge_level = 50  # Or some other logic to determine initial charge level
        return np.array([self.charge_level]).astype(np.float32)

    def step(self, action):

        done= False

        action_1 = [0,0,0,0,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0]
        action_2 = [-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]
        action_3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

        switcher = {
            0: action_1,
            1: action_2,
            2: action_3
        }

        selected_action = switcher.get(action, "Invalid action")
        
        self.charge_level = 0
        total_reward = 0

        for hour_action in selected_action:
            reward = 0
            if hour_action == 1:  # charge
                new_charge_level = self.charge_level + self.charge_rate
                if new_charge_level <= self.max_charge:
                    self.charge_level = new_charge_level
                    reward += 7.5
                else:
                    reward += -12.5  # Penalty for trying to overcharge

            elif hour_action == -1:  # discharge
                new_charge_level = self.charge_level - self.discharge_rate
                if new_charge_level >= self.min_charge:
                    self.charge_level = new_charge_level
                    reward += 7.5
                else:
                    reward += -12.5  # Penalty for trying to overdischarge

            elif hour_action == 0:  # do nothing
                reward += 0

            total_reward += reward
        done = True

        info = {}  # Additional information

        return np.array([self.charge_level]).astype(np.float32), total_reward, done, info
