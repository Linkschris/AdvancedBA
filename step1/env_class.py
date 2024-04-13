import numpy as np
from gym import Env, spaces
from gym.utils import seeding

class BatteryManagementEnv(Env):

    def __init__(self):
        #self.charge_level = 50  # Initial battery charge level
        self.action_space = 1  # 24-hour action vector with actions -1, 0, or 1
        self.observation_space = 1  # Battery charge level as a float
        self.max_episode_len = 1  # Maximum number of steps in an episode

        # Set other needed variables like the charge rate, discharge rate, etc.
        self.charge_rate = 5  # Amount of charge to increase per step when charging
        self.discharge_rate = 5  # Amount of charge to decrease per step when discharging
        self.max_charge = 100
        self.min_charge = 0

        self.nA = 4

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.t = 0  # Reset the time step
        # Reset the state of the environment to an initial state
        self.charge_level = 50  # Or some other logic to determine initial charge level
        self.state = np.array([self.charge_level]).astype(np.float32)
        self.lastaction = None
        self.lastreward = None

        self.total_reward = 0

        return self.state
    
    def step(self, action):
        self.t += 1
        self.lastaction = action

        old_charge_level = self.state[0]

        done= False

        action_0 = [0,0,0,0,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0]
        action_1 = [-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1]
        action_2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        action_3 = [0,0,0,0,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0]

        switcher = {
            0: action_0,
            1: action_1,
            2: action_2,
            3: action_3
        }

        selected_action = switcher.get(action, "Invalid action")
        
        #self.charge_level = 0
        daily_reward = 0

        for hour_action in selected_action:
            reward = 0
            if hour_action == 1:  # charge
                new_charge_level = self.state[0] + self.charge_rate
                if new_charge_level <= self.max_charge:
                    self.state[0] = new_charge_level
                    reward += 50
                else:
                    reward += -12.5  # Penalty for trying to overcharge

            elif hour_action == -1:  # discharge
                new_charge_level = self.state[0] - self.discharge_rate
                if new_charge_level >= self.min_charge:
                    self.state[0] = new_charge_level
                    reward += 50
                else:
                    reward += -12.5  # Penalty for trying to overdischarge

            elif hour_action == 0:  # do nothing
                reward += 0

            daily_reward += reward
        
        self.total_reward += daily_reward

        if self.t >= self.max_episode_len:
            return (self.state, daily_reward, True, {})  # (nextstate, reward, done, info)
        
        done = False

        info = {}  # Additional information
        self.lastreward = daily_reward

        #print("Action:", action, "Old charge level:",old_charge_level, " Charge level:", self.state[0], " Reward:", daily_reward, " Total reward:", self.total_reward)

        return self.state, daily_reward, done, info
