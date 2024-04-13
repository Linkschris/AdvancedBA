import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.utils import seeding

#load csv file
prices = pd.read_csv('../Data/prices.csv')
#change n/e values to np.NaN
prices = prices.replace('n/e', np.NaN)
prices.dropna(inplace=True)
countries = prices.columns[1:]
#change column dtype
for country in countries:
    prices[country] = prices[country].astype(float)

class BatteryManagementEnv(Env):

    def __init__(self):
        #self.charge_level = 50  # Initial battery charge level
        self.action_space = 1  # 24-hour action vector with actions -1, 0, or 1
        self.observation_space = 49  # Battery charge level as a float
        self.max_episode_len = 100  # Maximum number of steps in an episode

        # Set other needed variables like the charge rate, discharge rate, etc.
        self.charge_rate = 5  # Amount of charge to increase per step when charging
        self.discharge_rate = 5  # Amount of charge to decrease per step when discharging
        self.max_charge = 100
        self.min_charge = 0

        self.nA = 5

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.t = 0  # Reset the time step
        # Reset the state of the environment to an initial state
        self.charge_level = 10  # Or some other logic to determine initial charge level
        SoC = 10


        prices = self.get_prices(0)
        solar = [0]*5+[25]*4+[100]*6+[50]*4+[0]*5

        self.state = np.array([SoC]+prices+solar)
        self.lastaction = None
        self.lastreward = None

        self.total_reward = 0

        return self.state
    
    def get_prices(self, day):
        return list(prices['price_GER'][day*24:day*24+24])
    
    def step(self, action):
        self.lastaction = action

        old_charge_level = self.state[0]

        done= False

        action_0 = [1]*12 + [-1]*12
        action_1 = [-1]*12 + [1]*12	
        action_2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        action_3 = [1]*6+[-1]*5+[1]*6+[-1]*3+[1]*4#[0,0,0,0,-1,-1,-1,-1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,0,0,0,0]
        action_4 = [-1]*6+[1]*5+[-1]*6+[1]*3+[-1]*4

        switcher = {
            0: action_0,
            1: action_1,
            2: action_2,
            3: action_3,
            4: action_4
        }

        selected_action = switcher.get(action, "Invalid action")
        
        #self.charge_level = 0
        daily_reward = 0

        for hour in range(24):
            reward = 0
            if selected_action[hour] == 1:  # charge
                new_charge_level = self.state[0] + self.charge_rate
                if new_charge_level <= self.max_charge:
                    self.state[0] = new_charge_level
                    reward += -1*self.state[1+hour]
                else:
                    reward += -10  # Penalty for trying to overcharge

            elif selected_action[hour] == -1:  # discharge
                new_charge_level = self.state[0] - self.discharge_rate
                if new_charge_level >= self.min_charge:
                    self.state[0] = new_charge_level
                    reward += self.state[1+hour]
                else:
                    reward += -10  # Penalty for trying to overdischarge

            elif selected_action[hour] == 0:  # do nothing
                reward += 0

            daily_reward += reward
        
        self.total_reward += daily_reward
        self.lastreward = daily_reward
        self.t += 1
        self.state[1:25] = self.get_prices(self.t)

        if self.t >= self.max_episode_len:
            return (self.state, daily_reward, True, {})  # (nextstate, reward, done, info)
        
        done = False
        info = {}  # Additional information
    
        #print("Action:", action, "Old charge level:",old_charge_level, " Charge level:", self.state[0], " Reward:", daily_reward, " Total reward:", self.total_reward)

        return self.state, daily_reward, done, info
