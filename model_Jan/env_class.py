import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.utils import seeding

#load csv file
prices = pd.read_csv('../Data/prices.csv')
#change n/e values to np.NaN
prices = prices.replace('n/e', np.NaN)
prices.dropna(inplace=True)
prices.reset_index(drop=True, inplace=True)
countries = prices.columns[1:]
#change column dtype
for country in countries:
    prices[country] = prices[country].astype(float)

    #load csv file
res_gen = pd.read_csv('../Data/res_gen.csv')

class BatteryManagementEnv(Env):

    def __init__(self, start_day=0):
        #self.charge_level = 50  # Initial battery charge level
        self.start_day = start_day
        self.action_space = 1  # 24-hour action vector with actions -1, 0, or 1
        self.observation_space = 25  # Battery charge level as a float
        self.max_episode_len = 365  # Maximum number of steps in an episode

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
        # Reset the state of the environment to an initial state
        self.t = self.start_day
        self.charge_level = 10  # Or some other logic to determine initial charge level
        SoC = 10

        solar = self.get_solar(0)
        hours = range(24)

        self.state = np.array([SoC]+solar+hours)
        self.lastaction = None
        self.lastreward = None

        self.total_reward = 0

        return self.state
    
    def get_prices(self, day):
        return list(prices['price_GER'][day*24:day*24+24])
    
    def get_solar(self, day):
        return list(res_gen['solar_forecastGER'][day*24:day*24+24])
    
    def step(self, action):
        self.lastaction = action

        prices = self.get_prices(self.t)
        self.state[1:] = self.get_solar(self.t)


        old_charge_level = self.state[0]

        done= False

        action_0 = [1]*8+[-1]*8+[1]*8
        action_1 = [-1]*8+[1]*8+[-1]*8
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
                    #print(self.t,hour, prices)
                    reward += -self.charge_rate*prices[hour]
                else:
                    reward += -10  # Penalty for trying to overcharge

            elif selected_action[hour] == -1:  # discharge
                new_charge_level = self.state[0] - self.discharge_rate
                if new_charge_level >= self.min_charge:
                    self.state[0] = new_charge_level
                    reward += self.discharge_rate*prices[hour]
                else:
                    reward += -10  # Penalty for trying to overdischarge

            elif selected_action[hour] == 0:  # do nothing
                reward += 0

            daily_reward += reward
        
        self.total_reward += daily_reward
        self.lastreward = daily_reward
        self.t += 1

        if self.t-self.start_day >= self.max_episode_len:
            return (self.state, daily_reward, True, {})  # (nextstate, reward, done, info)
        
        done = False
        info = {}  # Additional information
    
        #print("Action:", action, "Old charge level:",old_charge_level, " Charge level:", self.state[0], " Reward:", daily_reward, " Total reward:", self.total_reward)

        return self.state, daily_reward, done, info