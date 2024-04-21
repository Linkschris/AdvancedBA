import numpy as np
import pandas as pd
from gym import Env, spaces
from gym.utils import seeding

#load csv file
data = pd.read_csv('../Data/data.csv')
prices = data[['price_GER']]
# solar = data[['solar_forecastGER']]
# wind_on = data[['windonshore_forecastGER']]
# wind_off = data[['windoffshore_forecastGER']]
# residual_gen = data[['residual_generationGER']]
# load = data[['load_GER']]
industrial_demand = data[['industrial_demand']]
# day_of_week = data[['day_of_week']]
action_plans = pd.read_csv('../final_project/action_plans.csv', header=0)


class BatteryManagementEnv(Env):

    def __init__(self, start_day=0):
        #self.charge_level = 50  # Initial battery charge level
        self.start_day = start_day
        self.action_space = 1  # 24-hour action vector with actions -1, 0, or 1
        self.observation_space = 121  # Battery charge level as a float
        self.max_episode_len = 730  # Maximum number of steps in an episode

        # Set other needed variables like the charge rate, discharge rate, etc.
        self.charge_rate = 15  # Amount of charge to increase per step when charging
        self.discharge_rate = 15  # Amount of charge to decrease per step when discharging
        self.max_charge = 15000
        self.min_charge = 0

        self.nA = action_plans.shape[1]

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

        input = self.get_input_data(0)
        #solar = self.get_solar(0)
        hours = range(24)

        self.state = np.array([SoC]+input)
        self.lastaction = None
        self.lastreward = None
        self.list_of_soc = []

        self.total_reward = 0

        return self.state
    
    def get_prices(self, day):
        return list(prices['price_GER'][day*24:day*24+24])
    
    def get_input_data(self, day):
        input_array = list(data['solar_forecastGER'][day*24:day*24+24])
        input_array += list(data['windonshore_forecastGER'][day*24:day*24+24])
        input_array += list(data['windoffshore_forecastGER'][day*24:day*24+24])
        input_array += list(data['load_GER'][day*24:day*24+24])
        input_array += list(data['month'][day*24:day*24+24])

        return input_array
    
    def get_industrial_demand(self, day):
        return list(industrial_demand['industrial_demand'][day*24:day*24+24])
    
    def step(self, action):
        self.lastaction = action

        prices = self.get_prices(self.t)
        input = self.get_input_data(self.t)
        industrial_demand = self.get_industrial_demand(self.t)
        self.state[1:] = input

        old_charge_level = self.state[0]

        done= False

        # Initialize an empty dictionary to store the column data
        action_dict = {}

        # Loop over the columns in the DataFrame
        for i in range(action_plans.shape[1]):
            # Store the column data in the dictionary
            action_dict[i] = list(action_plans.iloc[:, i])


        selected_action = action_dict[action]
        
        #self.charge_level = 0
        daily_cost = 0

        for hour in range(24):
            cost = 0
            if selected_action[hour] > 0:  # charge
                new_charge_level = self.state[0] + self.charge_rate*selected_action[hour]
                if new_charge_level <= self.max_charge:
                    self.state[0] = new_charge_level
                    #print(self.t,hour, prices)
                    cost += (self.charge_rate*selected_action[hour] + industrial_demand[hour])*prices[hour]
                else:
                    cost += 100000 + industrial_demand[hour]*prices[hour]  # Penalty for trying to overcharge

            elif selected_action[hour] < 0:  # discharge
                new_charge_level = self.state[0] + self.discharge_rate*selected_action[hour]
                if new_charge_level >= self.min_charge:
                    self.state[0] = new_charge_level
                    cost += -(self.discharge_rate*selected_action[hour] - industrial_demand[hour])*prices[hour]
                else:
                    cost += 100000 + industrial_demand[hour]*prices[hour]  # Penalty for trying to overdischarge

            elif selected_action[hour] == 0:  # do nothing
                cost += industrial_demand[hour]*prices[hour]

            daily_cost += cost
        #print(prices)
        #print("Daily cost:", daily_cost)
        #print("Original costs:", sum([a*b for a,b in zip(prices, industrial_demand)]))

        #print("Charge level:", self.state[0], " Reward:", daily_cost, " Total reward:", self.total_reward)

        daily_reward = (sum([a*b for a,b in zip(prices, industrial_demand)]) - daily_cost)
        self.total_reward += daily_reward
        self.lastreward = daily_reward
        self.t += 1
        self.list_of_soc += self.state[0]

        if self.t-self.start_day >= self.max_episode_len:
            return (self.state, daily_reward, True, {})  # (nextstate, reward, done, info)
        
        done = False
        info = {}  # Additional information
    
        #print("Action:", action, "Old charge level:",old_charge_level, " Charge level:", self.state[0], " Reward:", daily_reward, " Total reward:", self.total_reward)

        return self.state, daily_reward, done, info