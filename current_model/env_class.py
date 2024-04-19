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

        self.nA = 9

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

        action_0 = [1]*8+[-1]*8+[1]*8
        action_1 = [-1]*8+[1]*8+[-1]*8
        action_2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        action_3 = [-0.32908258, -0.09591184,  0.02884006,  0.1155446 ,  0.1362936 ,
        0.09953183, -0.05332585, -0.20597152, -0.20619535, -0.00375301,
        0.25355506,  0.39953504,  0.60008256,  0.87294083,  1.        ,
        0.85863536,  0.54955856,  0.02666993, -0.54452552, -0.93428972,
       -1.        , -0.83623318, -0.7262561 , -0.38906533] #summer plan
        action_4 = [ 0.55270185,  0.80394396,  0.92105201,  1.        ,  0.94345696,
        0.62038445, -0.21776353, -0.70342632, -0.74746067, -0.38508783,
       -0.0577231 ,  0.21024989,  0.49863939,  0.73273232,  0.81381002,
        0.68026072,  0.41151014, -0.18105938, -0.62959089, -1.        ,
       -0.83446471, -0.45384348, -0.21452104,  0.24968145] #autumn plan
        action_5 = [ 0.64944974,  0.79791761,  0.90065498,  1.        ,  0.95809578,
        0.7370679 ,  0.22651948, -0.4104339 , -0.73879535, -0.66994322,
       -0.49958381, -0.42005924, -0.26056604, -0.1397956 , -0.16212546,
       -0.3006938 , -0.50535707, -0.95602399, -1.        , -0.74057248,
       -0.24448104,  0.15298918,  0.28904361,  0.69328197]
        action_6 = [ 1.,  1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1.,  1.,  1.,  1.] #winter
        action_7 = [ 1.,  1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,
        1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1.,  1.] #autumn
        action_8 = [-1., -1.,  1.,  1.,  1.,  1., -1., -1., -1., -1.,  1.,  1.,  1.,
        1.,  1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1.] #summer
        



        switcher = {
            0: action_0,
            1: action_1,
            2: action_2,
            3: action_3,
            4: action_4,
            5: action_5,
            6: action_6,
            7: action_7,
            8: action_8
        }

        selected_action = switcher.get(action, "Invalid action")
        
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
                new_charge_level = self.state[0] - self.discharge_rate*selected_action[hour]
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

        daily_reward = (sum([a*b for a,b in zip(prices, industrial_demand)]) - daily_cost)/100000
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