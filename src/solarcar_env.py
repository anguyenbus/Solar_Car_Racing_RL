# new
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# end new


from enum import Enum
import math
import random
import datetime
import math
from datetime import timedelta

import gym
from gym import error, spaces, utils, wrappers
from gym.utils import seeding
from gym.envs.registration import register
from gym.spaces import Discrete, Box

import numpy as np
import pandas as pd
import os

from utils import get_master_table, add_day, get_dict_len, angle_between, sp_dr_to_uv



"""
The config_model below can be changed later on in case new parameters available for updates. For example, the air density value can be either updated from 
a fix value or from a function. 
"""
config_model = {'A_pv' : 4, #panel
               'A_car': 0.64, #frontal area
               'C_d': 0.145,
               'p': 1.225, #air density
               'm': 236,
               'C_r': 0.0085,
               'g': 9.8,
               'eta_elec': 0.24,
               'P_aux': 13,
               'eta_motor': 0.97, #from 0.36 to 0.96
               'E_cap': 5075*3600, # because watthour to joules
               'v_wind': 0,
               'start_time': '08-10-2017 08:00',
               'v_wind': 0,
               'wind_type' : 'normal', #there are three wind type we can set for this environment: light, normal, strong
               }

class SolarCarEnv(gym.Env):
#class solar_car():    
    
    # Timesteps are in minutes
    # Velocities are in m/s
    # action is the car velocity for the next minute
    
    def __init__(self, config = config_model):
        
        # read data from map.csv, this is the map for racing
        self.race_map = pd.read_csv("/opt/ml/code/map.csv")
        # config contains information like mass, g, ...
        self.config = config
        # speed level
        level = 0.05
        # There are 20 levels of speed, 
        self.speed_level = {19 : 0.05, # 5 percent
                            18: level*2, # 10 percent
                            17: level*3,
                            16: level*4,
                            15: level*5,
                            14 : level*6,
                            13: level*7,
                            12: level*8,
                            11: level*9,
                            10: level*10,
                            9 : level*11,
                            8: level*12,
                            7: level*13,
                            6: level*14,
                            5: level*15,
                            4 : level*16,
                            3: level*17,
                            2: level*18,
                            1: level*19,
                            0: level*20}
        
        # Reset environment
        self.reset()
        
        
    def find_speed_lim(self,km):
        """
        Find speed limit given current position
        """
        # convert distkm to an array
        array = np.asarray(self.race_map["distkm"])
        # from the array find the closest position to the km that the car in to find the sin_elevation
        idx = (np.abs(array - km)).argmin()
        # return the sin_elvevation
        return self.race_map["speed_lim"].iloc[idx]
    
    def find_bearing(self,km):
        """
        Find direction of the car in radian, this can 
        help calculate the impact of the wind to the car
        to calculate drag force
        """
        # convert distkm to an array
        array = np.asarray(self.race_map["distkm"])
        # from the array find the closest position to the km that the car in to find the sin_elevation
        idx = (np.abs(array - km)).argmin()
        # return the sin_elvevation
        return self.race_map["bearing"].iloc[idx]

    def find_solar_power(self, km):
        """
        Find solar power given the position in km
        """
        # convert distkm to an array
        array = np.asarray(self.race_map["distkm"])
        # from the array find the closest position to the km that the car in to find the sin_elevation
        idx = (np.abs(array - km)).argmin()
        # return the solar power
        return self.race_map["solar_power"].iloc[idx]

    def find_angle(self, km):
        """
        Find elevation angle, the car may run uphill
        or downhill. Running uphill will need more power
        than running downhill. 
        """
        # convert distkm to an array
        array = np.asarray(self.race_map["distkm"])
        # from the array find the closest position to the km that the car in to find the sin_elevation
        idx = (np.abs(array - km)).argmin()
        # return the solar power
        return self.race_map["elevation_angle"].iloc[idx]

    def reset(self):
        """
        Reser the whole environment
        """
        
        # STEP 1: Initialize some variables
        self.avg ,self.avg_prev, self.sum_speed, self.count_step = 0, 0, 0, 0
        # start time
        self.timestamp = self.config['start_time']
        # day ghi
        self.day_ghi_dict = {}
        # start from 0
        self.kms = 0
        # angle between wind and car
        self.angle_wind = 0
        # day list
        self.day_list = []
        # stop
        self.stop = False
        # stop times
        self.stop_times = 0
        # sin_elevation
        self.angle = self.race_map["elevation_angle"].iloc[0]
        # solar power
        self.solar_power = self.race_map["solar_power"].iloc[0]
        # speed lim
        self.speed_lim = self.race_map["speed_lim"].iloc[0]
        # original velocity is 0
        self.velocity = 0
        # reward
        self.reward = 0
        # v_wind
        self.v_wind = 0
        self.minutes = 0
        # maximum battery
        self.battery_joules_max = self.config['E_cap'] #
        self.battery_joules_min = self.config['E_cap']*0.05
        # battery 
        self.battery_joules_left = self.battery_joules_max
        
        # Action: agent can take action from as low as 5m/s or as high as 37m/s
        self.action_space = spaces.Discrete(len(self.speed_level)) # 0,1,2,3,4
        
        # Rach step, agent will look at current battery, angle, solar power, v_win, and speed limit
        self.observation_space = spaces.Box(low = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
                                            high = np.array([np.inf, np.inf, np.inf, np.inf, np.inf]),
                                            #shape = (1,8),
                                            dtype=np.float64)
        
        # self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
        # info dictionary
        # self.p = self.config['p']*self.A_car*self.C_d*self.v
        self.info = {}
        self.level = 0
        self.done = False
        return self._get_obs()
        
    
    def _add_minute(self,time_string, minutes):
        """
        Record time stamp for travel agent
        """
        # time format like 08-10-2017 08:00
        timeformat = '%d-%m-%Y %H:%M'
        time_point = datetime.datetime.strptime(time_string, timeformat)
        
        # After 17:00 the team will stop
        if time_point.hour >= 17 or time_point.hour < 8:
            # add 1 to day, because the team will start again next day
            time_point = time_point + timedelta(days = 1)
            # replace the current hour to morning, assume that the team start again at 8:00
            time_point = time_point.replace(hour = 8, minute = 0, second = 0)
            # record number of day to day_list
            if time_point.day not in self.day_list:
                self.day_list.append(time_point.day)
            # 4 hours charge, 240 minutes
            # self.battery_joules_left = self.battery_joules_left - self.power_required + self.solar_power*60*240
            
            # Assuming the battery fully charged every morning
            self.battery_joules_left = self.battery_joules_max
        else:
            
            # the car will move for each minute, minutes here equal 1 or 30, 30 means the car need to stop
            time_point += timedelta(minutes = minutes)
            # battery    
            self.battery_joules_left = self.battery_joules_left - self.power_required + self.solar_power*60*minutes
            
        return time_point.strftime("%d-%m-%Y %H:%M")
    
    def _update_info(self):
        
        self.info['kms'] = self.kms
        self.info['vehicle speed'] = self.v*3.6 #km/h
        self.info['wind speed'] = self.v_wind*3.6 #km/h
        self.info['angle_rads'] = self.angle
        self.info['average_speed'] = self.avg
        
        self.info['solar_power'] = self.solar_power
        self.info['battery_joules_left'] = self.battery_joules_left
        self.info['power_required'] = self.power_required
        self.info['speed_lim'] = self.speed_lim
        self.info['reward'] = self.reward
        self.info['timestamp'] = self.timestamp
        self.info['stop_times'] = self.stop_times
        self.info['days'] = len(self.day_list) + 1
        self.info['minutes'] = self.minutes
        
        self.info['batt_level'] = 100*np.clip(self.battery_joules_left, self.battery_joules_min, self.battery_joules_max)/self.battery_joules_max
       
    def _get_obs(self):
        """
        Get observation from the environment, which is an numpy array
        """
        self.obs = np.array([self.battery_joules_left, self.angle, self.solar_power, self.v_wind, self.speed_lim])

        return self.obs
    
    def _get_reward(self, stop):
        
        # reward for g
        self.reward += self.v/10 #avoid exploding gradient 
        
        # extra reward for faster speed
        if self.level <= 5:
            self.reward += 1
            
        
        # check if the car has to stop due to battery discharged
        if stop:
            # penalize
            self.reward -= 100
            # count stop times
            self.stop_times += 1
            # reset stop check
            self.stop = False
            self.v = 0
    
    def _get_normal_wind(self):
        # can do sample bearing from here, first
        if self.kms < 1000 or self.kms > 2000:
            # wind is lighter in the first 1000 km and 
            self.v_wind_raw = random.randint(1,3) #m/s
        else:
            # wind is strong in the middle 1000km (2000-3000)
            self.v_wind_raw = random.randint(3,5)
        # mostly wind direction is west or northwest    
        self.wind_dir = random.randint(270,315)
        self._get_v_wind_from_raw()
    
    
    def _get_v_wind_from_raw(self):
        #after getting raw value of wind velocity and wind direction, we can calculate the v_wind in regard of the vehicle v
        self.bearing = self.find_bearing(self.kms)
        try:
            self.angle_wind = angle_between(sp_dr_to_uv(self.v, self.bearing),sp_dr_to_uv(self.v_wind_raw, self.wind_dir))
        except:
            print('ERROR self.v {} self.bearing {} self.v_wind_raw {} self.wind_dir {}'.format(self.v, self.bearing, self.v_wind_raw, self.wind_dir))
        self.v_wind = self.v_wind_raw*np.cos(self.angle_wind*math.pi/180)
    
    def _get_light_wind(self):
        if self.dist < 1000 or self.dist > 2000:
            self.v_wind_raw = random.randint(1,2)
        else:
            self.v_wind_raw = random.randint(2,3)
        self.wind_dir = random.randint(270,315)
        self._get_v_wind_from_raw()
        
    def _get_strong_wind(self):
        if self.dist < 1000 or self.dist > 2000:
            self.v_wind_raw = random.randint(3,7)
        else:
            self.v_wind_raw = random.randint(7,10)
        self.wind_dir = random.randint(270,315)
        self._get_v_wind_from_raw()
        
    def _get_v_wind(self):
        if self.config['wind_type'] == 'normal':
            self._get_normal_wind()
        elif self.config['wind_type'] == 'light':
            self._get_light_wind()
        else:
            self._get_strong_wind()
    
    def _get_sim_ghi(self, time_string):
        timeformat = '%d-%m-%Y %H:%M'
        time_point = datetime.datetime.strptime(time_string, timeformat)
        day = time_point.day
        
        if day not in self.day_ghi_dict.keys():
            self.day_ghi_dict[day] = random.randint(600,1000) #2017, racing

        ghi_of_the_day = self.day_ghi_dict[day]
        hour = time_point.hour
        midday = 13
        sig = 3 # more aligned to Australian environment, can be adjusted here.
        intant_ghi = np.exp(-np.power(hour - midday, 2.) / (2 * np.power(sig, 2.)))*ghi_of_the_day
        return intant_ghi
    
    def render(self, mode='human'):    
        print('{}'.format(self.info))
        return self.info
            
    def step(self,action):
        
        # car velocity
        self.avg_prev = self.avg
        # find speed lim
        self.speed_lim = self.find_speed_lim(self.kms) 
        self.v = self.speed_level[action]*self.speed_lim/3.6
        # acceleration
        #self.E_acc = 0.5*self.config['m']*(self.v**2-self.v_prev**2) 
        self._get_v_wind()
        # power from the sun
        self.ghi = self._get_sim_ghi(self.timestamp)
        # find the solar power
        # self.solar_power = self.find_solar_power(self.kms)
        self.solar_power = self.ghi*self.config['eta_elec']*self.config['A_pv']
        # find angle elevation
        self.angle = self.find_angle(self.kms) 
        
        
        # p1 aerodynamic 
        p1 = 0.5*self.config['p']*self.config['A_car']*self.config['C_d']*self.v*(self.v+self.v_wind)**2
        # P2 rolling resistance
        p2 = self.v*self.config['C_r']*self.config['m']*self.config['g']*np.cos(self.angle)
        # P3 gravitational 
        p3 = self.v*self.config['m']*self.config['g']*np.sin(self.angle)

        # Power required to run one minute
        #self.power_required = ((p1 + p2 + p3)*1/self.config['eta_motor'] + self.config['P_aux'])*60
        self.power_required = ((p1 + p2 + p3) + self.config['P_aux'])*60
        
        # keep track of time, if the battery is enough to run, then the next one minute is added
        # if the car has to stop, then 30 minutes will be added.
        if self.battery_joules_left > self.battery_joules_min:
            # if battery is enough
            self.timestamp = self._add_minute(self.timestamp, 1)
            self.minutes += 1
        else: 
            # otherwise wait for 30 minutes
            self.stop = True
            self.timestamp = self._add_minute(self.timestamp, 30)
            self.minutes += 30

        # calculate left over battery, solar_power can be simulated
        self.battery_joules_left = self.battery_joules_left - self.power_required + self.solar_power*60 #one minute

        # clip value
        self.battery_joules_left = np.clip(self.battery_joules_left, self.battery_joules_min, self.battery_joules_left)

        # calculate the distance travelled
        self.kms = self.kms + self.v*60/1000

        # get reward
        if self.stop:
            self.count_step += 30
        else:
            self.count_step += 1
        
        
        
        self.sum_speed += self.v
        self.avg = (self.sum_speed/self.count_step)*3.6
        
        self._get_reward(self.stop)
        # check if done
        if self.kms >= 3030:
            self.done = True
        # get info
        self._update_info()
        self.level = action
        self.v_prev = self.v
        return self._get_obs(), self.reward, self.done, self.info
    
    
    def testEnv():

        env = SolarCarEnv()

        val = env.reset()
        print('val.shape = ', val.shape)

        for _ in range(5):
            print('env.observation_space =', env.observation_space)
            act = env.action_space.sample()
            print('\nact =', act)
            next_state, reward, done, _ = env.step(act)  # take a random action
            print('next_state = ', next_state)
            
        env.close()


if __name__ =='__main__':

    testEnv()