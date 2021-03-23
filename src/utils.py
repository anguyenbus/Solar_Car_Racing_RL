# python add-ons

#from tqdm.notebook import tqdm, trange

import numpy as np
import pandas as pd
#from tqdm import tqdm
#import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
import sys 

def angle_between(v1, v2):
        """
        Calcualates the angle between two wind vecotrs. Utilizes the cos equation:
                    cos(theta) = (u dot v)/(magnitude(u) dot magnitude(v))

        Input:
            v1 = vector 1. A numpy array, list, or tuple with 
                 u in the first index and v in the second --> vector1 = [u1,v1]
            v2 = vector 2. A numpy array, list, or tuple with 
                 u in the first index and v in the second --> vector2 = [u2,v2]
        Output:     
        Returns the angle in radians between vectors 'v1' and 'v2'::
                >>> angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        angle = np.arccos(np.dot(v1_u, v2_u))
        if np.isnan(angle):
            if (v1_u == v2_u).all():
                return np.rad2deg(0.0)
            else:
                return np.rad2deg(np.pi)
        return np.rad2deg(angle)

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    try: 
        vector / np.linalg.norm(vector)
    except:
        print(vector)
        print('error!')
    return vector / np.linalg.norm(vector)

def sp_dr_to_uv(spd,dir):
    """
    calculated the u and v wind components from wind speed and direction
    Input:
        wspd: wind speed
        wdir: wind direction
    Output:
        u: u wind component
        v: v wind component
    """    
    
    rad = 4.0*np.arctan(1)/180.
    u = -spd*np.sin(rad*dir)
    v = -spd*np.cos(rad*dir)

    return u,v

def get_dict_len(prep):
    dict_len = {}
    for i in range(len(prep.values())):
        length = len(list(prep.values())[i])
        if length not in set(dict_len.keys()):
            dict_len[length] = []
        dict_len[length].append(list(prep.keys())[i])
        
    return dict_len

def make_table(dict_key, prep):
    col_name = dict_key[0]
    table = pd.Series(prep[col_name].squeeze())
    table.name = col_name
    for i in range(len(dict_key)-1):
        col = pd.Series(prep[dict_key[i+1]].squeeze())
        col.name = dict_key[i+1]
        table = pd.concat([table, col], axis = 1)
    return table

def get_minute_data(time_string):
    try:
        timeformat = '%d-%m-%Y %H:%M:%S.%f'
    except:
        timeformat = '%d-%m-%Y %H:%M'
    time_point = datetime.datetime.strptime(time_string, timeformat)
    return time_point.strftime("%d-%m-%Y %H:%M")

def get_master_table(prep_list, prep_dict, name):
    # prep_dict_ele from prep_dict
    count = 0
    for prep_ele in prep_list:
        print('Fetching data from {} .mat file of day {}/6, {} rows, packet ID {}...'.format(name,count+1,len(prep_ele[list(prep_ele.keys())[0]]), prep_dict[name][1] ))
        dict_len = get_dict_len(prep_ele)
        # get datetime table from the .mat file
        t_datetime = make_table(dict_len[list(dict_len.keys())[0]],prep_ele)
        # get table of interest fromt .mat file, for example watt, batt, odo
        t_interest = make_table(dict_len[list(dict_len.keys())[prep_dict[name][0]]],prep_ele) # 341822 	
        
        # concatenate the time table, and the table of interest
        interest_time = t_datetime[t_datetime['ID'] == prep_dict[name][1]]
        interest_time = interest_time.reset_index()
        interest_time = interest_time.drop(['index'], axis=1)
        interest_time = pd.concat([interest_time,t_interest],axis=1)

        if count == 0:
            interest_time['dateAndTimeData'] = interest_time.apply(lambda x: x.dateAndTimeData[0], axis = 1)
            # interest time master is the table that contains data of all days
            interest_time_master = interest_time
        else:
            interest_time['dateAndTimeData'] = interest_time.apply(lambda x: add_day(x.dateAndTimeData,count), axis = 1)
            interest_time_master = interest_time_master.append(interest_time,ignore_index=True)
        count += 1
    # save to dist
    interest_time_master.to_csv('simple/' + name + '_time_master_trial.csv')#, index = False)
    # change into minute interval
    interest_time_master['dateAndTimeData'] = interest_time_master.apply(lambda x: get_minute_data(x.dateAndTimeData), axis = 1)
    # drop duplicated values and keep the first row only
    interest_time_master = interest_time_master.drop_duplicates(subset=['dateAndTimeData'], keep = 'first').reset_index().drop(['index'],axis = 1)
    
    if name == 'odo':
        # with odometer table, we need to process the total sum of distance at each time point because
        # it was set at zero again after break. 
        c = 0.
        interest_time_master['distkm'] = 0.
        for idx, loc in interest_time_master.iterrows():


            if idx < len(interest_time_master) - 1:
                if interest_time_master.at[idx+1,'odometerkm'] < interest_time_master.at[idx,'odometerkm']:
                    c+=interest_time_master.at[idx,'odometerkm']
                interest_time_master.at[idx,'distkm'] = interest_time_master.at[idx,'odometerkm'] + c
            else:
                c+=interest_time_master.at[idx,'odometerkm']
                interest_time_master.at[idx,'distkm'] = interest_time_master.at[idx,'odometerkm'] + c
    return interest_time_master

def add_day(time_string, count):
    # count is the number of the day
    # if the first day then count = 0
    # second day then count = 1
    time_string = time_string[0]
    timeformat = '%d-%m-%Y %H:%M:%S.%f'
    time_point = datetime.datetime.strptime(time_string, timeformat)
    time_point = time_point + timedelta(days = count)
    return time_point.strftime("%d-%m-%Y %H:%M:%S.%f")