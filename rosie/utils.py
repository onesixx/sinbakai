"""
This module contains utility functions that are used in the rosie_template app.
"""

#!/usr/bin/env python
# -*- coding: utf8 -*-

from pathlib import Path
from datetime import datetime, timedelta
import warnings
import json

import pandas as pd
import numpy as np

def conv_hms(miliseconds):
    #miliseconds = 19036
    seconds = (miliseconds//1000)%60
    minutes = (miliseconds// (1000*60))%60
    hours = (miliseconds// (1000*60*60))%24
    time_string = []
    if hours >0 :
        time_string.append(f"{hours}hrs")
    if minutes >0 :
        time_string.append(f"{minutes}min")
    if seconds >0 :
        time_string.append(f"{seconds}sec")
    return ''.join(time_string)

def my_function():
    return "Hello from my_function!!!"

def another_function():
    return "This is another function."

#--------------------------
# UTC -> KST(Korea time) 변경
#--------------------------
def convert_kst_ts(ts):
    '''
    ts : timestamp
    dttm_utc : UTC time
    dttm_kst : KST time
    str_kst : KST time 문자열
    result : KST time 문자열(밀리초 제외)
    '''
    dttm_utc = pd.to_datetime(ts.astype(int) / 1000, unit='s')
    dttm_kst = dttm_utc.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    str_kst = dttm_kst.dt.strftime('%Y-%m-%d %H:%M:%S.%f')
    str_kst_trimmed = str_kst.apply(lambda x: x[:x.rfind('.')+3])
    result = pd.to_datetime(str_kst_trimmed)
    return result


def convert_kst_utc(utc_string):
    # datetime 값으로 변환
    dt_tm_utc = datetime.strptime(utc_string,'%Y-%m-%d %H:%M:%S')

    # +9 time
    tm_kst = dt_tm_utc + timedelta(hours=9)

    # 일자 + time 문자열로 변환
    str_datetime = tm_kst.strftime('%Y-%m-%d %H:%M:%S')

    return str_datetime

def generate_round(bmsStat):
    round_list = []
    count = 1
    for i in range(len(bmsStat)):
        round_list.append(count)
        if bmsStat[i] == 7 and  (i == len(bmsStat)-1 or bmsStat[i+1] != 7):
            count += 1
    return round_list

def interpolate_by_time(data, option=1,  time = 'daytime', tindex= 'timeindx',
    target_except = [],  time_interval=1.0): #, rolling_interval=2.0):
    """ step1. insert uniform time interval (1.0 second) """
    if pd.api.types.is_datetime64_dtype(data[time].dtypes):
        time_start = data[time][0].floor('S')
        time_end   = data[time][len(data)-1].ceil('S')
        time_uniform = pd.date_range(time_start, time_end, freq=timedelta(seconds=time_interval))
    else:
        time_start = int(data[time].min())
        time_end   = int(data[time].max())
        time_uniform = list(range(time_start, time_end + 1, 1)) # [x for x in range(time_start, time_end + 1, 1)]
    data_uniform = pd.DataFrame({time: time_uniform})

    data = pd.merge(data, data_uniform, on=time, how='outer')
    data = data.sort_values(time)
    # data = data.iloc[1:]

    ### ------ step 2. interpolate the data by linear method ------
    if pd.api.types.is_datetime64_dtype(data[time].dtypes):
        data[tindex] = data[time]
    else:
        data[tindex] = data[time].apply(lambda x : f'{x:.3f}') # String
    data[tindex] = pd.to_datetime(data[tindex], format='%S.%f')
    data.set_index(tindex, inplace=True)

    ### ------ step 3 : over window size rolling_interval (2) seconds ------
    if len(target_except)>0:
        data[target_except] = data[target_except].fillna(method='ffill')
        data_target = data.drop(target_except, axis=1)
        data_without_time = data_target.drop(time, axis=1)
    else:
        data_without_time = data.drop(time, axis=1)

    if option == 1  :
        # Select only the columns that are valid for the rolling operation
        valid_columns = data_without_time.select_dtypes(include=['int64', 'float64']).columns
        # Perform the rolling operation on the selected columns
        data_without_time = data_without_time[valid_columns].rolling(window=2).mean()
        # data_without_time = data_without_time.rolling(f'{rolling_interval}S').mean()

    data_without_time = data_without_time.interpolate(method='time').round(2)
    data = pd.concat([data[time], data[target_except],  data_without_time], axis=1)

    # ------ 각 초별로 데이터를 생성 ------
    data = pd.merge(data_uniform, data, on=time, how='inner')

    return data

def formatter_2_decimals(x):
    return f"{x:.2f}"

def rleid(aserise):
    # aserise = df[col]
    char = "sixx"
    group = 0
    result = []
    for i in aserise.index:  # range(0, len(aserise)):
        # i = 11
        if aserise[i] == char:
            result.append(group)
        else:
            group = group + 1
            result.append(group)
            char = aserise[i]
    return result


def make_pdata_undup(df, col, bprocess=False):
    # df = px.data.iris()
    # col = "sepal_length"
    unique_idx = ~(pd.Series(rleid(df[col])).duplicated(keep='first')) | \
                ~(pd.Series(rleid(df[col])).duplicated(keep='last'))
    pdata = df.loc[unique_idx.tolist()]
    if bprocess :
        pdata = df.loc[:, [col]]
    return pdata

def get_project_root() -> Path:
    return Path(__file__).parent.parent

def clean_column_names(df):
    df.columns = df.columns.str.lower().str.replace(" ","_")
    return df
