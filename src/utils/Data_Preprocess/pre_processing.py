import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import seaborn as sns
from nltk.corpus import stopwords
from tqdm import tqdm
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import re

def remove_missing_values(train_data):
    train_df = train_data.dropna()
    return train_df

def remove_lat_long_outlier(train_data):
    a = train_data.shape[0]    
    print (f"Number of pickup records = {a}")
    print("=="*30)
    temp_frame_lat_long = train_data[((train_data.dropoff_longitude >= -74.15) & (train_data.dropoff_longitude <= -73.7004) &\
                       (train_data.dropoff_latitude >= 40.5774) & (train_data.dropoff_latitude <= 40.9176)) & \
                       ((train_data.pickup_longitude >= -74.15) & (train_data.pickup_latitude >= 40.5774)& \
                       (train_data.pickup_longitude <= -73.7004) & (train_data.pickup_latitude <= 40.9176))]
    b = temp_frame_lat_long.shape[0]
    print (f"Number of outlier coordinates lying outside NY boundaries: {a-b}")
    print("=="*30)
    return temp_frame_lat_long

def remove_fare_amount_outlier(train_data_fare):
    a = train_data_fare.shape[0]
    print (f"Number of records for fare amount = {a}")
    print("=="*30)
    temp_frame_fare_amount = train_data_fare[(train_data_fare.fare_amount <500) & (train_data_fare.fare_amount >0)]
    c = temp_frame_fare_amount.shape[0]
    print ("Number of outliers from fare analysis:",(a-c))
    print("=="*30)
    return temp_frame_fare_amount

def remove_passenger_count_outlier(train_data_passenger):
    a = train_data_passenger.shape[0]
    print (f"Number of records for passenger count = {a}")
    print("=="*30)
    temp_count_passenger = train_data_passenger[(train_data_passenger.passenger_count > 0) & (train_data_passenger.passenger_count < 7)]
    d = temp_count_passenger.shape[0]
    print(f"Number of outliers for passenger_count {a-d}")
    print("=="*30)
    return temp_count_passenger

def add_date_parts(train_df, test_df, col):
    train_df[col + '_year'] = train_df[col].dt.year
    train_df[col + '_month'] = train_df[col].dt.month
    train_df[col + '_day'] = train_df[col].dt.day
    train_df[col + '_weekday'] = train_df[col].dt.weekday
    train_df[col + '_hour'] = train_df[col].dt.hour

    test_df[col + '_year'] = test_df[col].dt.year
    test_df[col + '_month'] = test_df[col].dt.month
    test_df[col + '_day'] = test_df[col].dt.day
    test_df[col + '_weekday'] = test_df[col].dt.weekday
    test_df[col + '_hour'] = test_df[col].dt.hour

    return train_df, test_df

def calculate_distance(long1, lat1, long2, lat2):
    long1, lat1, long2, lat2 = map(np.radians, [long1, lat1, long2, lat2])
    result_long = long2 - long1
    result_lat = lat2 - lat1
    a = np.sin(result_lat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(result_long/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

def add_trip_distance(train_df, test_df):
    train_df['trip_distance'] = calculate_distance(train_df['pickup_longitude'], train_df['pickup_latitude'], train_df['dropoff_longitude'], train_df['dropoff_latitude'])
    test_df['trip_distance'] = calculate_distance(test_df['pickup_longitude'], test_df['pickup_latitude'], test_df['dropoff_longitude'], test_df['dropoff_latitude'])
    return train_df, test_df

jfk_lonlat = -73.7781, 40.6413
lga_lonlat = -73.8740, 40.7769
ewr_lonlat = -74.1745, 40.6895
met_lonlat = -73.9632, 40.7794
wtc_lonlat = -74.0099, 40.7126

def add_landmark_dropoff_distance(df, landmark_name, landmark_lonlat):
    lon, lat = landmark_lonlat
    df[landmark_name + '_drop_distance'] = calculate_distance(lon, lat, df['dropoff_longitude'], df['dropoff_latitude'])
    return df

def calculate_landmark_distance(train_df, test_df):
    for a_df in [train_df, test_df]:
        for name, lonlat in [('jfk', jfk_lonlat), ('lga', lga_lonlat), ('ewr', ewr_lonlat), ('met', met_lonlat), ('wtc', wtc_lonlat)]:
            add_landmark_dropoff_distance(a_df, name, lonlat)
    return train_df, test_df

def remove_extra_columns(train_data, test_data):
    train_df = train_data.drop(['pickup_datetime'], axis = 1)
    test_df = test_data.drop(['pickup_datetime'], axis = 1)
    return train_df, test_df




    





    
