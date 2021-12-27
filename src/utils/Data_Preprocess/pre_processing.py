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
    temp_frame_fare_amount = train_data_fare[(train_data_fare.fare_amount <1000) & (train_data_fare.fare_amount >0)]
    c = temp_frame_fare_amount.shape[0]
    print ("Number of outliers from fare analysis:",(a-c))
    print("=="*30)
    return temp_frame_fare_amount

def remove_passenger_count_outlier(train_data_passenger):
    a = train_data_passenger.shape[0]
    print (f"Number of records for passenger count = {a}")
    print("=="*30)
    temp_count_passenger = train_data_passenger[(train_data_passenger.passenger_count > 0) & (train_data_passenger.passenger_count < 5)]
    d = temp_count_passenger.shape[0]
    print(f"Number of outliers for passenger_count {a-d}")
    print("=="*30)
    return temp_count_passenger




    
