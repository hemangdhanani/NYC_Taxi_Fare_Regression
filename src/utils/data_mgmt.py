import pandas as pd
import random
sample_frac = 0.01

def skip_row(row_idx):
    if row_idx == 0:
        return False
    return random.random() > sample_frac

def get_data():
    selected_cols = 'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')
    dtypes = {
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'passenger_count': 'float32'
    }

    train_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\NYC_regression\new-york-city-taxi-fare-prediction\train.csv"
                            ,usecols=selected_cols,dtype=dtypes,parse_dates=['pickup_datetime'], skiprows=skip_row)
    test_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\NYC_regression\new-york-city-taxi-fare-prediction\test.csv")
    return (train_data, test_data)