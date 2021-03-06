import pandas as pd
import random
from utils.Data_Preprocess.pre_processing import remove_missing_values
from utils.Data_Preprocess.pre_processing import remove_lat_long_outlier
from utils.Data_Preprocess.pre_processing import remove_fare_amount_outlier
from utils.Data_Preprocess.pre_processing import remove_passenger_count_outlier
from utils.Data_Preprocess.pre_processing import add_date_parts
from utils.Data_Preprocess.pre_processing import add_trip_distance
from utils.Data_Preprocess.pre_processing import calculate_landmark_distance
from utils.Data_Preprocess.pre_processing import remove_extra_columns
from utils.Data_vectorization.data_vectorization import train_cv_split_data

sample_frac = 0.01

def skip_row(row_idx):
    if row_idx == 0:
        return False
    return random.random() > sample_frac

def get_data():
    selected_cols = 'fare_amount,pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')
    selected_cols_test = 'pickup_datetime,pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude,passenger_count'.split(',')
    dtypes = {
    'fare_amount': 'float32',
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'passenger_count': 'float32'
    }
    dtypes_test = {    
    'pickup_longitude': 'float32',
    'pickup_latitude': 'float32',
    'dropoff_longitude': 'float32',
    'passenger_count': 'float32'
    }

    train_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\NYC_regression\new-york-city-taxi-fare-prediction\train.csv"
                            ,usecols=selected_cols, dtype=dtypes, parse_dates=['pickup_datetime'], skiprows=skip_row)

    test_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\NYC_regression\new-york-city-taxi-fare-prediction\test.csv"
                            ,usecols=selected_cols_test, dtype=dtypes_test, parse_dates=['pickup_datetime'])

    sample_submission_data = pd.read_csv(r"C:\Users\Hemang\Desktop\DataSet\NYC_regression\new-york-city-taxi-fare-prediction\sample_submission.csv")  

    return (train_data, test_data, sample_submission_data)

def get_data_overview(train_data, test_data):
    print("=="*30)
    print(f"Shape of training data {train_data.shape[0]}")
    print("=="*30)
    print(f"Shape of test data {test_data.shape[0]}")
    print("=="*30)
    print(f"train_data column names are: {train_data.columns}")
    print("=="*30)
    print(f"test_data column names are: {test_data.columns}")
    train_data_null = train_data.isnull().sum()
    test_data_null = test_data.isnull().sum()
    print("=="*30)
    print(f"null data for training set")
    print(train_data_null)
    print("=="*30)
    print(f"null data for testing set")
    print(test_data_null)

def get_eda_results(train_data, test_data):
    train_data.info()
    print("=="*30)
    test_data.info()
    print("=="*30)
    train_data.describe()
    print("=="*30)
    test_data.describe()
    print("=="*30)

def data_preprocess(train_data, test_data):
    train_remove_nan = remove_missing_values(train_data)
    train_remove_lat_long = remove_lat_long_outlier(train_remove_nan)
    train_remove_fare = remove_fare_amount_outlier(train_remove_lat_long)
    train_remove_passenger_count = remove_passenger_count_outlier(train_remove_fare)
    train_add_date_parts , test_add_date_parts= add_date_parts(train_remove_passenger_count, test_data, 'pickup_datetime')
    train_added_trip_distance , test_added_trip_distance= add_trip_distance(train_add_date_parts, test_add_date_parts)
    train_calculate_distance , test_calculate_distance= calculate_landmark_distance(train_added_trip_distance, test_added_trip_distance)
    train_df, test_df = remove_extra_columns(train_calculate_distance , test_calculate_distance)
    return train_df, test_df

def data_vectorization_process(train_data_):
    X_train, X_cv, y_train, y_cv = train_cv_split_data(train_data_)    
    return X_train, X_cv, y_train, y_cv

