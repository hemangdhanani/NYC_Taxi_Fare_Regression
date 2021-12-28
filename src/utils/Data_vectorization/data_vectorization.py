import warnings
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from scipy.sparse import hstack

def train_cv_split_data(data):
    train_df, val_df = train_test_split(data, test_size=0.2, random_state=42)
    input_cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'passenger_count']
    target_col = 'fare_amount'
    X_train = train_df[input_cols]
    y_train = train_df[target_col]
    X_cv = val_df[input_cols]
    y_cv = val_df[target_col]
    return  X_train, X_cv, y_train, y_cv
    