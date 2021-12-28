import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def generate_submission(sub_df, test_preds, fname):    
    sub_df['fare_amount'] = test_preds
    sub_df.to_csv(fname, index=None)