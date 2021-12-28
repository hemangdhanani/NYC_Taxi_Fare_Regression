import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def generate_submission(sub_df, test_preds, fname):    
    sub_df['fare_amount'] = test_preds
    sub_df.to_csv(fname, index=None)

def evaluate(X_train,y_train, X_cv, y_cv, model):
    train_preds = model.predict(X_train)
    train_rmse = mean_squared_error(y_train, train_preds, squared=False)
    val_preds = model.predict(X_cv)
    val_rmse = mean_squared_error(y_cv, val_preds, squared=False)
    return train_rmse, val_rmse, train_preds, val_preds

def predict_and_submit(model, test_inputs, sub_df, fname):
    test_preds = model.predict(test_inputs)    
    sub_df['fare_amount'] = test_preds
    sub_df.to_csv(fname, index=None)
    return sub_df