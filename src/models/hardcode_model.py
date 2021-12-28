import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from .model_utils import generate_submission

class MeanRegressor():
    def fit(self, inputs, targets):
        self.mean = targets.mean()

    def predict(self, inputs):
        return np.full(inputs.shape[0], self.mean)

def mean_regression_model(X_train, X_cv, y_train, y_cv, X_test, sample_submission_data):
    mean_model = MeanRegressor()
    mean_model.fit(X_train, y_train)
    mean_model.mean

    X_train_preds = mean_model.predict(X_train)
    X_cv_preds = mean_model.predict(X_cv)
    X_test_preds = mean_model.predict(X_test)

    train_rmse = mean_squared_error(y_train, X_train_preds, squared=False)
    cv_rmse = mean_squared_error(y_cv, X_cv_preds, squared=False)    
    generate_submission(sample_submission_data, X_test_preds, 'random_model_submission.csv')
    print("**"*50)
    print(f"mean model result :")
    print("--"*30)
    print(f"Train RMSE for mean model : {train_rmse}")
    print("--"*30)
    print(f"CV RMSE for mean model : {cv_rmse}")
    print("--"*30)   
    print("**"*50)
