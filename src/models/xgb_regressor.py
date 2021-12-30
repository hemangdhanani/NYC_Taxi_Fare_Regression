from xgboost import XGBRegressor
from .model_utils import generate_submission
from .model_utils import evaluate
from .model_utils import predict_and_submit

def XGB_regressor_model(X_train, X_cv, y_train, y_cv, test_df, sample_submission_df):
    xgb_model = XGBRegressor(random_state=42, n_jobs=-1, objective='reg:squarederror')
    xgb_model.fit(X_train, y_train)
    train_rmse, val_rmse, train_preds, val_preds = evaluate(X_train, y_train, X_cv, y_cv, xgb_model)
    print("=="*30)
    print(f"Train RMSE for XGB model is {train_rmse}, and CV RMSE is {val_rmse}")
    print("=="*30)
    predict_and_submit(xgb_model, test_df, sample_submission_df, 'XGB_model_submission.csv')
