from sklearn.ensemble import RandomForestRegressor
from .model_utils import generate_submission
from .model_utils import evaluate
from .model_utils import predict_and_submit

def random_forest_regressor_model(X_train, X_cv, y_train, y_cv, test_df, sample_submission_df):
    random_forest_model = RandomForestRegressor(max_depth=10, n_jobs=-1, random_state=42, n_estimators=50)
    random_forest_model.fit(X_train, y_train)
    train_rmse, val_rmse, train_preds, val_preds = evaluate(X_train, y_train, X_cv, y_cv, random_forest_model)
    print("=="*30)
    print(f"Train RMSE for Random Forest model is {train_rmse}, and CV RMSE is {val_rmse}")
    print("=="*30)
    predict_and_submit(random_forest_model, test_df, sample_submission_df, 'random_forest_model_submission.csv')