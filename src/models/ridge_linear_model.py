from sklearn.linear_model import Ridge
from .model_utils import generate_submission
from .model_utils import evaluate
from .model_utils import predict_and_submit

def ridge_liner_regressor_model(X_train, X_cv, y_train, y_cv, test_df, sample_submission_df):
    model_ridge = Ridge(random_state=42)
    model_ridge.fit(X_train, y_train)
    train_rmse, val_rmse, train_preds, val_preds = evaluate(X_train, y_train, X_cv, y_cv, model_ridge)
    print("=="*30)
    print(f"Train RMSE for Ridge model is {train_rmse}, and CV RMSE is {val_rmse}")
    print("=="*30)
    predict_and_submit(model_ridge, test_df, sample_submission_df, 'ridge_model_submission.csv')
