from sklearn.linear_model import LinearRegression
from .model_utils import generate_submission
from .model_utils import evaluate
from .model_utils import predict_and_submit

def simple_linear_regression_model(X_train, X_cv, y_train, y_cv, test_df, sample_submission_df):
    linreg_model = LinearRegression()
    linreg_model.fit(X_train, y_train)
    train_rmse, val_rmse, train_preds, val_preds = evaluate(X_train, y_train, X_cv, y_cv, linreg_model)
    print("=="*30)
    print(f"Train RMSE for Linear regression is {train_rmse}, and CV RMSE is {val_rmse}")
    print("=="*30)
    predict_and_submit(linreg_model, test_df, sample_submission_df, 'linrar_model_submission.csv')
    
