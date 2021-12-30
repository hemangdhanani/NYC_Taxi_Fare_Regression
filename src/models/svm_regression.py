from sklearn.svm import LinearSVR
from .model_utils import generate_submission
from .model_utils import evaluate
from .model_utils import predict_and_submit

def svm_regression_model(X_train, X_cv, y_train, y_cv, test_df, sample_submission_df):
    svm_reg_model = LinearSVR(verbose=0, dual=True)
    svm_reg_model.fit(X_train, y_train)
    train_rmse, val_rmse, train_preds, val_preds = evaluate(X_train, y_train, X_cv, y_cv, svm_reg_model)
    print("=="*30)
    print(f"Train RMSE for SVM model is {train_rmse}, and CV RMSE is {val_rmse}")
    print("=="*30)
    predict_and_submit(svm_reg_model, test_df, sample_submission_df, 'svm_reg_model_submission.csv')