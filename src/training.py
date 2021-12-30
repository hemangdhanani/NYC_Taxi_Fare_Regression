import argparse
import time
from utils.common import read_config
from utils.data_mgmt import get_data
from utils.data_mgmt import get_data_overview
from utils.data_mgmt import get_eda_results
from utils.data_mgmt import data_preprocess
from utils.data_mgmt import data_vectorization_process
from models.hardcode_model import mean_regression_model
from models.linear_regression_model import simple_linear_regression_model
from models.ridge_linear_model import ridge_liner_regressor_model
from models.random_forest_model import random_forest_regressor_model
from models.svm_regression import svm_regression_model
from models.xgb_regressor import XGB_regressor_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def training(config_path):
    config = read_config(config_path)
    testing_datasize = config['params']['testing_datasize'] #TO DO from config.yaml
    start_time = time.time()
    train_data, test_data, sample_submission_data = get_data() #TO DO send sample fraction from config.yaml
    get_data_overview(train_data, test_data)
    get_eda_results(train_data, test_data)
    train_data_added_feature, test_data_added_feature = data_preprocess(train_data, test_data)
    X_train, X_cv, y_train, y_cv = data_vectorization_process(train_data_added_feature)

    mean_regression_model(X_train, X_cv, y_train, y_cv, test_data_added_feature, sample_submission_data)
    simple_linear_regression_model(X_train, X_cv, y_train, y_cv, test_data_added_feature, sample_submission_data)
    ridge_liner_regressor_model(X_train, X_cv, y_train, y_cv, test_data_added_feature, sample_submission_data)
    random_forest_regressor_model(X_train, X_cv, y_train, y_cv, test_data_added_feature, sample_submission_data)
    XGB_regressor_model(X_train, X_cv, y_train, y_cv, test_data_added_feature, sample_submission_data)
    svm_regression_model(X_train, X_cv, y_train, y_cv, test_data_added_feature, sample_submission_data)
    

    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"Toatl minutes {(time.time() - start_time)/60}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
