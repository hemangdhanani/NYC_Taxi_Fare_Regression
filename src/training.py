import argparse
import time
from utils.common import read_config
from utils.data_mgmt import get_data
from utils.data_mgmt import get_data_overview
from utils.data_mgmt import get_eda_results
from utils.data_mgmt import data_preprocess
from utils.data_mgmt import data_vectorization_process
from models.hardcode_model import mean_regression_model


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

    print("**"*30)
    print(f"X_train : {X_train.shape[0]}, X_cv : {X_cv.shape[0]}")
    print(f"y_train : {y_train.shape}, y_cv : {y_cv.shape}")
    print("**"*30)
    print("--- %s seconds ---" % (time.time() - start_time))
    print(f"Toatl minutes {(time.time() - start_time)/60}")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
