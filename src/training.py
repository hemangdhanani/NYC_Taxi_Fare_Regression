import argparse
import time
from utils.common import read_config
from utils.data_mgmt import get_data
from utils.data_mgmt import get_data_overview
from utils.data_mgmt import get_eda_results
from utils.data_mgmt import data_preprocess


def training(config_path):
    config = read_config(config_path)
    testing_datasize = config['params']['testing_datasize'] #TO DO from config.yaml
    start_time = time.time()
    train_data, test_data = get_data() #TO DO send sample fraction from config.yaml
    get_data_overview(train_data, test_data)
    get_eda_results(train_data, test_data)
    train_data_remove_outliers = data_preprocess(train_data)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
