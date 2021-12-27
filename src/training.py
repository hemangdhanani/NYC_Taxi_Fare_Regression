from utils.common import read_config
import argparse
from utils.data_mgmt import get_data
import time

def training(config_path):
    config = read_config(config_path)
    testing_datasize = config['params']['testing_datasize'] #TO DO from config.yaml
    start_time = time.time()
    train_data, test_data = get_data()
    print(f"train data shape {train_data.shape}")
    print(f"test data shape {test_data.shape}")
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="config.yaml")
    parsed_args = args.parse_args()
    training(config_path=parsed_args.config)
