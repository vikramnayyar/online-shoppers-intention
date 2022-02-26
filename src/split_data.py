"""
The script splits and stores the prepared data 
"""

import os
import argparse
from nbformat import read
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params

def split_and_save_data(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    raw_data_path = config["data"]["raw_data"]
    split_ratio = config["split_data"]["test_size"]
    random_state = config["base"]["random_state"]

    df = pd.read_csv(raw_data_path)
    train, test = train_test_split(
        df, 
        test_size=split_ratio, 
        random_state=random_state
        )
    train.to_csv(train_data_path, index =False)
    test.to_csv(test_data_path, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")          
    parsed_args = args.parse_args()                             # identifies yaml file as config = params.yaml
    config_path = parsed_args.config                            # extracts yaml file name in config_path
    split_and_save_data(config_path = parsed_args.config)       # reads data 



