
"""
The script reads the given data 
"""

import yaml
import pandas as pd
import argparse


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)                      # reads contents of yaml file
    return config

def get_data(config_path):
    config =read_params(config_path)
    data_path = config["datasource"]["given"]                   # reads the data path in config
    df= pd.read_csv(data_path)
    return df

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")          
    parsed_args = args.parse_args()                             # identifies yaml file as config = params.yaml
    config_path = parsed_args.config                            # extracts yaml file name in config_path
    data = get_data(config_path = parsed_args.config)           # reads data 