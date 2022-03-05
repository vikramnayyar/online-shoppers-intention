
"""
The script reads the given data 
"""

import yaml
import pandas as pd
import argparse
import pickle 


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)                      # reads contents of yaml file
    return config

def get_data(config_path):
    config =read_params(config_path)
    data_path = config["datasource"]["given"]                   # reads the data path in config
    df= pd.read_csv(data_path)
    return df


def extract_dict(df, col):
    feature = df[col].astype('category')
    dict_val = dict(enumerate(feature.cat.categories))
    dict_inv = {a:b for b,a in dict_val.items()}
    return dict_inv

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")          
    parsed_args = args.parse_args()                             # identifies yaml file as config = params.yaml
    config_path = parsed_args.config                            # extracts yaml file name in config_path
    data = get_data(config_path = parsed_args.config)           # reads data 


    # saving dictionaries for the app
    # admin_dur_dict = extract_dict(data, 'Administrative_Duration')
    # file = open('dict/admin_dur_dict.pkl', 'wb')     # Open a file to store model
    # pickle.dump(admin_dur_dict, file)                   # dumping information to the file
    # file.close()

    month_dict = extract_dict(data, 'Month')
    file = open('dict/month_dict.pkl', 'wb')     # Open a file to store model
    pickle.dump(month_dict, file)                   # dumping information to the file
    file.close()


    visitor_dict = extract_dict(data, 'VisitorType')
    file = open('dict/visitor_dict.pkl', 'wb')     # Open a file to store model
    pickle.dump(visitor_dict, file)                   # dumping information to the file
    file.close()


    weekend_dict = extract_dict(data, 'Weekend')
    file = open('dict/weekend_dict.pkl', 'wb')     # Open a file to store model
    pickle.dump(weekend_dict, file)                   # dumping information to the file
    file.close()
    