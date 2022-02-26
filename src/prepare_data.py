"""
The script reads the analyzes and pre-processes the given data 
"""
import pandas as pd
import numpy as np

from json.tool import main
import matplotlib.pyplot as plt
import seaborn as sns


from get_data import read_params, get_data
import argparse


def analyze_data(df): 
    print('\n * Size of dataframe: {}\n'.format(df.shape))
    print('* Datatype of columns are:')
    print('{}\n\n'.format(df.info()))
    print('* Column-wise NaNs can be identified as: ')
    print('{}\n'.format(df.isnull().sum()))
    print('Total NaNs:{}'.format(df.isnull().sum().sum()))


def balance_dataset(df,col):
    def dataset_balance(df, col):
        fig, ax = plt.subplots()
        sns.countplot(x = col, data = df, palette = 'viridis')
        
        plt.title('Deposit Distribution of Bank Customers', fontsize = 16)
        plt.xlabel('Deposit', fontsize = 14)
        plt.ylabel('Total Customers', fontsize = 14)
        plt.xticks(fontsize = 12)
        plt.show()


    dataset_balance(df, col)

    df.Revenue = df.Revenue.astype("string")  # Changing from bool type

    # remove random indices from majority category
    to_remove = np.random.choice(df[df[col] == "False"].index, size = 5000, replace=False)   # default 5000 
    df = df.drop(df.index[to_remove])

    # add random indices to minority category
    to_add = np.random.choice(df[df[col] == "True"].index, size = 300, replace=False) 
    df_replicate = df[df.index.isin(to_add)]
    df = pd.concat([df, df_replicate])
    return df


# convert given features to categorical features 


def convert_cat(df, col_list): 
    df_temp = pd.DataFrame() 
    df_temp = df
#     col_list = col_list
    
    for col in col_list:
        df_temp[col] = df_temp[col].astype('category')
        df_temp[col] = df_temp[col].cat.codes
    
    print('Categorical conversion of columns was completed successfully.')  # Writing to logfile
    
    return df_temp 




def prepare_data(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    new_cols = [col for col in df.columns]
    
    analyze_data(df)                            # analyzing data
    
    df = balance_dataset(df, "Revenue")         # balancing dataset

    col_list = [
        'Month', 'VisitorType', 
        'Weekend', 'Revenue'
        ]
    df = convert_cat(df, col_list)              # converting cols to categories
    
    raw_data_path = config["data"]["raw_data"]
    df.to_csv(raw_data_path, index = False)
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")          
    parsed_args = args.parse_args()                             # identifies yaml file as config = params.yaml
    config_path = parsed_args.config                            # extracts yaml file name in config_path
    data = prepare_data(config_path = parsed_args.config)       # reads data 







