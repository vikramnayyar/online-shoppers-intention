"""
The script reads the analyzes and pre-processes the given data 
"""
import pandas as pd
import numpy as np
import scipy.stats as stats

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



def remove_outliers(df):
    def cols_with_ouliers(df):
        def outlier_cols(x):    
            n = len(x)
            mean_x = np.mean(x)
            sd_x = np.std(x)
            numerator = max(abs(x-mean_x))
            g_calculated = numerator/sd_x
            t_value = stats.t.ppf(1 - 0.05 / (2 * n), n - 2)
            g_critical = ((n - 1) * np.sqrt(np.square(t_value))) / (np.sqrt(n) * np.sqrt(n - 2 + np.square(t_value)))
            return col if (g_critical) < g_calculated else 0
        
        # Finding columns with outliers
        col_with_outliers = []
        for col in df.columns:
            outlier_col = outlier_cols(df[col])
            col_with_outliers.append(outlier_col)
        
        while (col_with_outliers.count(0)):
            col_with_outliers.remove(0)
        
        print('Columns with outliers are: {}'.format(col_with_outliers) )
        return col_with_outliers


    cols_with_outliers = cols_with_ouliers(df) 


    # Scaling
    for col in cols_with_outliers:
        df[col] = (df[col]**(1/3.7))

    # cols still possesing outliers
    cols_with_outliers = cols_with_ouliers(df) 

    # removing scaling from 2 cols that still possess outliers
    for col in cols_with_outliers:
        df[col] = df[col]**(3.7)

    # removing outliers from col 1
    cut_off = 365
    for i in df['Informational_Duration']:
        if i >= cut_off:
            df['Informational_Duration'] = df['Informational_Duration'].replace(i, cut_off)


    # removing outliers from col 2
    cut_off = 200    
    for i in df['ProductRelated']:
        if i >= cut_off:
            df['ProductRelated'] = df['ProductRelated'].replace(i, cut_off)
    
    return df





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

    df = remove_outliers(df)                    # removing outliers
    
    raw_data_path = config["data"]["raw_data"]
    df.to_csv(raw_data_path, index = False)
    

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")          
    parsed_args = args.parse_args()                             # identifies yaml file as config = params.yaml
    config_path = parsed_args.config                            # extracts yaml file name in config_path
    data = prepare_data(config_path = parsed_args.config)       # reads data 

