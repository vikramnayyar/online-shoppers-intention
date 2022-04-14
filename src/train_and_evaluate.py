import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

import argparse
import joblib
import json
import pickle 

from get_data import read_params

from sklearn.ensemble import ExtraTreesClassifier  
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def custom_cmap():
    import matplotlib.colors
    
    norm = matplotlib.colors.Normalize(-1,1)
    colors = [[norm(-1.0), "#e9fcdc"], 
              [norm(-0.6), "#d9f0c9"], 
              [norm( 0.6), "#4CBB17"],
              [norm( 1.0), "#0B6623"]]
    
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    return cmap


def plot_cm(model, cm, accuracy):

    fig = plt.figure(figsize=(7, 5))
    plt.title(model, size = 15)
     
    # Declaring heatmap labels
    group_counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v2}\n{v3}" for v2, v3 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    
    # Plotting heatmap 
    cmap = custom_cmap()
    sns.heatmap(cm, annot=labels, annot_kws={"size": 15}, fmt = '', cmap=cmap)
    
    # Adding figure labels
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted Values \n \n Accuracy: {}'.format(round(accuracy, 4)))
    
    plt.show('cm_{}'.format(model))   # show figure

    
def evaluate_model(model, test_data, test_labels, model_label):
    
    pred = model.predict(test_data)
    accuracy = accuracy_score(test_labels, pred)
    cm = confusion_matrix(test_labels, pred)
    
    if accuracy > 0.90:    
        plot_cm(model_label, cm, accuracy)
    return accuracy

    
def compare_models(train_data, train_labels, test_data, test_labels):
    
    model_comparison = pd.DataFrame()
    model_names = [ExtraTreesClassifier, AdaBoostClassifier, BaggingClassifier, 
                   GradientBoostingClassifier, RandomForestClassifier, DecisionTreeClassifier,
                   XGBClassifier, CatBoostClassifier, LGBMClassifier]
    
    
    model_labels = ["etc", "abc", "bc", "gbc", "rfc", "dtc", "xgb", "cbc", "lgbm"]
    i = 0
    
    for model_name in model_names:
        
        model_label = model_labels[i]
        i += 1   
        model = model_name()   # learning_rate does not work here
        model.fit(train_data, train_labels)
                
        accuracy = evaluate_model(model, test_data, test_labels, model_label)

        model_comparison = model_comparison.append({'model_name': model_name, 
                                                    'Accuracy': accuracy}, ignore_index = True)
    
    model_comparison.sort_values(by = ['Accuracy'], ascending = False, inplace = True ) 
    
    model_comparison.reset_index(drop = True)

    return model_comparison


def read_train_test():
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)
    train_data = train.drop("Revenue", axis=1)
    train_labels = train["Revenue"] 
    test_data = test.drop("Revenue", axis=1)
    test_labels = test["Revenue"]
    
    return train_data, train_labels, test_data, test_labels


def save_records(model_comparison):

        config = read_params(config_path)
        scores_file = config["reports"]["scores"]
        
        with open(scores_file, "w") as f:
            scores = {
                # "Model": model_comparison.iloc[0][0],
                "Accuracy": model_comparison.iloc[0][1]
             }
            json.dump(scores, f, indent=4)


# def save_model(best_model):
#     # file = open('saved_models/model.pkl', 'wb')   # Open a file to store model
#     # pickle.dump(best_model, file)   # dumping information to the file
#     # file.close()    
    
#     os.makedirs("saved_models", exist_ok=True)                    # creates dir, if not present
#     model_path = os.path.join("saved_models", "best_model.joblib")     # store model location
#     print(model_path)
#     joblib.dump(best_model, model_path)



if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")          
    parsed_args = args.parse_args()                             # identifies yaml file as config = params.yaml
    config_path = parsed_args.config                            # extracts yaml file name in config_path

    train_data, train_labels, test_data, test_labels = read_train_test()

    train_data.to_csv("data/processed/X_train.csv")
    train_labels.to_csv("data/processed/Y_train.csv")
    model_comparison = compare_models(train_data, train_labels, test_data, test_labels)
    
    # best_model = model_comparison.iloc[0][0]
    best_model = RandomForestClassifier()

    # Fitting and saving best model
    best_model.fit(train_data, train_labels)
    os.makedirs("saved_models", exist_ok=True)                    # creates dir, if not present
    model_path = os.path.join("saved_models", "best_model.joblib")     # store model location
    joblib.dump(best_model, model_path)


    # save_model(best_model)
    save_records(model_comparison)

    print(best_model)