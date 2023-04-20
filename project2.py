# install dependencies:
# pip install pandas
# pip install numpy
# pip install scikit-learn
# pip install matplotlib (optional)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random


# Part 1 - Preprocessing
def preprocess_data(file_name):
    # Load the data and perform preprocessing steps
    data = pd.read_csv(file_name)

    # Removing the attributes "fnlwgt", "education", and "relationship"
    data = data.drop(["fnlwgt", "education", "relationship"], axis = 1)

    # Checking if there are any missing values 
    # workclass and occupation have '?' values, so we are removing those rows
    data = data.drop(data[data['occupation'] == '?'].index | data[data['workclass'] == '?'].index)

    # Preprocessing for native.country
    data.loc[data['native.country'] != 'United-States', 'native.country'] = 'Other'
    
    # Preprocessing for workclass
    data['workclass'] = np.where((data['workclass'] == 'Without-pay') | (data['workclass'] == 'Never-worked'), 'Not-working', data['workclass'])
    data['workclass'] = np.where((data['workclass'] == 'Self-emp-inc') | (data['workclass'] == 'Self-emp-not-inc'),  'Self-employed', data['workclass'])
    data['workclass'] = np.where((data['workclass'] == 'Federal-gov') | (data['workclass'] == 'Local-gov') | (data['workclass'] == 'State-gov'), 'Gov', data['workclass'])
    
    # Preprocessing for marital.status
    data['marital.status'] = np.where((data['marital.status'] == 'Married-AF-spouse') | (data['marital.status'] == 'Married-civ-spouse'), 'Married', data['marital.status'])
    data['marital.status'] = np.where((data['marital.status'] == 'Married-spouse-absent') | (data['marital.status'] == 'Separated') |
                                      (data['marital.status'] == 'Divorced') | (data['marital.status'] == 'Widowed'), 'Not-married', data['marital.status'])
    # Preprocessing for occupation                                  
    data['occupation'] =  np.where((data['occupation'] == 'Tech-support') | (data['occupation'] == 'Adm-clerical') | (data['occupation'] == 'Priv-house-serv') |
                                (data['occupation'] == 'Protective-serv') | (data['occupation'] == 'Armed-Forces') | (data['occupation'] == 'Other-service'), 'Other', data['occupation'])
    data['occupation'] =  np.where((data['occupation'] == 'Craft-repair') | (data['occupation'] == 'Farming-fishing') | (data['occupation'] == 'Handlers-cleaners') |
                                (data['occupation'] == 'Machine-op-inspct') | (data['occupation'] == 'Transport-moving'), 'Other', data['occupation'])

    return data


# Part 2 - Data Splitting
def split_data(train_data, test_data):
    # Do things...

    return X_train, X_val, y_train, y_val, X_test, y_test


# Part 3 - Model Training and Evaluation
def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    # M0 - Random prediction model
    pos_prob = y_train.mean()
    random.seed(8)
    y_pred_random = np.random.choice([0, 1], p=[1-pos_prob, pos_prob], size=len(y_test))
    
    # M1 - Full-grown decision tree
    dt_full = DecisionTreeClassifier()
    dt_full.fit(X_train, y_train)
    y_pred_full = dt_full.predict(X_test)

    # M2 - Pruned decision tree
    nleafnodes = [None, 1000, 800, 600, 400, 200, 100, 50, 20, 10, 5, 2]
    best_model = None
    best_val_accuracy = 0

    for nleaves in nleafnodes:
        clf = DecisionTreeClassifier(max_leaf_nodes=nleaves)
        clf.fit(X_train, y_train)

        val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = clf

    y_pred_best = best_model.predict(X_test)

    return y_pred_random, y_pred_full, y_pred_best


# Part 4 - Model Analysis and Report
def analyze_models_and_report(trained_models_file):
    # do things...
    pass


# Evaluate the model
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


if __name__ == "__main__":
    # Uncomment the parts you want to execute
    train_data = preprocess_data("./data/adult_train.csv")
    test_data = preprocess_data("./data/adult_test.csv")
    
    print(train_data.head())
    print(test_data.head())

    # X_train, X_val, y_train, y_val, X_test, y_test = split_data(train_data, test_data)

    # y_pred_random, y_pred_full, y_pred_best = train_and_evaluate_models(
    #     X_train, y_train, X_val, y_val, X_test, y_test
    # )

    # print("Random model")
    # print(evaluate_model(y_test, y_pred_random))
    # print("Full-grown decision tree")
    # print(evaluate_model(y_test, y_pred_full))
    # print("Pruned decision tree")
    # print(evaluate_model(y_test, y_pred_best))
