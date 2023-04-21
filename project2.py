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
    # Save the preprocessed data to a CSV file
    data = pd.read_csv(file_name)

    # Remove instances that have any missing values
    data.replace("?", np.nan, inplace=True)
    data.dropna(inplace=True)

    # Remove attributes: fnlwgt, education, relationship
    data.drop(columns=["fnlwgt", "education", "relationship"], inplace=True)

    # Transform the dataset into a numerical one
    data["native.country"] = data["native.country"].apply(
        lambda x: "United-States" if x == "United-States" else "other"
    )
    data["workclass"] = data["workclass"].replace(
        ["Federal-gov", "Local-gov", "State-gov"], "Gov"
    )
    data["workclass"] = data["workclass"].replace(
        ["Without-pay", "Never-worked"], "Not-working"
    )
    data["workclass"] = data["workclass"].replace(
        ["Self-emp-inc", "Self-emp-not-inc"], "Self-employed"
    )
    data["marital.status"] = data["marital.status"].replace(
        ["Married-AF-spouse", "Married-civ-spouse"], "Married"
    )
    data["marital.status"] = data["marital.status"].replace(
        ["Married-spouse-absent", "Separated", "Divorced", "Widowed"], "Not-married"
    )
    data["occupation"] = data["occupation"].replace(
        [
            "Tech-support",
            "Adm-clerical",
            "Priv-house-serv",
            "Protective-serv",
            "Armed-Forces",
            "Other-service",
        ],
        "Other",
    )
    data["occupation"] = data["occupation"].replace(
        [
            "Craft-repair",
            "Farming-fishing",
            "Handlers-cleaners",
            "Machine-op-inspct",
            "Transport-moving",
        ],
        "Manual-Work",
    )

    # Create 'income_>50K' column from 'income'
    data["income_>50K"] = data["income"].apply(lambda x: 1 if x == ">50K" else 0)
    data.drop(columns=["income"], inplace=True)

    # Encode categorical columns
    data = pd.get_dummies(
        data,
        columns=[
            "workclass",
            "marital.status",
            "occupation",
            "native.country",
            "race",
            "sex",
        ],
        drop_first=True,
    )

    return data


# Part 2 - Data Splitting
def split_data(train_data, test_data):
    # Split the training data into training and validation sets
    train_data, val_data = train_test_split(
        train_data, test_size=0.2, random_state=8
    )  # Replace 0 with your team's ID

    # Separate the features and target variables for train, val, and test sets
    X_train = train_data.drop("income_>50K", axis=1)
    y_train = train_data["income_>50K"]

    X_val = val_data.drop("income_>50K", axis=1)
    y_val = val_data["income_>50K"]

    X_test = test_data.drop("income_>50K", axis=1)
    y_test = test_data["income_>50K"]

    return X_train, X_val, y_train, y_val, X_test, y_test


# Part 3 - Model Training and Evaluation
def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test):
    # M0 - Random prediction model
    pos_prob = y_train.mean()
    random.seed(8)
    y_pred_random = np.random.choice(
        [0, 1], p=[1 - pos_prob, pos_prob], size=len(y_test)
    )

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
def analyze_models_and_report(y_pred_random, y_pred_full, y_pred_best, y_test):
    # Print the evaluation metrics for each model
    print("Random Prediction Model Metrics:", evaluate_model(y_test, y_pred_random))

    print("\nFull-grown Decision Tree Metrics:", evaluate_model(y_test, y_pred_full))

    print("\nPruned Decision Tree Metrics:", evaluate_model(y_test, y_pred_best))


# Evaluate the model
def evaluate_model(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def add_missing_columns(data, reference_columns):
    # Exclude the target column from the reference_columns list
    reference_columns = [
        column for column in reference_columns if column != "income_>50K"
    ]

    for column in reference_columns:
        if column not in data.columns:
            data[column] = 0
    return data


def reorder_columns(test_data, train_columns):
    ordered_columns = [col for col in train_columns if col != "income_>50K"]
    ordered_columns.append("income_>50K")

    reordered_test_data = test_data[ordered_columns]

    return reordered_test_data


if __name__ == "__main__":
    # Uncomment the parts you want to execute
    train_data = preprocess_data("./data/adult_train.csv")
    test_data = preprocess_data("./data/adult_test.csv")

    test_data = add_missing_columns(test_data, train_data.columns)

    test_data = reorder_columns(test_data, train_data.columns)

    X_train, X_val, y_train, y_val, X_test, y_test = split_data(train_data, test_data)

    y_pred_random, y_pred_full, y_pred_best = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test
    )

    analyze_models_and_report(y_pred_random, y_pred_full, y_pred_best, y_test)
