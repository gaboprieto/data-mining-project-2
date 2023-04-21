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
import matplotlib.pyplot as plt
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
    # set random seed for reproducibility
    random.seed(8)
    # Split the training data into training and validation sets
    train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=8)
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

    training_accuracies = []
    validation_accuracies = []

    for nleaves in nleafnodes[1:]:
        clf = DecisionTreeClassifier(max_leaf_nodes=nleaves)
        clf.fit(X_train, y_train)

        val_pred = clf.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = clf

        train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)
        training_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)

    y_pred_best = best_model.predict(X_test)

    return (
        y_pred_random,
        y_pred_full,
        y_pred_best,
        training_accuracies,
        validation_accuracies,
        nleafnodes,
    )


# Part 4 - Model Analysis and Report
def analyze_models_and_report(y_pred_random, y_pred_full, y_pred_best, y_test):
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
    reference_columns = [
        column for column in reference_columns if column != "income_>50K"
    ]
    # Add missing columns to the test data
    for column in reference_columns:
        if column not in data.columns:
            data[column] = 0
    return data


def reorder_columns(test_data, train_columns):
    # Reorder the columns in the test data to match the training data
    ordered_columns = [col for col in train_columns if col != "income_>50K"]
    ordered_columns.append("income_>50K")

    reordered_test_data = test_data[ordered_columns]

    return reordered_test_data


def plot_accuracies(training_accuracies, validation_accuracies, nleafnodes):
    plt.plot(nleafnodes, training_accuracies, label="Training")
    plt.plot(nleafnodes, validation_accuracies, label="Validation")
    plt.xlabel("Max Leaf Nodes")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Load the data
    train_data = preprocess_data("./data/adult_train.csv")
    test_data = preprocess_data("./data/adult_test.csv")

    # Preprocess the data
    test_data = add_missing_columns(test_data, train_data.columns)

    # Reorder the columns
    test_data = reorder_columns(test_data, train_data.columns)

    random.seed(8)

    # Split the data
    X_train, X_val, y_train, y_val, X_test, y_test = split_data(train_data, test_data)

    # Get the shapes of the data
    print("Train set shape:", X_train.shape)
    print("Validation set shape:", X_val.shape)
    print("Test set shape:", X_test.shape)

    # Train and evaluate the models
    (
        y_pred_random,
        y_pred_full,
        y_pred_best,
        training_accuracies,
        validation_accuracies,
        nleafnodes,
    ) = train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test)

    # Plot the accuracies
    plot_accuracies(training_accuracies, validation_accuracies, nleafnodes[1:])
    # Analyze and report the models
    analyze_models_and_report(y_pred_random, y_pred_full, y_pred_best, y_test)
