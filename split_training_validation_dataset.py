'''
    Author(s): Katy
    Creation Date: August 29th, 2025
    Description: This script subsets the training dataset as the validation dataset using 2 approaches
                    1. Random sampling
                    2. Stratified sampling to ensure class balance of iceberg & non-iceberg
                See file "clean_training_dataset.py" for more about data inspection
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# DATA INSPECTION
filepath = "Datasets/clean_initial_dataset.csv"
print("Loading data...")
df1 = pd.read_csv(filepath)
print("Initial Data Shape: ", df1.shape, "\nColumns: ", df1.columns, "\n", df1.head(), "\n")
# ***Note: each ID appears once

# SET DATASETS & VALIDATION DATASET SIZE
X = df1.drop(columns = "is_iceberg")
y = df1["is_iceberg"]
print ("X size: ", X.shape)
print("Y size: ", y.shape, "\n")
val_proportion = 0.2

# OPTION 1: RANDOM SAMPLING
#print("Performing random sampling...")
#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_proportion, shuffle = True)
#print("Random sampling complete!")

# OPTION 2: STRATIFY SAMPLING
print("Performing stratify sampling...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_proportion, random_state=1234, stratify = y)
print("Stratify sampling complete!")

# VIEW CLASS BALANCE
print("\nChecking class balance...")
print("Training Data: \n", y_train.value_counts())
print("\nValidation Data: \n", y_val.value_counts())

# CONVERT TO CSV FILES
X_train.to_csv("X_train.csv", index = False)
y_train.to_csv("y_train.csv", index = False)
X_val.to_csv("X_val.csv", index = False)
y_val.to_csv("y_val.csv", index = False)