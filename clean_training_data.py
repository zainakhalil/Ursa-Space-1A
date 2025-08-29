'''
    Author(s): Katy
    Creation Date: August 15th, 2025
    Description: This script cleans train.csv: cleans nulls, transforms HV & VV into matrices. 
                Test.csv is not trained as it's already deemed to be clean.
'''

# IMPORT LIBRARIES
import pandas as pd
import numpy as np

# LOAD & INSPECT DATASET
filepath = "Datasets/train.json"
print("Loading data...")
df = pd.read_json(filepath)
print("Training Data Shape: ", df.shape, "\nColumns: ", df.columns, "\n", df.head())

# CHECK FOR NULL VALUES
#print(df["inc_angle"].unique)
print("\nChecking for null values...")
df = df.replace("na", np.nan)
#print(df["inc_angle"].unique)

nan_count = np.sum(df.isnull(), axis = 0)
print("Null values: \n", nan_count, "\n")

# OPTION 1: REPLACE NULL VALUES
#print("Filling null values...")
#mean_inc_ang = df["inc_angle"].mean()
#df["inc_angle"] = df["inc_angle"].fillna(mean_inc_ang)
#nan_count = np.sum(df.isnull(), axis = 0)
#print("New null values: \n", nan_count)

# OPTION 2: DELETE NULL VALUES
print("Deleting null values")
df = df.dropna()
print("Null values dropped!")

# CHECK FOR NULL VALUES
nan_count = np.sum(df.isnull(), axis = 0)
print("\nNew null values (if any): \n", nan_count)

# CHECK CLASS IMBALANCE
print("\nChecking class imbalance...")
print("Icebergs: \n", df["is_iceberg"].value_counts())
print("Icebergs and non-icebergs are both well-represented with less than a 10`%` difference.")
print("Thus, we decided to keep the training dataset as is with no additional sampling.")

# TURN BAND_1 & BAND_2 INTO A MATRIX
print("\nTransforming HH & HV bands into 75x75 matrices...")
df["band_1"] = df["band_1"].apply(lambda x: np.array(x).reshape(75, 75))
df["band_2"] = df["band_2"].apply(lambda x: np.array(x).reshape(75, 75))
print("Transformed!\nCheck data type and shape below:")
print(type(df["band_1"].iloc[0]))
print(df["band_1"].iloc[0].shape)

# SAVE CLEAN DATASET
df.to_csv("clean_initial_dataset.csv", index = False)