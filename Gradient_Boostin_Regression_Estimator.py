# Cell 1: Install packages (if needed)
# !pip install scikit-learn pandas matplotlib joblib # run to install modules

# Cell 2: Import libraries
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from google.colab import files
import joblib

#load data
with open('/content/drive/MyDrive/public_cases.json', 'r') as f:
    raw_data = json.load(f)

df = pd.json_normalize(raw_data, sep='_')

#extract features and target
X_raw = df[['input_trip_duration_days', 'input_miles_traveled', 'input_total_receipts_amount']]
y = df['expected_output']

#rename columns for ease
X_raw.columns = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']

#feature engineering
X = X_raw.copy()
X['receipts_per_day'] = X['total_receipts_amount'] / X['trip_duration_days']
X['miles_per_day'] = X['miles_traveled'] / X['trip_duration_days']
X['miles_receipts_ratio'] = X['miles_traveled'] / (X['total_receipts_amount'] + 1e-6)
X['interaction_1'] = X['trip_duration_days'] * X['miles_per_day']
X['interaction_2'] = X['trip_duration_days'] * X['receipts_per_day']

#train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)

#train model
model = GradientBoostingRegressor(n_estimators=500, max_depth=4, learning_rate=0.05)
model.fit(X_train, y_train)

#output estimators
from google.colab import files  # Import files module for downloading

#save the model temporarily to extract estimators
joblib.dump(model, '/content/gbr_model.pkl')

#function to convert tree to a string representation
def tree_to_string(tree):
    tree_str = []
    tree_ = tree.tree_
    tree_str.append(f"children_left: {tree_.children_left.tolist()}")
    tree_str.append(f"children_right: {tree_.children_right.tolist()}")
    tree_str.append(f"feature: {tree_.feature.tolist()}")
    tree_str.append(f"threshold: {tree_.threshold.tolist()}")
    tree_str.append(f"value: {tree_.value.flatten().tolist()}")
    return "\n".join(tree_str)

#extract and save all estimators to a text file
with open('/content/drive/MyDrive/gbr_estimators.txt', 'w') as f:
    f.write(f"n_estimators: {model.n_estimators}\n")
    f.write(f"learning_rate: {model.learning_rate}\n")
    f.write(f"n_features: {X_train.shape[1]}\n")
    f.write(f"init_prediction: {model.init_.constant_[0][0]}\n")  # Save initial prediction for accuracy
    for i, estimator in enumerate(model.estimators_):
        f.write(f"\n[Estimator {i}]\n")
        f.write(tree_to_string(estimator[0]))  # estimator[0] because GradientBoostingRegressor wraps trees in a list

#save feature names for reference in predict.py
with open('/content/drive/MyDrive/feature_names.txt', 'w') as f:
    f.write("\n".join(X_train.columns))

files.download('/content/drive/MyDrive/gbr_estimators.txt')
files.download('/content/drive/MyDrive/feature_names.txt')
