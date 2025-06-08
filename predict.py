import numpy as np
import pandas as pd
import sys

class TreePredictor:
    def __init__(self, children_left, children_right, feature, threshold, value):
        self.children_left = np.array(children_left)
        self.children_right = np.array(children_right)
        self.feature = np.array(feature)
        self.threshold = np.array(threshold)
        self.value = np.array(value)

    def predict(self, x):
        node = 0
        while True:
            if self.children_left[node] == -1 and self.children_right[node] == -1:
                return self.value[node]
            feature_idx = self.feature[node]
            if feature_idx < 0:
                return self.value[node]
            if x[feature_idx] <= self.threshold[node]:
                node = self.children_left[node]
            else:
                node = self.children_right[node]

def load_estimators(filename):
    estimators = []
    init_prediction = 0.0
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        raise
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("n_estimators:"):
            n_estimators = int(line.split(":")[1])
        elif line.startswith("learning_rate:"):
            learning_rate = float(line.split(":")[1])
        elif line.startswith("n_features:"):
            n_features = int(line.split(":")[1])
        elif line.startswith("init_prediction:"):
            init_prediction = float(line.split(":")[1])
        elif line.startswith("[Estimator"):
            estimator_data = {}
            i += 1
            while i < len(lines) and not lines[i].startswith("[Estimator"):
                try:
                    key, value = lines[i].strip().split(": ", 1)
                    estimator_data[key] = eval(value)
                    i += 1
                except Exception as e:
                    raise
            estimators.append(TreePredictor(
                estimator_data["children_left"],
                estimator_data["children_right"],
                estimator_data["feature"],
                estimator_data["threshold"],
                estimator_data["value"]
            ))
            continue
        i += 1
    return estimators, learning_rate, n_estimators, init_prediction

def load_feature_names(filename):
    try:
        with open(filename, 'r') as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        raise

def predict_reimbursement(input_data, estimators_file='gbr_estimators.txt', feature_names_file='feature_names.txt'):
    # load estimators and metadata
    estimators, learning_rate, n_estimators, init_prediction = load_estimators(estimators_file)
    feature_names = load_feature_names(feature_names_file)
    
    # convert input to DataFrame
    if isinstance(input_data, dict):
        X_raw = pd.DataFrame([input_data])
    else:
        X_raw = input_data.copy()
    
    # validate input
    required_columns = ['trip_duration_days', 'miles_traveled', 'total_receipts_amount']
    if not all(col in X_raw.columns for col in required_columns):
        raise ValueError(f"Input must contain columns: {required_columns}")
    
    if (X_raw['trip_duration_days'] <= 0).any():
        raise ValueError("trip_duration_days must be positive")
    
    # feature engineering
    X = X_raw[required_columns].copy()
    X['receipts_per_day'] = X['total_receipts_amount'] / X['trip_duration_days']
    X['miles_per_day'] = X['miles_traveled'] / X['trip_duration_days']
    X['miles_receipts_ratio'] = X['miles_traveled'] / (X['total_receipts_amount'] + 1e-6)
    X['interaction_1'] = X['trip_duration_days'] * X['miles_per_day']
    X['interaction_2'] = X['trip_duration_days'] * X['receipts_per_day']
    
    # reorder feature
    try:
        X = X[feature_names]
    except Exception as e:
        raise
    
    # make prediction
    predictions = np.zeros(len(X))
    for i, estimator in enumerate(estimators):
        try:
            tree_pred = np.array([estimator.predict(x) for x in X.values])
            predictions += learning_rate * tree_pred
        except Exception as e:
            raise
    
    predictions += init_prediction
    
    return round(predictions[0], 2) if len(predictions) == 1 else np.round(predictions, 2)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    
    try:
        trip_duration_days = float(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
    except ValueError:
        sys.exit(1)
    
    sample_input = {
        'trip_duration_days': trip_duration_days,
        'miles_traveled': miles_traveled,
        'total_receipts_amount': total_receipts_amount
    }
    
    try:
        prediction = predict_reimbursement(sample_input)
        print(f"{prediction:.2f}")
        with open('private_answers.txt', 'a') as f:
                f.write(f"{prediction:.2f}\n")
    except Exception as e:

        sys.exit(1)