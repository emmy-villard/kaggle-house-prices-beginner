# Load a raw CSV, apply the saved preprocessor (imputation + one-hot),
# then save a new CSV with the preprocessed data.
# Note: run build_preprocessor.py first to create preprocess.joblib.

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

preprocess = joblib.load("preprocessing/preprocess.joblib")

dataset = pd.read_csv('data/train.csv')
to_predict = pd.read_csv('data/test.csv')

X_dataset = dataset.iloc[:, :-1]
y_dataset = pd.DataFrame(dataset.iloc[:, -1])

X_dataset = preprocess.transform(dataset)
X_to_predict = preprocess.transform(to_predict)

X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size = 0.2, random_state = 0)

# Save on new files
X_train.to_csv("data/processed_data/X_train.csv")
X_test.to_csv("data/processed_data/X_test.csv")
y_train.to_csv("data/processed_data/y_train.csv")
y_test.to_csv("data/processed_data/y_test.csv")
X_to_predict.to_csv("data/processed_data/to_predict.csv")