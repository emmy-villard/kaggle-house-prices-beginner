# Prepare and save a scikit-learn preprocessor:
# - impute missing values
# - one-hot encode categorical features
# - pass through numeric features
# Fit on the training set, then serialize with joblib.

# Import libs
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Import dataset
dataset = pd.read_csv('data/train.csv')
dataset = dataset.iloc[:, :-1]

# Create and fit the OneHotEncoder
cat_cols = dataset.select_dtypes(include=['str']).columns
num_cols = dataset.columns.difference(cat_cols)

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), )
])

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

preprocess = ColumnTransformer([
    ("cat", cat_pipe, cat_cols),
    ("num", num_pipe, num_cols)
])

preprocess.fit(dataset)
preprocess.set_output(transform="pandas")
joblib.dump(preprocess, "preprocessing/preprocess.joblib")