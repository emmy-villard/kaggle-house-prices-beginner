# Data Preprocessing
# launch python3 file from project root

# Importer les librairies
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importer le dataset
dataset = pd.read_csv('data/train.csv')
to_predict = pd.read_csv('data/test.csv')
y_dataset = dataset.iloc[:, -1].values

# Handle categorical variables
def handle_categorical_variable(pandas_df):
    categorical_cols = pandas_df.select_dtypes(include=['object', 'string']).columns
    return pd.get_dummies(pandas_df, columns=categorical_cols)

dataset = handle_categorical_variable(dataset)

to_predict = handle_categorical_variable(to_predict)
to_predict = to_predict.reindex(columns=dataset.columns[:-1], fill_value=0) # align test cols on train cols

X_dataset = dataset.iloc[:, :-1].values
X_to_predict_dataset = to_predict.values

# Handle missing values
imp_mean = SimpleImputer(strategy='mean')
X_dataset = imp_mean.fit_transform(X_dataset)
X_to_predict_dataset = imp_mean.transform(X_to_predict_dataset)


# Divide : Train and test set
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, test_size = 0.2, random_state = 0)


# Feature Scaling
sc = StandardScaler()
X_train_ = sc.fit_transform(X_train)
X_test_ = sc.fit_transform(X_test)
X_to_predict_dataset_ = sc.fit_transform(X_to_predict_dataset)


# Keep id intact
X_train_[:, 0] = X_train[:, 0]
X_test_[:, 0] = X_test[:, 0]
X_to_predict_dataset_[:, 0] = X_to_predict_dataset[:, 0]

# Save on new files
np.save("data/processed_data/X_train.npy", X_train)
np.save("data/processed_data/X_test.npy", X_test)
np.save("data/processed_data/y_train.npy", y_train)
np.save("data/processed_data/y_test.npy", y_test)
np.save("data/processed_data/to_predict.npy", to_predict)