import xgboost as xgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

X_train = pd.read_csv("data/processed_data/X_train.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")
y_train = pd.DataFrame(y_train.iloc[:, -1])

X_test = pd.read_csv("data/processed_data/X_test.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")
y_test = pd.DataFrame(y_test.iloc[:, -1])

# initialize data
model = xgb.XGBRegressor(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=200
)

# make the prediction using the resulting model
preds = model.predict(X_test)

print("MAE :", mean_absolute_error(y_test, preds))
print("R2:", r2_score(y_test, preds))

joblib.dump(model, "modeling/model.joblib")