# Load a raw CSV, apply the saved preprocessor (imputation + one-hot),
# then save a new CSV with the preprocessed data.
# Note: run build_preprocessor.py first to create preprocess.joblib.

import pandas as pd
import joblib

model = joblib.load("modeling/model.joblib")

to_predict = pd.read_csv('data/processed_data/to_predict.csv')
preds = model.predict(to_predict)

# Save predictions
submission = pd.DataFrame({"Id": to_predict.iloc[:, 0], "SalePrice": preds})
submission.to_csv("submission.csv", index=False)