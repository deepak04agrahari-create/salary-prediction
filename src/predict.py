import joblib
import numpy as np

MODEL_PATH = "models/salary_mlr.joblib"

# Load model and transformer
ct, model = joblib.load(MODEL_PATH)

# Example input
new_data = {
    'YearsExperience': [4],
    'Education': ['B.Tech'],
    'City': ['Noida'],
    'Role': ['Developer']
}

import pandas as pd
new_df = pd.DataFrame(new_data)

# Transform input
X_new = ct.transform(new_df)

# Predict salary
predicted_salary = model.predict(X_new)
print("Predicted Salary:", predicted_salary[0])
