import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Paths
DATA_PATH = "dataset/salary_data_extended.csv"
MODEL_PATH = "models/salary_mlr.joblib"

def main():
    # Load data
    data = pd.read_csv(DATA_PATH)
    print("Data preview:\n", data.head())

    # Separate features and target
    X = data[['YearsExperience', 'Education', 'City', 'Role']]
    y = data['Salary']

    # Define which columns are categorical
    categorical_features = ['Education', 'City', 'Role']

    # Apply OneHotEncoding to categorical features
    ct = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(drop='first'), categorical_features)
        ],
        remainder='passthrough'
    )

    X = ct.fit_transform(X)

    # Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
    print("R^2 Score:", r2_score(y_test, y_pred))

    # Save both model and transformer
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump((ct, model), MODEL_PATH)
    print("\nModel saved successfully!")

if __name__ == "__main__":
    main()
