import streamlit as st
import pandas as pd
import joblib
import os

# --- Paths ---
DATA_PATH = os.path.join("dataset", "salary_data_extended.csv")
MODEL_PATH = os.path.join("models", "salary_mlr.joblib")

# --- Load dataset for dropdowns ---
df = pd.read_csv(DATA_PATH)

st.title("Salary Prediction App")

# --- User inputs ---
experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
education = st.selectbox("Education", sorted(df['Education'].unique()))
city = st.selectbox("City", sorted(df['City'].unique()))
role = st.selectbox("Role", sorted(df['Role'].unique()))

# --- Load model ---
if st.button("Predict Salary"):
    try:
        ct, model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"Model file not found at {MODEL_PATH}")
    else:
        # Prepare input
        input_df = pd.DataFrame({
            "YearsExperience": [experience],
            "Education": [education],
            "City": [city],
            "Role": [role]
        })

        # Transform and predict
        X_new = ct.transform(input_df)
        prediction = model.predict(X_new)[0]

        st.success(f"Estimated Salary: ${prediction:,.2f}")

# --- Optionally show dataset ---
if st.checkbox("Show Dataset"):
    st.dataframe(df.head())
