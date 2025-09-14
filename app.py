import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier

st.title("Cellphone Access Prediction App")
st.write("Fill in the details below to predict cellphone access:")

# Load dataset
df = pd.read_csv('Financial_inclusion_dataset.csv')
df.columns = [col.strip().lower() for col in df.columns]

# Detect target column
target_candidates = [col for col in df.columns if 'cellphone' in col and 'access' in col]
if not target_candidates:
    st.error("No target column containing 'cellphone' and 'access' found.")
    st.stop()
target_col = target_candidates[0]

# Check if model exists
model_file = 'financial_model.pkl'
if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    st.info("Loaded existing model.")
else:
    st.warning("Model not found. Training a small model for testing...")

    # Handle missing values
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle outliers
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR))]

    # Split target and features
    y = df[target_col]
    X = df.drop(target_col, axis=1)

    # Encode categorical features
    X = pd.get_dummies(X, drop_first=True)

    # Train a **smaller Random Forest**
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X, y)

    # Save model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    st.success("Small model trained and saved as financial_model.pkl.")

# Features for input
features = [col for col in df.columns if col != target_col]

# Create dynamic input fields
input_data = {}
for col in features:
    if df[col].dtype == 'object':
        options = df[col].dropna().unique().tolist()
        input_data[col] = st.selectbox(f"{col}", options)
    else:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        input_data[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Predict button
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    input_df = pd.get_dummies(input_df)

    # Align columns with training features
    missing_cols = set(model.feature_names_in_) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0
    input_df = input_df[model.feature_names_in_]

    # Make prediction
    prediction = model.predict(input_df)
    st.success(f"Predicted cellphone access: {prediction[0]}")



