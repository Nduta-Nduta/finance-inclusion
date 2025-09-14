import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('Financial_inclusion_dataset.csv')

# Clean column names
df.columns = [col.strip().lower() for col in df.columns]

# Detect target column
target_candidates = [col for col in df.columns if 'cellphone' in col and 'access' in col]
if not target_candidates:
    raise ValueError("No target column containing 'cellphone' and 'access' found.")
target_col = target_candidates[0]
print(f"Detected target column: {target_col}")

# Generate profiling report
profile = ProfileReport(df, title="Financial Inclusion Profiling Report")
profile.to_file("financial_inclusion_report.html")
print("Profile report saved as financial_inclusion_report.html")

# Handle missing values
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicates
df = df.drop_duplicates()

# Handle outliers (IQR)
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR))]

# Split target and features BEFORE encoding
y = df[target_col]
X = df.drop(target_col, axis=1)

# Encode categorical features
X = pd.get_dummies(X, drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
with open('financial_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model saved as financial_model.pkl")



