import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Load data
df = pd.read_csv("adult.csv", header=None)
df.columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income"
]

# Strip whitespace
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)

# Drop unused columns
df = df.drop(columns=["fnlwgt", "education-num", "capital-gain", "capital-loss"])

# Encode categorical columns
encoders = {}
for column in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    encoders[column] = le

# Train model
X = df.drop("income", axis=1)
y = df["income"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
print("✅ Model and encoders saved.")
