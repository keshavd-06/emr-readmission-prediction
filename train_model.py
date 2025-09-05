import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Step 1: Load the dataset
df = pd.read_csv("data/diabetic_data.csv")
print("📥 Dataset loaded successfully.")

# Step 2: Convert readmission column to binary
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)
print("🔁 Converted 'readmitted' into binary labels (1 = <30 days).")

# Step 3: Drop non-useful ID columns
df.drop(columns=['encounter_id', 'patient_nbr'], inplace=True, errors='ignore')

# Step 4: Fill missing values
df.fillna("Unknown", inplace=True)

# Step 5: Encode categorical features
df = pd.get_dummies(df)
print(f"🧾 Data shape after encoding: {df.shape}")

# Step 6: Split into features and target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("📊 Split data into training and testing sets.")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/diabetic_data.csv")

# Clean and preprocess
df.drop(['weight', 'payer_code', 'medical_specialty', 'encounter_id', 'patient_nbr'], axis=1, inplace=True)
df = df.replace('?', pd.NA).dropna()
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Encode categorical
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop("readmitted", axis=1)
y = df["readmitted"]

# Apply SMOTE to balance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n✅ Model Performance:")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Precision:", round(precision_score(y_test, y_pred) * 100, 2), "%")
print("Recall:", round(recall_score(y_test, y_pred) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Save model and features
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/readmission_model.pkl")
joblib.dump(X.columns.tolist(), "models/column_names.pkl")
print("\n💾 Model and column names saved to /models")
