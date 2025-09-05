import os
import sys
import pandas as pd
import streamlit as st
import joblib

# ✅ Add project root to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocess import Preprocessor

# App UI setup
st.set_page_config(page_title="Readmission Predictor", layout="wide")
st.title("🩺 EMR Readmission Predictor")

# File uploader
uploaded_file = st.file_uploader("📁 Upload your EMR CSV file", type="csv")

if uploaded_file:
    try:
        # Step 1: Load user data
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Data Preview")
        st.dataframe(df.head())

        # Step 2: Preprocess input
        preprocessor = Preprocessor()
        processed = preprocessor.preprocess_input(df)

        # Step 3: Load model and training columns
        model = joblib.load("models/readmission_model.pkl")
        training_columns = joblib.load("models/column_names.pkl")

        # Step 4: Align input with training columns
        for col in training_columns:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[training_columns]

        # Step 5: Predict
        predictions = model.predict(processed)

        # ✅ Add explanation column
        df['Readmission_Predicted'] = predictions
        df['Readmission_Label'] = df['Readmission_Predicted'].map({
            0: "Not Readmitted",
            1: "Readmitted (<30 Days)"
        })

        # Step 6: Display result
        st.success("✅ Prediction Completed!")
        st.subheader("📋 Prediction Results")
        st.dataframe(df[['Readmission_Predicted', 'Readmission_Label']])

    except Exception as e:
        st.error(f"🚫 Error occurred: {e}")

else:
    st.info("👈 Please upload a .csv file to begin.")
