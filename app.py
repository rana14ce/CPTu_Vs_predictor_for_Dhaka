import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import zipfile

# --- Configuration ---
MODEL_ZIP_PATH = "LightGBM_Augmented-Linear_4F.zip"
MODEL_TXT_PATH = "LightGBM_Augmented-Linear_4F.txt"
SCALER_PATH = "Scaler_LightGBM_Augmented-Linear_4F.joblib"

# --- Page Setup ---
st.set_page_config(page_title="CPTu to Vs Predictor", layout="wide")
st.title("Geotechnical Predictor: CPTu to Vs")


# --- Load Model & Scaler ---
@st.cache_resource
def load_artifacts():
    try:
        # 1. Check if the text file exists. If not, unzip it!
        if not os.path.exists(MODEL_TXT_PATH):
            with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(".")

                # 2. Load the model and scaler normally
        model = lgb.Booster(model_file=MODEL_TXT_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()


model, scaler = load_artifacts()

# --- Application Tabs ---
tab1, tab2 = st.tabs(["Single Entry Prediction", "Batch Prediction & Evaluation (CSV Upload)"])

# ==========================================
# TAB 1: Single Entry Prediction
# ==========================================
with tab1:
    st.header("Single Data Point Input")
    st.write("Enter your 4 CPTu parameters below to predict the Shear Wave Velocity (Vs).")

    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.number_input("Input Feature 1 (Column B)", value=0.0, format="%.4f")
        feat2 = st.number_input("Input Feature 2 (Column D)", value=0.0, format="%.4f")
    with col2:
        feat3 = st.number_input("Input Feature 3 (Column E)", value=0.0, format="%.4f")
        feat4 = st.number_input("Input Feature 4 (Column F)", value=0.0, format="%.4f")

    if st.button("Predict Vs", type="primary"):
        input_data = np.array([[feat1, feat2, feat3, feat4]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        st.success("Prediction Complete!")
        st.metric(label="Predicted Vs (m/s)", value=f"{prediction:.2f}")

# ==========================================
# TAB 2: Batch Prediction & CSV Upload
# ==========================================
with tab2:
    st.header("Batch Prediction via CSV")
    st.write(
        "Upload a CSV file containing your CPTu data. Ensure the 4 input columns match the order used during training.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        st.subheader("Map Your Columns")
        input_cols = st.multiselect("Select the 4 input feature columns:", options=df.columns.tolist(),
                                    max_selections=4)

        if len(input_cols) == 4:
            X_batch = df[input_cols]
            X_batch_scaled = scaler.transform(X_batch)

            predictions = model.predict(X_batch_scaled)
            df['Predicted_Vs'] = predictions

            st.success("Batch Prediction Complete!")
            st.dataframe(df[[*input_cols, 'Predicted_Vs']])

            st.divider()
            st.subheader("Evaluate Model Accuracy")
            st.write(
                "If your uploaded CSV already contains actual Vs measurements, select the column below to evaluate accuracy.")

            target_col = st.selectbox("Select Target Vs Column (Leave blank if none)", ["<None>"] + df.columns.tolist())

            if target_col != "<None>":
                y_true = df[target_col]

                rmse = np.sqrt(mean_squared_error(y_true, predictions))
                mae = mean_absolute_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)

                st.write("**Evaluation Metrics:**")
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{rmse:.4f}")
                m2.metric("MAE", f"{mae:.4f}")
                m3.metric("R² Score", f"{r2:.4f}")

                fig, ax = plt.subplots(figsize=(7, 5))
                hb = ax.hexbin(y_true, predictions, gridsize=40, cmap='viridis', mincnt=1)
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Count')

                min_val = min(y_true.min(), predictions.min())
                max_val = max(y_true.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Ideal Fit')

                ax.set_title('Predicted vs. Measured Vs')
                ax.set_xlabel('Measured Vs')
                ax.set_ylabel('Predicted Vs')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.3)

                st.pyplot(fig)