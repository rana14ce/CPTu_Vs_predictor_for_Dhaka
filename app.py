import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os

# --- Configuration ---
# Replace these with your actual saved file names
MODEL_PATH = "LightGBM_Augmented-Linear_4F.txt"
SCALER_PATH = "Scaler_LightGBM_Augmented-Linear_4F.joblib"

# --- Page Setup ---
st.set_page_config(page_title="CPTu to Vs Predictor for Dhaka", layout="wide")
st.title("Geotechnical Predictor for Dhaka Soil: CPTu to Vs")


# --- Load Model & Scaler ---
@st.cache_resource
def load_artifacts():
    try:
        model = lgb.Booster(model_file=MODEL_PATH)
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

    # Create input columns for a clean layout
    col1, col2 = st.columns(2)
    with col1:
        feat1 = st.number_input("Input Feature 1 (Depth, m)", value=0.0, format="%.4f")
        feat2 = st.number_input("Input Feature 2 (fs, kPa)", value=0.0, format="%.4f")
    with col2:
        feat3 = st.number_input("Input Feature 3 (u2, Kpa)", value=0.0, format="%.4f")
        feat4 = st.number_input("Input Feature 4 (qt, kPa)", value=0.0, format="%.4f")

    if st.button("Predict Vs", type="primary"):
        # Format input and scale
        input_data = np.array([[feat1, feat2, feat3, feat4]])
        input_scaled = scaler.transform(input_data)

        # Predict
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

        # Select input columns
        st.subheader("Map Your Columns")
        input_cols = st.multiselect("Select the 4 input feature columns: L_m--fs_kpa--u2_kpa--qt_kpa", options=df.columns.tolist(),
                                    max_selections=4)

        if len(input_cols) == 4:
            X_batch = df[input_cols]
            X_batch_scaled = scaler.transform(X_batch)

            # Predict
            predictions = model.predict(X_batch_scaled)
            df['Predicted_Vs'] = predictions

            st.success("Batch Prediction Complete!")
            st.dataframe(df[[*input_cols, 'Predicted_Vs']])

            # --- Optional Accuracy Evaluation ---
            st.divider()
            st.subheader("Evaluate Model Accuracy")
            st.write(
                "If your uploaded CSV already contains actual Vs measurements, select the column below to evaluate accuracy.")

            target_col = st.selectbox("Select Target Vs Column (Leave blank if none)", ["<None>"] + df.columns.tolist())

            if target_col != "<None>":
                y_true = df[target_col]

                # Calculate metrics
                rmse = np.sqrt(mean_squared_error(y_true, predictions))
                mae = mean_absolute_error(y_true, predictions)
                r2 = r2_score(y_true, predictions)

                st.write("**Evaluation Metrics:**")
                m1, m2, m3 = st.columns(3)
                m1.metric("RMSE", f"{rmse:.4f}")
                m2.metric("MAE", f"{mae:.4f}")
                m3.metric("R² Score", f"{r2:.4f}")

                # Generate Hexbin Plot
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