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
st.set_page_config(page_title="CPTu to Vs Predictor for Dhaka", layout="wide")

st.title("Advanced Predictive Modeling of Shear Wave Velocity of Dhaka Soils from High-Resolution CPTu data using Advanced Algorithms")
st.write("Geotechnical Predictor for Dhaka Soil: CPTu to Vs")
st.write("Model: LightGBM_Augmented Linear_4F")

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
        feat1 = st.number_input("Input Feature 1 (Depth, m)", value=0.0, format="%.4f")
        feat2 = st.number_input("Input Feature 2 (fs, kPa)", value=0.0, format="%.4f")
    with col2:
        feat3 = st.number_input("Input Feature 3 (u2, Kpa)", value=0.0, format="%.4f")
        feat4 = st.number_input("Input Feature 4 (qt, kPa)", value=0.0, format="%.4f")

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
    st.write("Upload a CSV file containing your CPTu data. Map your columns to the required inputs. The dataset shall consist of Depth in meters, qt in kPa, fs in kPa, and u2 in kPa, with Vs expressed in m/s.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.dataframe(df.head())

        st.subheader("Map Your Columns")
        st.write("Match your CSV columns to the model's exact feature order.")

        # Create 4 specific dropdowns to ensure exact ordering
        colA, colB = st.columns(2)
        options = ["<Select>"] + df.columns.tolist()

        with colA:
            col_feat1 = st.selectbox("Select Feature 1 (Depth, m)", options)
            col_feat2 = st.selectbox("Select Feature 2 (fs, kPa)", options)
        with colB:
            col_feat3 = st.selectbox("Select Feature 3 (u2, Kpa)", options)
            col_feat4 = st.selectbox("Select Feature 4 (qt, kPa)", options)

        # Only proceed if all 4 dropdowns have been mapped by the user
        if "<Select>" not in [col_feat1, col_feat2, col_feat3, col_feat4]:

            # 1. Grab exactly the columns selected, in the exact order needed
            selected_cols = [col_feat1, col_feat2, col_feat3, col_feat4]

            # 2. Extract ONLY the numbers using .values. This strips the header names away!
            X_batch_values = df[selected_cols].values

            # 3. Scale and Predict
            X_batch_scaled = scaler.transform(X_batch_values)
            predictions = model.predict(X_batch_scaled)

            # 4. Add predictions back to the dataframe
            df['Predicted_Vs'] = predictions

            st.success("Batch Prediction Complete!")
            st.dataframe(df[[*selected_cols, 'Predicted_Vs']])

            # --- Accuracy Evaluation ---
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
                # Calculate VAF
                vaf = (1 - (np.var(y_true - predictions) / np.var(y_true))) * 100

                # Display metrics above the chart (Optional, but good for UI)
                st.write("**Evaluation Metrics:**")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("R² Score", f"{r2:.4f}")
                m2.metric("RMSE", f"{rmse:.4f}")
                m3.metric("MAE", f"{mae:.4f}")
                m4.metric("VAF", f"{vaf:.2f}%")

                # Generate Hexbin Plot
                fig, ax = plt.subplots(figsize=(7, 5))
                hb = ax.hexbin(y_true, predictions, gridsize=40, cmap='viridis', mincnt=1)
                cb = plt.colorbar(hb, ax=ax)
                cb.set_label('Count of Data Points')

                min_val = min(y_true.min(), predictions.min())
                max_val = max(y_true.max(), predictions.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='1:1 Ideal Fit')

                # --- NEW: Add the Metrics Text Box to the Plot ---
                metrics_text = (f"R² = {r2:.4f}\n"
                                f"RMSE = {rmse:.4f}\n"
                                f"MAE = {mae:.4f}\n"
                                f"VAF = {vaf:.2f}%")

                # Create a semi-transparent white box for readability
                props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')

                # Place text at 5% on the X axis, 95% on the Y axis (Top Left)
                ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)
                # ------------------------------------------------

                ax.set_title('Predicted vs. Measured Vs')
                ax.set_xlabel('Measured Vs (m/s)')
                ax.set_ylabel('Predicted Vs (m/s)')

                # Move legend to the lower right so it doesn't overlap the new text box
                ax.legend(loc='lower right')
                ax.grid(True, linestyle='--', alpha=0.3)

                st.pyplot(fig)