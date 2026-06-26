import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import zipfile # <--- New import required for unzipping

# --- Configuration ---
MODEL_ZIP_PATH = "LightGBM_Augmented-Linear_4F.zip" # <--- Your uploaded ZIP file
MODEL_TXT_PATH = "LightGBM_Augmented-Linear_4F.txt" # <--- What it looks like inside the ZIP
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
                zip_ref.extractall(".") # Extracts into the current cloud folder
                
        # 2. Load the model and scaler normally
        model = lgb.Booster(model_file=MODEL_TXT_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

model, scaler = load_artifacts()

# ... (The rest of your Tabs and UI code stays exactly the same) ...