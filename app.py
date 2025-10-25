import streamlit as st
import pandas as pd
import joblib
import altair as alt

# --- LOAD MODEL ---
MODEL_PATH = "carbon_model.pkl"

try:
    data = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Extract regression pipeline and clustering info
reg_model = data['regression']           # pipeline with preprocessing + RandomForest
cluster_summary = data['cluster_summary']  # dict with cluster averages

# Extract exact categorical and numerical columns from pipeline
preprocessor = reg_model.named_steps['preprocess']  # ColumnTransformer
categorical_cols = preprocessor.transformers_[0][2]  # list of categorical columns
numerical_cols = preprocessor.transformers_[1][2]    # list of numerical columns

# --- LOAD DATASET ---
df = pd.read_csv("Carbon emission - Sheet1f.csv")
target_col = 'CarbonEmission'

# --- DYNAMIC THRESHOLDS ---
low_thresh = df[target_col].quantile(0.33)
med_thresh = df[target_col].quantile(0.66)

def impact_category(value):
    if value < low_thresh:
        return "B1"
    elif value < med_thresh:
        return "B2"
    else:
        return "B3"

df['Impact'] = df[target_col].apply(impact_category)

# --- STREAMLIT APP ---
st.title("Carbon Footprint Impact Calculator 🌍")

# --- USER INPUTS ---
user_input = {}

# Categorical → dropdowns
for col in categorical_cols:
    if col in df.columns:
        options = sorted(df[col].dropna().unique())
    else:
        options = ['Unknown']  # fallback if column not in CSV
    user_input[col] = st.selectbox(col, options)

# Numerical → number_input
for col in numerical_cols:
    if col in df.columns:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
    else:
        min_val, max_val, mean_val = 0.0, 100.0, 50.0  # fallback defaults
    user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)

input_df = pd.DataFrame([user_input])
input_df = input_df[categorical_cols + numerical_cols]  # reorder exactly as pipeline expects

# --- PREDICTION ---
try:
    carbon_pred = reg_model.predict(input_df)[0]
    st.write(f"Predicted Carbon Emission: {carbon_pred:.2f} kg CO2")
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# --- IMPACT CATEGORY ---
if carbon_pred < low_thresh:
    impact = "B1 (Low Impact)"
elif carbon_pred < med_thresh:
    impact = "B2 (Medium Impact)"
else:
    impact = "B3 (High Impact)"

st.success(f"Your Impact Category: {impact}")
