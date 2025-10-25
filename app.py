import streamlit as st
import pandas as pd
import joblib
import altair as alt
import os

# --- FILE AND MODEL PATH ---
MODEL_PATH = "carbon_model.pkl"

# --- SAFE MODEL LOADING ---
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Please upload the correct pipeline file.")
    st.stop()
else:
    try:
        reg_model = joblib.load(MODEL_PATH)
        st.success("Model loaded successfully ✅")
    except EOFError:
        st.error("Model file is corrupted (EOFError). Please re-save the pipeline.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error while loading model: {e}")
        st.stop()

# --- LOAD DATASET ---
df = pd.read_csv("Carbon emission - Sheet1f.csv")

# --- FEATURE TYPES ---
target_col = 'CarbonEmission'
categorical_cols = [col for col in df.columns if df[col].dtype == 'object' and col != target_col]
numerical_cols = [col for col in df.columns if df[col].dtype != 'object' and col != target_col]

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
cluster_avg = df.groupby('Impact')[target_col].mean().reset_index()

# --- STREAMLIT APP ---
st.title("Carbon Footprint Impact Calculator 🌍")

# --- USER INPUTS ---
user_input = {}

# Categorical → dropdowns
for col in categorical_cols:
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.selectbox(col, options)

# Numerical → number_input
for col in numerical_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.number_input(col, min_value=min_val, max_value=max_val, value=mean_val)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])
input_df = input_df[categorical_cols + numerical_cols]  # reorder

# --- PREDICTION (pipeline handles preprocessing) ---
try:
    carbon_pred = reg_model.predict(input_df)[0]
    st.write(f"Predicted Carbon Emission: {carbon_pred:.2f} kg CO2")
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Determine impact
if carbon_pred < low_thresh:
    impact = "B1 (Low Impact)"
elif carbon_pred < med_thresh:
    impact = "B2 (Medium Impact)"
else:
    impact = "B3 (High Impact)"

st.success(f"Your Impact Category: {impact}")

# --- COMPARISON BAR CHART ---
cluster_avg['User Emission'] = carbon_pred
cluster_avg['Color'] = cluster_avg['Impact'].apply(lambda x: 'green' if x in impact else 'lightgray')

st.subheader("Your Emission vs Cluster Averages")
chart = alt.Chart(cluster_avg).mark_bar().encode(
    x=alt.X('Impact:N', title='Cluster'),
    y=alt.Y(f'{target_col}:Q', title='Emission (kg CO2)'),
    color=alt.Color('Color:N', scale=None)
)
text = alt.Chart(cluster_avg).mark_text(dy=-5, color='black').encode(
    x='Impact:N',
    y=alt.Y(f'{target_col}:Q'),
    text=alt.Text('User Emission:Q', format=".2f")
)
st.altair_chart(chart + text, use_container_width=True)
