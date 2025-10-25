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
reg_model = data['regression']           # Pipeline with preprocessing + RandomForest
cluster_summary = data['cluster_summary']  # Dict with cluster averages

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

input_df = pd.DataFrame([user_input])
input_df = input_df[categorical_cols + numerical_cols]  # reorder to match training

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

# --- CLUSTER COMPARISON ---
cluster_data = pd.DataFrame([
    {'Cluster': f"Cluster {k}", 'Average Emission': v['Average Carbon Emission']} 
    for k, v in cluster_summary.items()
])
cluster_data['User Emission'] = carbon_pred
cluster_data['Color'] = cluster_data['Average Emission'].apply(
    lambda x: 'green' if abs(x - carbon_pred) < 1e-6 else 'lightgray'
)

st.subheader("Your Emission vs Cluster Averages")
chart = alt.Chart(cluster_data).mark_bar().encode(
    x=alt.X('Cluster:N'),
    y=alt.Y('Average Emission:Q', title='Emission (kg CO2)'),
    color=alt.Color('Color:N', scale=None)
)
text = alt.Chart(cluster_data).mark_text(dy=-5, color='black').encode(
    x='Cluster:N',
    y='Average Emission:Q',
    text=alt.Text('User Emission:Q', format=".2f")
)
st.altair_chart(chart + text, use_container_width=True)
