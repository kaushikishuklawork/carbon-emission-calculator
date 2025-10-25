import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Load dataset for thresholds and defaults
df = pd.read_csv("Carbon emission - Sheet1f.csv")

# Load trained pipeline (preprocessing + model)
reg_model = joblib.load("carbon_model.pkl")

# Features your model expects
required_cols = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 
                 'Heating Energy Source', 'Transport', 'Vehicle Type']

# Compute dynamic thresholds
low_thresh = df['CarbonEmission'].quantile(0.33)
med_thresh = df['CarbonEmission'].quantile(0.66)

# Impact category function
def impact_category(value):
    if value < low_thresh:
        return "B1"
    elif value < med_thresh:
        return "B2"
    else:
        return "B3"

# Cluster averages
df['Impact'] = df['CarbonEmission'].apply(impact_category)
cluster_avg = df.groupby('Impact')['CarbonEmission'].mean().reset_index()

st.title("Carbon Footprint Impact Calculator 🌍")

# --- USER INPUTS ---
user_input = {}
for col in required_cols:
    user_input[col] = st.selectbox(col, df[col].unique())

# Convert to DataFrame and ensure all required columns exist
input_df = pd.DataFrame([user_input])
for col in required_cols:
    if col not in input_df.columns:
        input_df[col] = df[col].mode()[0]

# --- PREDICTION ---
carbon_pred = reg_model.predict(input_df)[0]
st.write(f"Predicted Carbon Emission: {carbon_pred:.2f} kg CO2")

# Determine impact category
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

chart = alt.Chart(cluster_avg).mark_bar().encode(
    x=alt.X('Impact:N', title='Cluster'),
    y=alt.Y('Average Emission (kg CO2):Q', title='Emission (kg CO2)'),
    color=alt.Color('Color:N', scale=None)
)

st.subheader("Your Emission vs Cluster Averages")
st.altair_chart(chart, use_container_width=True)

