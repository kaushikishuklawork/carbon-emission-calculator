import streamlit as st
import pandas as pd
import joblib
import altair as alt

# --- LOAD DATA AND MODEL ---
df = pd.read_csv("Carbon emission - Sheet1f.csv")
reg_model = joblib.load("carbon_model.pkl")  # pipeline: preprocessing + model

# --- DYNAMIC THRESHOLDS ---
low_thresh = df['CarbonEmission'].quantile(0.33)
med_thresh = df['CarbonEmission'].quantile(0.66)

def impact_category(value):
    if value < low_thresh:
        return "B1"
    elif value < med_thresh:
        return "B2"
    else:
        return "B3"

df['Impact'] = df['CarbonEmission'].apply(impact_category)
cluster_avg = df.groupby('Impact')['CarbonEmission'].mean().reset_index()

# --- STREAMLIT APP ---
st.title("Carbon Footprint Impact Calculator 🌍")

# Create dropdowns for user input (only options you allow)
user_input = {}
for col in reg_model.feature_names_in_:
    # Only allow values present in dataset for this column
    options = sorted(df[col].dropna().unique())
    user_input[col] = st.selectbox(col, options)

input_df = pd.DataFrame([user_input])

# Ensure columns match pipeline
input_df = input_df[reg_model.feature_names_in_]

# --- PREDICTION ---
carbon_pred = reg_model.predict(input_df)[0]
st.write(f"Predicted Carbon Emission: {carbon_pred:.2f} kg CO2")

# Impact category
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
    y=alt.Y('Average Emission (kg CO2):Q', title='Emission (kg CO2)'),
    color=alt.Color('Color:N', scale=None)
)

text = alt.Chart(cluster_avg).mark_text(
    dy=-5,
    color='black'
).encode(
    x='Impact:N',
    y='Average Emission (kg CO2):Q',
    text=alt.Text('User Emission:Q', format=".2f")
)

st.altair_chart(chart + text, use_container_width=True)



