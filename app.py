import streamlit as st
import pandas as pd
import joblib

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
cluster_avg = df.groupby('Impact')['CarbonEmission'].mean()

st.title("Carbon Footprint Impact Calculator 🌍")

# --- USER INPUTS ---
user_input = {}
for col in required_cols:
    # Use the unique values from the dataset as options
    user_input[col] = st.selectbox(col, df[col].unique())

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Ensure all required columns exist
for col in required_cols:
    if col not in input_df.columns:
        # Fill missing columns with most common value from dataset
        input_df[col] = df[col].mode()[0]

st.write("Input DataFrame for model:")
st.dataframe(input_df)

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

# --- COMPARISON WITH CLUSTER AVERAGES ---
st.subheader("Comparison with Average Emissions per Cluster")
comparison_df = pd.DataFrame({
    'Cluster': cluster_avg.index,
    'Average Emission (kg CO2)': cluster_avg.values
})

# Highlight user's cluster
comparison_df['Your Emission'] = carbon_pred
st.dataframe(comparison_df.style.apply(
    lambda x: ['background-color: lightgreen' if x['Cluster'] in impact else '' for i in x], axis=1
))
