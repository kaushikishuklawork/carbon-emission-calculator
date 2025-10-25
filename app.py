import streamlit as st
import pandas as pd
import joblib

# Load your dataset (for thresholds and cluster averages)
df = pd.read_csv("Carbon emission - Sheet1f.csv")

# Calculate dynamic thresholds based on quantiles
low_thresh = df['CarbonEmission'].quantile(0.33)
med_thresh = df['CarbonEmission'].quantile(0.66)

# Assign impact category to each dataset entry
def impact_category(value):
    if value < low_thresh:
        return "B1"
    elif value < med_thresh:
        return "B2"
    else:
        return "B3"

df['Impact'] = df['CarbonEmission'].apply(impact_category)

# Compute average carbon emission per impact cluster
cluster_avg = df.groupby('Impact')['CarbonEmission'].mean()

# Load your trained regression model
reg_model = joblib.load("carbon_model.pkl")

st.title("Carbon Footprint Impact Calculator 🌍")

# Example user inputs
body_type = st.selectbox("Body Type", df['Body Type'].unique())
sex = st.selectbox("Sex", df['Sex'].unique())
diet = st.selectbox("Diet", df['Diet'].unique())
# … add other features here …

# Collect inputs
input_df = pd.DataFrame({
    'Body Type': [body_type],
    'Sex': [sex],
    'Diet': [diet],
    # … add other features …
})

# Predict carbon emission
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

# Compare to cluster averages
st.subheader("Comparison with Average Emissions per Cluster")
comparison_df = pd.DataFrame({
    'Cluster': cluster_avg.index,
    'Average Emission (kg CO2)': cluster_avg.values
})

# Highlight the user's cluster
comparison_df['Your Emission'] = carbon_pred
st.dataframe(comparison_df.style.apply(
    lambda x: ['background-color: lightgreen' if x['Cluster'] in impact else '' for i in x], axis=1
))


