# app.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# Load trained models
models = joblib.load("carbon_model.pkl")
reg_model = models['regression']
kmeans_model = models['clustering']
preprocessor = models['preprocessor']
cluster_summary = models['cluster_summary']

# Load CSV for dropdowns
df = pd.read_csv("Carbon emission - Sheet1f.csv")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['CarbonEmission']]

# Map cluster numbers to names
cluster_names = {0: "High 🌱", 1: "Medium 🌿", 2: "Low 🌳"}

st.title("🌍 Carbon Footprint Predictor with Clustering & Visualization 🌿")

# User inputs
user_input = {}
st.subheader("Categorical Inputs")
for col in categorical_cols:
    options = df[col].astype(str).unique().tolist()
    user_input[col] = st.selectbox(f"{col}", options)

st.subheader("Numeric Inputs")
for col in numeric_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# Fix types
for col in categorical_cols:
    input_df[col] = input_df[col].astype(str)

for col in numeric_cols:
    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

input_df.fillna(0, inplace=True)

if st.button("Predict Carbon Emission & Cluster"):
    # Predict carbon emission
    prediction = reg_model.predict(input_df)[0]
    
    # Transform input for clustering
    input_transformed = preprocessor.transform(input_df)
    cluster_label = kmeans_model.predict(input_transformed)[0]
    
    st.success(f"Predicted Carbon Emission: {prediction:.2f} kg CO₂")
    st.info(f"Cluster Assignment: {cluster_names.get(cluster_label, f'Cluster {cluster_label+1}')}")

    # Show cluster summary
    summary = cluster_summary[cluster_label]
    st.write(f"**Cluster Summary:**")
    st.write(f"- Average Carbon Emission in Cluster: {summary['Average Carbon Emission']:.2f} kg CO₂")
    st.write(f"- Number of People in Cluster: {summary['Sample Size']}")
    
    # Visualization: user vs cluster average
    vis_df = pd.DataFrame({
        'Type': ['Cluster Average', 'Your Prediction'],
        'CarbonEmission': [summary['Average Carbon Emission'], prediction]
    })

    chart = alt.Chart(vis_df).mark_bar(color='steelblue').encode(
        x='Type',
        y='CarbonEmission'
    ).properties(
        title=f"🌷 Your Carbon Emission vs {cluster_names.get(cluster_label)} Average"
    )

    st.altair_chart(chart, use_container_width=True) 



