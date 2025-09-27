# app.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt

st.set_page_config(page_title="Carbon Footprint Predictor", layout="wide")

st.title("🌱 Carbon Footprint Predictor with Clustering & Visualization")

# ----------------------------
# Load models and preprocessing
# ----------------------------
models = joblib.load("carbon_model.pkl")
reg_model = models['regression']       # Regression model
kmeans_model = models['clustering']    # Clustering model
preprocessor = models['preprocessor']  # Preprocessing pipeline
cluster_summary = models['cluster_summary']  # Cluster info (dict)

# ----------------------------
# Load CSV for dropdowns
# ----------------------------
df = pd.read_csv("Carbon emission - Sheet1f.csv")

# Detect columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['CarbonEmission']]

# ----------------------------
# User Inputs
# ----------------------------
st.subheader("Categorical Inputs")
user_input = {}
for col in categorical_cols:
    options = df[col].unique().tolist()
    user_input[col] = st.selectbox(f"{col}", options)

st.subheader("Numeric Inputs")
for col in numeric_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

input_df = pd.DataFrame([user_input])

# ----------------------------
# Prediction & Clustering
# ----------------------------
if st.button("Predict Carbon Emission & Cluster"):

    # Preprocess input for clustering
    input_transformed = preprocessor.transform(input_df)

    # Regression prediction
    prediction = reg_model.predict(input_transformed)[0]

    # Clustering
    cluster_label = kmeans_model.predict(input_transformed)[0]

    st.success(f"Predicted Carbon Emission: {prediction:.2f}")
    st.info(f"Cluster Assignment: Cluster {cluster_label + 1}")

    # Show cluster summary
    if cluster_label in cluster_summary:
        summary = cluster_summary[cluster_label]
        st.write(f"**Cluster Summary:**")
        st.write(f"- Average Carbon Emission in Cluster: {summary['Average Carbon Emission']:.2f}")
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
            title=f"Your Carbon Emission vs Cluster {cluster_label + 1} Average"
        )

        st.altair_chart(chart, use_container_width=True)
