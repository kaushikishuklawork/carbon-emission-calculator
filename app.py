import streamlit as st
import pandas as pd
import joblib

# Load model package (regressor + kmeans + preprocessor + summary)
models = joblib.load("carbon_model.pkl")
reg_model = models['regression']
kmeans_model = models['clustering']
preprocessor = models['preprocessor']
cluster_summary = models['cluster_summary']

# Load CSV for dropdown data
df = pd.read_csv("Carbon emission - Sheet1f.csv")

# Feature columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['CarbonEmission']]

st.title("🌍 Carbon Emission Calculator & Lifestyle Cluster")
st.write("Estimate your carbon footprint and discover your sustainability lifestyle category! ♻️")

# User Inputs
user_inputs = {}

st.subheader("📌 Categorical Inputs")
for col in categorical_cols:
    options = df[col].dropna().unique().tolist()
    user_inputs[col] = st.selectbox(f"{col}", options)

st.subheader("📌 Numeric Inputs")
for col in numeric_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_inputs[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

if st.button("Calculate ✅"):
    # Convert user input to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # Ensure all expected columns exist
    expected_cols = categorical_cols + numeric_cols
    for col in expected_cols:
        if col not in input_df.columns:
            # Fill missing categorical with "", numeric with 0
            if col in numeric_cols:
                input_df[col] = 0
            else:
                input_df[col] = ""

    # Reorder columns to match training
    input_df = input_df[expected_cols]

    # Predict carbon footprint
    carbon_pred = reg_model.predict(input_df)[0]
    carbon_pred = round(carbon_pred, 2)

    # Cluster prediction
    X_processed = preprocessor.transform(input_df)
    cluster = kmeans_model.predict(X_processed)[0]

    # Friendly labels
    cluster_labels = {
        0: "Low Impact 🌱",
        1: "Medium Impact 🌍",
        2: "High Impact 🔥"
    }
    cluster_name = cluster_labels.get(cluster, "Unknown")

    st.subheader("📊 Your Sustainability Insights")
    st.write(f"**💨 Carbon Emission:** `{carbon_pred} kg CO₂/year`")
    st.write(f"**🏷 Lifestyle Category:** {cluster_name}")

    # Guidance
    if cluster == 0:
        st.success("Amazing! You live a very eco-friendly life 🌱💚")
    elif cluster == 1:
        st.warning("Average impact! Small improvements can make a big difference 🌍✨")
    else:
        st.error("High environmental impact! Try reducing unnecessary energy, travel or waste 🔥🌡️")
