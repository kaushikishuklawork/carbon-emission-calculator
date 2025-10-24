
import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load models
# -------------------------------
# Regression pipeline now includes preprocessing
reg_pipeline = joblib.load("full_carbon_pipeline.pkl")  # regressor + preprocessor
kmeans_model = joblib.load("kmeans_model.pkl")          # clustering model
preprocessor = joblib.load("preprocessor.pkl")          # preprocessing only for clustering

# Load CSV for dropdowns
df = pd.read_csv("Carbon emission - Sheet1f.csv")

# Identify categorical and numeric columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['CarbonEmission']]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌍 Carbon Emission Calculator & Lifestyle Cluster")
st.write("Estimate your carbon footprint and discover your sustainability lifestyle category! ♻️")

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

# -------------------------------
# Prediction
# -------------------------------
if st.button("Calculate ✅"):
    # Convert to DataFrame
    input_df = pd.DataFrame([user_inputs])

    # -------------------------------
    # Handle missing columns and order
    # -------------------------------
    # Use preprocessor's expected features for clustering
    try:
        expected_cols = preprocessor.feature_names_in_
    except AttributeError:
        expected_cols = categorical_cols + numeric_cols

    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0 if col in numeric_cols else ""
    input_df = input_df[expected_cols]

    # -------------------------------
    # Regression prediction (pipeline handles preprocessing)
    # -------------------------------
    carbon_pred = reg_pipeline.predict(input_df)[0]
    carbon_pred = round(carbon_pred, 2)

    # -------------------------------
    # Clustering prediction
    # -------------------------------
    X_processed = preprocessor.transform(input_df)
    cluster = kmeans_model.predict(X_processed)[0]

    cluster_labels = {
        0: "Low Impact 🌱",
        1: "Medium Impact 🌍",
        2: "High Impact 🔥"
    }
    cluster_name = cluster_labels.get(cluster, "Unknown")

    # -------------------------------
    # Display results
    # -------------------------------
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
