# app.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# -------------------------------
# Load trained models
# -------------------------------
models = joblib.load(r"C:\Users\lapmall\Desktop\CE PROJECT\carbon_model_3clusters.pkl")
reg_model = models['regression']
kmeans_model = models['clustering']
preprocessor = models['preprocessor']
cluster_summary = models['cluster_summary']

# -------------------------------
# Load CSV for dropdowns
# -------------------------------
df = pd.read_csv(r"C:\Users\lapmall\Desktop\CE PROJECT\Carbon emission - Sheet1f.csv")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['CarbonEmission']]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Carbon Footprint Predictor with Clustering & Visualization")

# User inputs
user_input = {}

# Categorical Inputs (Dropdowns restricted to training values)
st.subheader("Categorical Inputs")
for col in categorical_cols:
    options = df[col].dropna().unique().tolist()
    user_input[col] = st.selectbox(f"{col}", options)

# Numeric Inputs
st.subheader("Numeric Inputs")
for col in numeric_cols:
    min_val = float(df[col].min())
    max_val = float(df[col].max())
    mean_val = float(df[col].mean())
    user_input[col] = st.number_input(f"{col}", min_value=min_val, max_value=max_val, value=mean_val)

# Convert input to DataFrame
input_df = pd.DataFrame([user_input])

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict Carbon Emission & Cluster"):
    # Ensure input_df has all columns expected by preprocessor
    expected_cols = getattr(preprocessor, "feature_names_in_", categorical_cols + numeric_cols)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0 if col in numeric_cols else ""
    input_df = input_df[expected_cols]  # Reorder columns

    # -------------------------------
    # Predict carbon emission
    # -------------------------------
    try:
        prediction = reg_model.predict(input_df)[0]
    except ValueError as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # -------------------------------
    # Predict cluster
    # -------------------------------
    input_transformed = preprocessor.transform(input_df)
    cluster_label = kmeans_model.predict(input_transformed)[0]

    # -------------------------------
    # Map 3 clusters to friendly names
    # -------------------------------
    cluster_labels = {
        0: "Low Impact 🌱",
        1: "Medium Impact 🌍",
        2: "High Impact 🔥"
    }
    cluster_name = cluster_labels.get(cluster_label, "Unknown Impact")

    st.success(f"Predicted Carbon Emission: {prediction:.2f}")
    st.info(f"Lifestyle Category: {cluster_name}")

    # -------------------------------
    # Cluster summary (without number of people)
    # -------------------------------
    summary = cluster_summary.get(cluster_label, {"Average Carbon Emission": 0})
    st.write(f"**Cluster Summary:**")
    st.write(f"- Average Carbon Emission in Cluster: {summary['Average Carbon Emission']:.2f}")

    # -------------------------------
    # Advice messages
    # -------------------------------
    advice_messages = {
        "Low Impact 🌱": "Amazing! You live a very eco-friendly life 🌱💚",
        "Medium Impact 🌍": "Average impact! Small improvements can make a big difference 🌍✨",
        "High Impact 🔥": "High environmental impact! Try reducing unnecessary energy, travel, or waste 🔥🌡️"
    }
    advice = advice_messages.get(cluster_name, "")
    if advice:
        if cluster_name == "Low Impact 🌱":
            st.success(advice)
        elif cluster_name == "Medium Impact 🌍":
            st.warning(advice)
        else:
            st.error(advice)

    # -------------------------------
    # Visualization: user vs cluster
    # -------------------------------
    vis_df = pd.DataFrame({
        'Type': ['Cluster Average', 'Your Prediction'],
        'CarbonEmission': [summary['Average Carbon Emission'], prediction]
    })

    chart = alt.Chart(vis_df).mark_bar(color='steelblue').encode(
        x='Type',
        y='CarbonEmission'
    ).properties(
        title=f"Your Carbon Emission vs {cluster_name} Average"
    )

    st.altair_chart(chart, use_container_width=True)

