# app.py
import streamlit as st
import pandas as pd
import joblib
import altair as alt

# -------------------------------
# Load trained models (3-cluster fixed)
# -------------------------------
models = joblib.load("carbon_model_3clusters_fixed.pkl")  # Use fixed 3-cluster model
reg_model = models['regression']
kmeans_model = models['clustering']
preprocessor = models['preprocessor']

# -------------------------------
# Load CSV for dropdowns
# -------------------------------
df = pd.read_csv("Carbon emission - Sheet1f.csv")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_cols = [col for col in df.columns if col not in categorical_cols + ['CarbonEmission']]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌍 Carbon Footprint Predictor with Lifestyle Clusters")

# User inputs
user_input = {}

# Categorical Inputs (dropdown only)
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

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# -------------------------------
# Prediction button
# -------------------------------
if st.button("Predict Carbon Emission & Cluster"):
    # Ensure all columns are present and in correct order
    expected_cols = getattr(preprocessor, "feature_names_in_", categorical_cols + numeric_cols)
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0 if col in numeric_cols else ""
    input_df = input_df[expected_cols]

    # Predict carbon emission
    prediction = reg_model.predict(input_df)[0]

    # Predict cluster
    input_transformed = preprocessor.transform(input_df)
    cluster_label = kmeans_model.predict(input_transformed)[0]

    # Map clusters to friendly names
    cluster_labels = {
        0: "Low Impact 🌱",
        1: "Medium Impact 🌍",
        2: "High Impact 🔥"
    }
    cluster_name = cluster_labels.get(cluster_label, "Unknown Impact")

    # Show results
    st.success(f"💨 Predicted Carbon Emission: {prediction:.2f} kg CO₂/year")
    st.info(f"🏷 Lifestyle Category: {cluster_name}")

    # -------------------------------
    # Cluster summary (average only)
    # -------------------------------
    df_aligned = df.copy()
    for col in expected_cols:
        if col not in df_aligned.columns:
            df_aligned[col] = 0 if col in numeric_cols else ""
    df_aligned = df_aligned[expected_cols]

    cluster_indices = df_aligned.index[
        kmeans_model.predict(preprocessor.transform(df_aligned)) == cluster_label
    ]
    cluster_data = df.loc[cluster_indices, 'CarbonEmission']

    avg_emission = cluster_data.mean()

    st.write(f"**Cluster Summary:**")
    st.write(f"- Average Carbon Emission: {avg_emission:.2f} kg CO₂/year")

    # Advice messages
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
    # Visualization: user vs cluster average with highlight
    # -------------------------------
    user_color = 'green' if prediction <= avg_emission else 'orange'

    vis_df = pd.DataFrame({
        'Type': ['Cluster Average', 'Your Prediction'],
        'CarbonEmission': [avg_emission, prediction],
        'Color': ['steelblue', user_color]
    })

    chart = alt.Chart(vis_df).mark_bar().encode(
        x='Type',
        y='CarbonEmission',
        color='Color'
    ).properties(
        title=f"Your Carbon Emission vs {cluster_name} Average"
    )

    st.altair_chart(chart, use_container_width=True)
