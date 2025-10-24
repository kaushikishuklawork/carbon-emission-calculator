import streamlit as st
import pandas as pd
import joblib

# Load trained models
pipeline = joblib.load("carbon_model.pkl")  # full preprocessing + regression model
cluster_model = joblib.load("cluster_model.pkl")

# All columns required by the model
required_cols = ['Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source', 'Transport',
                 'Vehicle Type', 'Social Activity', 'Monthly Grocery Bill',
                 'Frequency of Traveling by Air', 'Vehicle Monthly Distance Km', 'Waste Bag Size',
                 'Waste Bag Weekly Count', 'How Long TV PC Daily Hour',
                 'How Many New Clothes Monthly', 'How Long Internet Daily Hour',
                 'Energy efficiency', 'Recycling', 'Cooking_With']

st.title("🌍 Carbon Emission Calculator & Lifestyle Cluster")

st.write("Estimate your carbon footprint and find your sustainability lifestyle category!")

# UI Inputs
user_inputs = {}
for col in required_cols:
    if col in ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 'Waste Bag Weekly Count',
               'How Long TV PC Daily Hour', 'How Many New Clothes Monthly',
               'How Long Internet Daily Hour']:
        user_inputs[col] = st.number_input(f"{col}", min_value=0.0)
    else:
        user_inputs[col] = st.selectbox(f"{col}", [
            "Unknown", "Low", "Medium", "High"
        ])

if st.button("Calculate 🌱"):
    # Create a DataFrame with all required columns
    input_df = pd.DataFrame([user_inputs])

    # Add missing columns if deployment environment changes
    for col in required_cols:
        if col not in input_df.columns:
            input_df[col] = "Unknown"

    # Predict carbon emission
    prediction = pipeline.predict(input_df)[0]
    prediction = round(prediction, 2)

    # Predict cluster
    cluster = cluster_model.predict(input_df)[0]

    # Map clusters
    cluster_labels = {
        0: "Low 🌱",
        1: "Medium 🌍",
        2: "High 🔥"
    }
    cluster_name = cluster_labels.get(cluster, "Unknown Category")

    st.subheader("📊 Your Results")
    st.write(f"**Carbon Emission:** {prediction} kg CO₂/year")
    st.write(f"**Lifestyle Category:** {cluster_name}")

    if cluster == 0:
        st.success("Amazing! You live a very eco-friendly life 🌱💚")
    elif cluster == 1:
        st.warning("Average impact! Small improvements can make a big difference 🌍✨")
    else:
        st.error("High environmental impact! Try reducing consumption 🔥🌡️")






