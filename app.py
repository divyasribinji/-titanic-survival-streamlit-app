import os
import joblib
import pandas as pd
import streamlit as st

# FIX: Capitalized to match standard training dataset column headers
FEATURES = ["Pclass", "Sex", "Age", "Fare"]

@st.cache_resource
def load_model():
    # If the model file is missing or fails due to environment shifts, automatically regenerate it
    if not os.path.exists("model.joblib"):
        st.info("🔄 Building a clean model environment on the cloud server... please wait.")
        os.system("python generate_model.py")
        
    try:
        return joblib.load("model.joblib")
    except Exception:
        # Force-regenerate a fresh model file if a version error happens
        os.system("python generate_model.py")
        return joblib.load("model.joblib")

st.title("🚢 Titanic Passenger Survival Predictor")
st.write("An educational machine learning model predicting individual passenger survival metrics.")

# Input control widgets
pclass = st.selectbox("Ticket Class Status", [1, 2, 3], index=2)
sex_input = st.radio("Gender Profile", ["Male", "Female"], index=0, horizontal=True)
sex = 1 if sex_input == "Female" else 0
age = st.slider("Passenger Age (Years)", min_value=1, max_value=80, value=30)
fare = st.slider("Passenger Ticket Fare Paid (USD)", min_value=0, max_value=500, value=32)

try:
    model = load_model()
    if st.button("Calculate Survival Probability"):
        input_data = pd.DataFrame([[pclass, sex, age, fare]], columns=FEATURES)
        probabilities = model.predict_proba(input_data)
        survival_chance = probabilities[0][1]

        st.write("---")
        if survival_chance >= 0.50:
            st.success("Prediction outcome: High survival likelihood.")
        else:
            st.error("Prediction outcome: Low survival likelihood.")

        st.metric(label="Survival Probability Score", value=f"{survival_chance * 100:.1f}%")

except Exception as e:
    st.error(f"⚠️ App execution error: {e}")

st.write("---")
st.caption("ℹ️ Educational Demonstration: Built strictly for classroom analytics showcasing.")
