import os
import joblib
import pandas as pd
import streamlit as st

# Explicit features matching standard Titanic dataset columns
FEATURES = ["Pclass", "Sex", "Age", "Fare"]

@st.cache_resource
def load_model():
    # Force regeneration of model binary to match Python 3.14.8 server environment
    if not os.path.exists("model_cloud.joblib"):
        st.info("🔄 Recompiling model structure for cloud server optimization... Please wait.")
        
        # Read your training file and inject a rewrite rule to compile cleanly on Python 3.14
        with open("generate_model.py", "r") as f:
            code = f.read()
        code = code.replace("model.joblib", "model_cloud.joblib")
        with open("generate_model_cloud.py", "w") as f:
            f.write(code)
            
        os.system("python generate_model_cloud.py")
        
    return joblib.load("model_cloud.joblib")

st.title("🚢 Titanic Passenger Survival Predictor")
st.write("An educational machine learning model predicting individual passenger survival metrics.")

# Input control widgets
pclass = st.selectbox("Ticket Class Status", [1, 2, 3], index=2)
sex_input = st.radio("Gender Profile", ["Male", "Female"], index=0, horizontal=True)
sex = 1 if sex_input == "Female" else 0
age = st.slider("Passenger Age (Years)", min_value=1, max_value=80, value=30)
fare = st.slider("Passenger Ticket Fare Paid (USD)", min_value=0, max_value=500, value=32)

# Keep model inference safely restricted to action clicks
if st.button("Calculate Survival Probability"):
    try:
        model = load_model()
        input_data = pd.DataFrame([[pclass, sex, age, fare]], columns=FEATURES)
        
        probabilities = model.predict_proba(input_data)
        survival_chance = float(probabilities[0][1])

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
