import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler

# Load the Light GBM Classifier model
with open('finalized_model.sav', 'rb') as model_file:
    model = pickle.load(model_file)

# Instantiate StandardScaler
scaler = StandardScaler()

# Set page title and header
st.title("SaluSite – Diabetes Risk Calculator")
st.header("Background Information")
st.write("Welcome to Salusite - Diabetes Risk Calculator. This application is designed to help assess the risk of diabetes based on various factors.")

# Show statistics of Diabetes
diabetes_statistics = {
    "Prevalence": "9.3% of the US population have diabetes",
    "Type 2 Diabetes": "Most common type, comprising 90-95% of all diabetes cases",
    "Risk Factors": "Family history, obesity, sedentary lifestyle, etc."
}

st.subheader("Diabetes Statistics")
for stat, value in diabetes_statistics.items():
    st.write(f"- {stat}: {value}")

# Add sliders for input parameters
hba1c_level = st.slider("HbA1c Level", min_value=3.5, max_value=9.0, step=0.1, value=6.0)
blood_glucose_level = st.slider("Blood Glucose Level", min_value=60.0, max_value=400.0, step=1.0, value=120.0)
ahd_level = st.slider("AHD Level", min_value=0, max_value=100, step=1, value=50)

# Add input fields for age, BMI, gender, and hypertension
age = st.number_input("Age", min_value=1, max_value=150, value=30, step=1)
height = st.number_input("Height (cm)", min_value=30, max_value=300, value=170, step=1)
weight = st.number_input("Weight (kg)", min_value=1, max_value=500, value=70, step=1)

gender_options = {"Female": 1, "Male": 0}
gender = st.selectbox("Gender", options=list(gender_options.keys()))

hypertension_options = {"Yes": 1, "No": 0}
hypertension = st.selectbox("Hypertension", options=list(hypertension_options.keys()))

# Convert height and weight to BMI
bmi = weight / ((height / 100) ** 2)

# Button to calculate diabetes risk
if st.button("Calculate Diabetes Risk"):
    # Preprocess input features
    features = [hba1c_level, ahd_level, blood_glucose_level, age, bmi, gender_options[gender], hypertension_options[hypertension]]
    
    # Try-catch block for safe model prediction
    try:
        # Scale the input features
        scaled_features = scaler.fit_transform([features])
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        
        # Convert prediction to high risk (1) or not high risk (0)
        risk_label = "High Risk" if prediction == 1 else "Not High Risk"
        
        # Display prediction
        st.write(f"Diabetes Risk Prediction: {risk_label}")
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
