

import streamlit as st
import numpy as np
import pickle

# --- Load model, scaler, and encoders ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

st.title("🚢 Titanic Survival Prediction App")

st.write("Enter passenger details below:")

# --- Input fields ---
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", label_encoders['Sex'].classes_)
age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0)
sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0)
parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=32.0)
embarked = st.selectbox("Port of Embarkation", label_encoders['Embarked'].classes_)

# --- Encode categorical inputs ---
sex_encoded = label_encoders['Sex'].transform([sex])[0]
embarked_encoded = label_encoders['Embarked'].transform([embarked])[0]

# --- Prepare input for model ---
input_features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])
input_scaled = scaler.transform(input_features)

# --- Predict ---
if st.button("Predict Survival"):
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.success(f"Prediction: Survived (Probability: {probability:.2%})")
    else:
        st.error(f"Prediction: Did NOT Survive (Probability: {1 - probability:.2%})")
