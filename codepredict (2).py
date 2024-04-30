import streamlit as st
import pandas as pd
import numpy as np
import pickle

model_filename = 'BestModel.pkl'
model = pickle.load(open(model_filename, 'rb'))

gender_encoder_filename = 'binarygender.pkl'
gender_encoder = pickle.load(open(gender_encoder_filename, 'rb'))

geo_encoder_filename = 'onehotgeo.pkl'
geo_encoder = pickle.load(open(geo_encoder_filename, 'rb'))

robust_scaler_filename = 'robust.pkl'
robust_scaler = pickle.load(open(robust_scaler_filename, 'rb'))

minmax_scaler_filename = 'minmax.pkl'
minmax_scaler = pickle.load(open(minmax_scaler_filename, 'rb'))

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #0052D4, #65C7F7, #9CECFB);
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def main():
    st.title('Churn Prediction App')

    credit_score_input = st.slider("Credit Score", 0, 1000)
    age_input = st.slider("Age", 0, 100)
    tenure_input = st.slider("Tenure", 0, 10)
    balance_input = st.slider("Balance", 0, 100000)
    num_of_products_input = st.slider("Number of Products", 0, 10)
    estimated_salary_input = st.slider("Estimated Salary", 0, 1000000)

    geography_input = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender_input = st.selectbox("Gender", ["Male", "Female"])
    has_cr_card_input = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member_input = st.selectbox("Is Active Member", ["Yes", "No"])

    if st.button('Predict'):
        features = preprocess_features(credit_score_input, age_input, tenure_input, balance_input, num_of_products_input, estimated_salary_input, geography_input, gender_input, has_cr_card_input, is_active_member_input)
        prediction = model.predict(features.values.reshape(1, -1))
        if prediction[0] == 0:
            result = 'Not Churn'
        else:
            result = 'Churn'
        st.success(f'The prediction is: {result}')

def preprocess_features(credit_score, age, tenure, balance, num_of_products, estimated_salary, geography, gender, has_cr_card, is_active_member):
    gender_encoded = gender_encoder["Gender"][gender]
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_member_encoded = 1 if is_active_member == "Yes" else 0
    geo_encoded = geo_encoder.transform([[geography]]).toarray()[0]

    age_scaled, credit_score_scaled = robust_scaler.transform([[age, credit_score]])[0]
    balance_scaled, estimated_salary_scaled = minmax_scaler.transform([[balance, estimated_salary]])[0]

    features = pd.DataFrame({
        'CreditScore': [credit_score_scaled],
        'Age': [age_scaled],
        'Tenure': [tenure],
        'Balance': [balance_scaled],
        'NumOfProducts': [num_of_products],
        'EstimatedSalary': [estimated_salary_scaled],
        'Geography_France': [geo_encoded[0]],
        'Geography_Germany': [geo_encoded[1]],
        'Geography_Spain': [geo_encoded[2]],
        'Gender': [gender_encoded],
        'HasCrCard': [has_cr_card_encoded],
        'IsActiveMember': [is_active_member_encoded]
    })

    return features

if __name__ == '__main__':
    main()
