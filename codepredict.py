import streamlit as st
import pandas as pd
import numpy as np
import pickle

filename = 'BestModel.pkl'
model = pickle.load(open(filename, 'rb'))

filename_gender_encode = 'binarygender.pkl'
gender_encode = pickle.load(open(filename_gender_encode, 'rb'))

filename_geo_encode = 'onehotgeo.pkl'
geo_encoder = pickle.load(open(filename_geo_encode, 'rb'))

robust_scaler = pickle.load(open('robust.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmax.pkl', 'rb'))

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

    credit_score = st.slider("Credit Score", 0, 1000)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 0, 100)
    tenure = st.slider("Tenure", 0, 100)
    balance = st.slider("Balance", 0, 100000)
    num_of_products = st.slider("Number of Products", 0, 10)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.slider("Estimated Salary", 0, 1000000)

    if st.button('Predict'):
        features = preprocess_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary)

        prediction = model.predict(features.values.reshape(1, -1))
        if prediction[0] == 0:
            result = 'Not Churn'
        else:
            result = 'Churn'

        st.success(f'The prediction is: {result}')

def preprocess_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
    gender_encoded = gender_encode["Gender"][gender]
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_member_encoded = 1 if is_active_member == "Yes" else 0
    geo_encoded = geo_encoder.transform([[geography]]).toarray()[0]

    age_scaled, credit_score_scaled = robust_scaler.transform([[age, credit_score]])[0]
    
    balance_scaled, estimated_salary_scaled = minmax_scaler.transform([[balance, estimated_salary]])[0]
    
    features = pd.DataFrame({
        'CreditScore': [credit_score_scaled],
        'Gender': [gender_encoded],
        'Age': [age_scaled],
        'Tenure': [tenure],
        'Balance': [balance_scaled],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card_encoded],
        'IsActiveMember': [is_active_member_encoded],
        'EstimatedSalary': [estimated_salary_scaled]
    })

    geo_df = pd.DataFrame({f'Geography_{col}': geo_encoded[i] for i, col in enumerate(['France', 'Germany', 'Spain'])}, index=[0])

    features = pd.concat([features, geo_df], axis=1)

    return features


if __name__ == '__main__':
    main()
