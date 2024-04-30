import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model_filename = 'BestModel.pkl'
model = pickle.load(open(model_filename, 'rb'))

# Load the encoder objects
gender_encoder_filename = 'binarygender.pkl'
gender_encoder = pickle.load(open(gender_encoder_filename, 'rb'))

geo_encoder_filename = 'onehotgeo.pkl'
geo_encoder = pickle.load(open(geo_encoder_filename, 'rb'))

# Load scalers
robust_scaler = pickle.load(open('robust.pkl', 'rb'))
minmax_scaler = pickle.load(open('minmax.pkl', 'rb'))

def main():
    st.title('Churn Prediction App')

    # User input
    credit_score = st.number_input("Credit Score", min_value=0, max_value=1000, step=1)
    geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=100, step=1)
    tenure = st.number_input("Tenure", min_value=0, max_value=100, step=1)
    balance = st.number_input("Balance", min_value=0, max_value=100000, step=100)
    num_of_products = st.number_input("Number of Products", min_value=0, max_value=10, step=1)
    has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
    is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", min_value=0, max_value=1000000, step=100)

    if st.button('Predict'):
        # Preprocess features
        features = preprocess_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary)

        # Make prediction
        prediction = model.predict(features.values.reshape(1, -1))
        result = 'Churn' if prediction[0] == 1 else 'Not Churn'
        st.success(f'The prediction is: {result}')

def preprocess_features(credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary):
    # Encode categorical features
    gender_encoded = gender_encoder['Gender'][gender]
    has_cr_card_encoded = 1 if has_cr_card == "Yes" else 0
    is_active_member_encoded = 1 if is_active_member == "Yes" else 0
    geo_encoded = geo_encoder.transform([[geography]]).toarray()[0]

    # Scale numerical features
    scaled_age_credit = robust_scaler.transform([[age, credit_score]])[0][0]
    
    # Gabungkan nilai balance dan estimated_salary menjadi satu array
    combined_features = [[balance, estimated_salary]]

    # Transformasi gabungan fitur menggunakan MinMaxScaler
    combined_scaled_values = minmax_scaler.transform(combined_features)

    # Ambil nilai hasil transformasi dari gabungan fitur
    combined_scaled_value = combined_scaled_values[0][0]

    # Mendefinisikan geo_feature_names
    geo_feature_names = geo_encoder.get_feature_names_out()

    # Create DataFrame for features
    features_df = pd.DataFrame({
        'Scaled Credit Score': [scaled_age_credit],  # Menggunakan hasil scaling age dan credit_score
        'Encoded Gender': [gender_encoded],
        'Scaled Age': [scaled_age_credit],  # Menggunakan hasil scaling age
        'Tenure': [tenure],
        'Scaled Balance': [combined_scaled_value],  # Menggunakan hasil scaling gabungan balance dan estimated_salary
        'Number of Products': [num_of_products],
        'Has Credit Card': [has_cr_card_encoded],
        'Is Active Member': [is_active_member_encoded],
        'Scaled Estimated Salary': [combined_scaled_value],  # Menggunakan hasil scaling gabungan balance dan estimated_salary
        # Anda mungkin perlu menyesuaikan fitur-fitur kategorikal yang lain jika ada
        # Misalnya, fitur-fitur dari hasil encoding geo_encoded
        **{f'Geography_{col}': [val] for col, val in zip(geo_feature_names, geo_encoded)}
    })

    return features_df

if __name__ == '__main__':
    main()
