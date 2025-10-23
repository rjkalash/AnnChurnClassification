import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import streamlit as st

model= load_model('churn_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler= pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    encoder= pickle.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    onehotencoder= pickle.load(f) 

st.title('Customer Churn Prediction')

# Input fields 'CreditScore', 'Geography','Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard','IsActiveMember', 'EstimatedSalary'
CreditScore= st.number_input('Credit Score', min_value=300, max_value=850, value=600)
Geography= st.selectbox('Geography', onehotencoder.categories_[0].tolist())
Gender= st.selectbox('Gender', encoder.classes_.tolist())
Age= st.select_slider('Age', min_value=18, max_value=100, value=30)
Tenure= st.select_slider('Tenure', min_value=0, max_value=10, value=5)
Balance= st.number_input('Balance', min_value=0.0, value=1000.0)
NumOfProducts= st.select_slider('Number of Products', min_value=1, max_value=4, value=1)
HasCrCard= st.selectbox('Has Credit Card', options=[0, 1]) 
IsActiveMember= st.selectbox('Is Active Member', options=[0, 1])
EstimatedSalary= st.number_input('Estimated Salary', min_value=0.0, value=50000.0) 


if st.button('Predict Churn'):
    input_data= pd.DataFrame({
        'CreditScore': [CreditScore],
        'Gender': [encoder.transform([Gender])[0]],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'EstimatedSalary': [EstimatedSalary]
    })

    # One-hot encode 'Geography'
    Geography_encoded= onehotencoder.transform([[Geography]])
    Geography_encoded_df= pd.DataFrame(Geography_encoded, columns=onehotencoder.get_feature_names_out(['Geography']))
    input_data= pd.concat([input_data.reset_index(drop=True), Geography_encoded_df], axis=1)


    input_data_scaled= scaler.transform(input_data)
    prediction= model.predict(input_data_scaled)
    churn_probability= prediction[0][0]
    print('Churn Probability:', churn_probability)
    if churn_probability > 0.5:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is unlikely to churn.')
    st.write('Churn Probability:', churn_probability)
