import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder



model= load_model('salary_regression_model.h5')
with open('scalerr.pkl', 'rb') as f:
    scaler= pickle.load(f)
with open('le_gender.pkl', 'rb') as f:
    encoder= pickle.load(f)
with open('ohe_geography.pkl', 'rb') as f:
    onehotencoder= pickle.load(f)

st.title('Salary Prediction')

# Input fields ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts','HasCrCard', 'IsActiveMember',  'Exited', 'Geography_France', 'Geography_Germany', 'Geography']
CreditScore= st.number_input('Credit Score', min_value=300, max_value=850, value=600)
Gender= st.selectbox('Gender', encoder.classes_.tolist())
Age= st.number_input('Age', min_value=18, max_value=100, value=30)
Tenure= st.number_input('Tenure', min_value=0, max_value=10, value=5)
Balance= st.number_input('Balance', min_value=0.0, value=1000.0)
NumOfProducts= st.number_input('Number of Products', min_value=1, max_value=4, value=1)
HasCrCard= st.selectbox('Has Credit Card', options=[0, 1])
IsActiveMember= st.selectbox('Is Active Member', options=[0, 1])
Exited= st.selectbox('Exited', options=[0, 1])
Geography= st.selectbox('Geography', onehotencoder.categories_[0].tolist())

if st.button('Predict Salary'):
    input_data= pd.DataFrame({
        'CreditScore': [CreditScore],
        'Gender': [encoder.transform([Gender])[0]],
        'Age': [Age],
        'Tenure': [Tenure],
        'Balance': [Balance],
        'NumOfProducts': [NumOfProducts],
        'HasCrCard': [HasCrCard],
        'IsActiveMember': [IsActiveMember],
        'Exited': [Exited]
    })
    Geography=pd.DataFrame({'Geography': [Geography]})  
    Geography_encoded = onehotencoder.transform(Geography)
    Geography_encoded_df= pd.DataFrame(Geography_encoded, columns=['Geography_Germany', 'Geography_Spain '])
    input_data= pd.concat([input_data.reset_index(drop=True), Geography_encoded_df], axis=1)
    input_data_scaled= scaler.transform(input_data.values)
    prediction= model.predict(input_data_scaled)

    predicted_salary= prediction[0][0]
    st.write(f'The predicted salary is: {predicted_salary:.2f}')
    
