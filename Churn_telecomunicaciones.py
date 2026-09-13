#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
# In[4]:

st.markdown(
    """
    <style>
    /* Fondo */
    .stApp {
        background-color: #f0f4f7;
        background-image: linear-gradient(120deg, #f0f4f7 0%, #d9e4ec 100%);
    }

    /* Botones */
    div.stButton > button:first-child {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        margin-top: 20px; /* separación extra arriba */
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)
                                          
predictor = joblib.load("modelo_smote_1.pkl")

with st.container():
    c1, c2, c3 = st.columns([1,6,1]) 
    
    
    with c2:
        st.header("Predicción con el mejor modelo")
        st.write("Esta aplicación predice la probabilidad de que un cliente deje el servicio (Churn) usando un modelo entrenado con datos históricos.")


st.markdown("<hr style='margin:40px 0;'>", unsafe_allow_html=True)


with st.container():
    st.write("Introduce los valores manualmente:")

st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

with st.container():
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        status = st.selectbox("Status", [0, 1])
        subscription_length = st.number_input("Subs Length", min_value=1)
        
    
    with col2:
        age_group = st.selectbox("Age Group", [1, 2, 3, 4, 5])
        customer_value = st.number_input("Customer Value", min_value=0.0)
    
    with col3:
        frequency_of_use = st.number_input("Frequency of use", min_value=0)
        frequency_of_sms = st.number_input("Frequency of SMS", min_value=0)
    
    
    with col4:
        complains = st.selectbox("Complains", [0, 1])
        charge_amount = st.number_input("Charge Amount", min_value=0.0)
    
    
    with col5:
        seconds_of_use = st.number_input("Seconds of Use", min_value=0)
        distinct_called_numbers = st.number_input("Distinct Called", min_value=0)
        
        
    with col6:
        call_failure = st.number_input("Call Failure", min_value=0)

st.markdown("<div style='margin-top: 50px;'></div>", unsafe_allow_html=True)

co1, co2 = st.columns(2)


with co1:
    if st.button("Predecir con mis datos"):
        X_new = pd.DataFrame([[
            status, complains, seconds_of_use, age_group, subscription_length,
            customer_value, distinct_called_numbers, call_failure,
            frequency_of_use, frequency_of_sms, charge_amount
        ]], columns=[
            'Status','Complains','Seconds of Use','Age Group','Subscription  Length',
            'Customer Value','Distinct Called Numbers','Call  Failure',
            'Frequency of use','Frequency of SMS','Charge  Amount'
        ])
    
        pred = predictor.predict(X_new)[0]
        probas = predictor.predict_proba(X_new)[0]
        st.markdown("---")  
        st.write("📊 Probabilidad de que el cliente no deje el servicio:", round(probas[0]*100,2), "%")
        st.write("📊 Probabilidad de que el cliente deje el servicio:", round(probas[1]*100,2), "%")


with co2:
    if st.button("Generar caso aleatorio"):
        X_random = pd.DataFrame([[
            np.random.choice([0,1]), np.random.choice([0,1]), np.random.randint(0,17090),
            np.random.randint(1,5), np.random.randint(1,47),
            np.random.uniform(0,2165.28), np.random.randint(0,97), np.random.randint(0,36),
            np.random.randint(0,255), np.random.randint(0,522), np.random.uniform(0,10)
        ]], columns=[
            'Status','Complains','Seconds of Use','Age Group','Subscription  Length',
            'Customer Value','Distinct Called Numbers','Call  Failure',
            'Frequency of use','Frequency of SMS','Charge  Amount'
        ])
    
        pred = predictor.predict(X_random)[0]
        probas = predictor.predict_proba(X_random)[0]
        st.markdown("---") 
    
        
        st.write("📊 Probabilidad de que el cliente no deje el servicio:", round(probas[0]*100,2), "%")
        st.write("📊 Probabilidad de que el cliente deje el servicio:", round(probas[1]*100,2), "%")
        st.write("🎲 Valores generados:", X_random)
