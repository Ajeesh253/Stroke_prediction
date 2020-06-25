#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 00:55:31 2020

@author: ajeeshsunny
"""


import numpy as np
import pickle
import pandas as pd
import streamlit as st
from PIL import Image



pickle_in = open("DTC_model.pkl","rb")
classifier=pickle.load(pickle_in)

def welcome():
    return "Welcome All"

def predict_stroke(gender, ever_married, work_type, Residence_type, smoking_status, age, hypertension, heart_disease, avg_glucose_level, bmi):
    prediction=classifier.predict([[gender, ever_married, work_type, Residence_type, smoking_status, age, hypertension, heart_disease, avg_glucose_level, bmi]])
    print(prediction)
    return prediction

st.sidebar.header('User Input Parameters')

def main():
    st.title("Brian Stroke Prediction")
    html_temp = """
    <div style="background-color:tomato;padding:8px">
    <h2 style="color:white;text-align:center;">Brain Stroke Prediction ML App </h2>
    </div>
    """

    genderF ={ 0: "Male", 1: "Female"}
    YN = { 0:"No", 1:"Yes"}
    WT = {0:"Government job", 1:"Never worked", 2:"Private", 3:"Self employed", 4:"Children"}
    RT = {0:"Rural", 1:"Urban"}
    SS = {0:"Formerly Smoked", 1:"Never smoked", 2:"Smokes"}


    st.markdown(html_temp,unsafe_allow_html=True)
    st.write("""
    This app predicts whether a **Patient will have stroke or not based on some given attributes.**
    """)
    gender = st.sidebar.selectbox("Gender", (0, 1), format_func = genderF.get)
    ever_married = st.sidebar.selectbox("Ever_married", (0, 1), format_func = YN.get)
    work_type = st.sidebar.selectbox("Work_type", (0, 1, 2, 3, 4), format_func = WT.get)
    Residence_type = st.sidebar.selectbox("Residence_type", (0, 1), format_func = RT.get)
    smoking_status = st.sidebar.selectbox("Smoking_status", (0, 1, 2), format_func = SS.get)
    #id = st.sidebar.slider("ID", 1, 72938, 50000)
    age = st.sidebar.slider("Age", 18, 90, 54)
    hypertension = st.sidebar.selectbox("Hypertension", (0, 1), format_func = YN.get)
    heart_disease = st.sidebar.selectbox("Heart_disease", (0, 1), format_func = YN.get)
    avg_glucose_level = st.sidebar.slider("Avg_glucose_level", 55.0, 281.60, 100.0)
    bmi = st.sidebar.slider("BMI", 10.0, 92.0, 30.0)

    if st.button("About"):
        st.text("This web app predicts the chance of developing stroke based on RandomForestClassifier")


    data = {'Gender': gender,
            'Ever_married': ever_married,
            'Work_type': work_type,
            'Residence_type': Residence_type,
            'Smoking_status': smoking_status,
            #'id': id,
            'Age': age,
            'Hypertension': hypertension,
            'Heart_disease': heart_disease,
            'Avg_glucose_level': avg_glucose_level,
            'BMI': bmi}

    df = pd.DataFrame(data, index=[0])
    st.subheader('User Input parameters')
    st.write(df)

    result=""
    if st.button("Predict"):
        result = predict_stroke(gender, ever_married, work_type, Residence_type, smoking_status, age, hypertension, heart_disease, avg_glucose_level, bmi)
        st.success('The output is {}'.format(result))
    st.write("""
    If predict value is **0** then patient has **No chance of developing Stroke**.
    If predict value is **1** then patient has **Chance of developing Stroke**.
    """)
    st.subheader('Prediction Probability')
    prediction_proba = classifier.predict_proba(df)
    st.write(prediction_proba)

if __name__=='__main__':
    main()
