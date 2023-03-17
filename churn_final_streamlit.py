#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 13:50:18 2023

@author: hongjiang
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


#load the model and data
df = pd.read_csv("train_final.csv")

model = pickle.load(open("model.pkl","rb"))

st.title("Customer Churn Perdiction")
st.subheader('E-Commerce')

#Tenure
Tenure = st.slider('Tenure (Month)', 0,70, 2)

#CashbackAmount
TotalCashback = st.number_input(label='Total Cash back Amount per month', value=300.00)

#Complain
Complain = st.selectbox('Complain', ['Yes','No'])

#WarehouseToHome
WarehouseToHome = st.slider('Ware house To Home (miles)', 0, 100, 30)

#DaySinceLastOrder
DaySinceLastOrder =  st.slider('Day Since Last Order in Last Month', 0, 31, 10)

#NumberOfAddress
NumberOfAddress = st.slider('Number Of Address', 0, 20, 1)

#SatisfactionScore
SatisfactionScore = st.selectbox('SatisfactionScore', ['Very Satisfied','Satisfied', 'Neutral', 'Disappointed', 'Very Disappointed'])

#PreferedOrderCat_Mobile
PreferedOrderCat_Mobile = st.selectbox('Prefered to order mobile', ['Yes','No'])

#CouponUsed
CouponUsed = st.slider('Coupon Used', 0, 30, 1)

#CityTier
CityTier = st.selectbox('CityTier', ['1','2', '3'])


if st.button('Predict Churn'):
    #complain
    if Complain == "Yes":
        Complain = 1
    else:
        Complain = 0
    
    #SatisfactionScore   
    if SatisfactionScore == 'Very Satisfied':
        score = 1

    elif SatisfactionScore == "Satisfied":
        score = 2

    elif SatisfactionScore == "Neutral":
        score = 3
        
    elif SatisfactionScore == 'Disappointed':
        score = 4    
    else:
        score = 5
    
    #PreferedOrderCat_Mobile
    if PreferedOrderCat_Mobile == 'Yes':
        PreferedOrderCat_Mobile = 1
    else:
        PreferedOrderCat_Mobile = 0
    
    #CityTier
    if CityTier == '1':
        CityTier = 1
    elif CityTier == '2':
        CityTier = 2
    else:
        CityTier = 3
    
        
    query = np.array([Tenure, TotalCashback, Complain, WarehouseToHome, \
                      DaySinceLastOrder, NumberOfAddress, score, \
                      PreferedOrderCat_Mobile, CouponUsed, CityTier], dtype=object)
    

    query = query.reshape(1, 10)
    print(query)
    prediction = str(round(model.predict_proba(query)[0][-1] * 100, 3))
    
    st.subheader("Churn probability is " + prediction + "% for this customer.")

    shap.initjs()

    #set the tree explainer as the model of the pipeline
    explainer = shap.TreeExplainer(model['xgb'], df, model_output="probability")

    #get Shap values from preprocessed data
    shap_value = explainer.shap_values(query)

    #plot the feature importance
    fig = shap.force_plot(explainer.expected_value, shap_value, query, matplotlib=True,show=False, \
                          feature_names=['Tenure', 'TotalCashback', 'Complain', 'WarehouseToHome', \
                                            'DaySinceLastOrder', 'NumberOfAddress', 'SatisfactionScore', \
                                                'PreferedOrderCat_Mobile', 'CouponUsed', 'CityTier'])
    st.pyplot(fig)




