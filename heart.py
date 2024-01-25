import numpy as np
import pandas as pd
import streamlit as st
import joblib as jb
from sklearn.preprocessing import StandardScaler

st.title('HEART FAILURE  DECTECTOR')
st.sidebar.header('Enter the input data')
load = jb.load('model.py')
scalars=jb.load('scalar.py')


def input_values():

    age = st.sidebar.slider('age', 0, 100, 1)
    anaemia = st.sidebar.selectbox('anaemia', (0, 1))
    creatinine_phosphokinase = st.sidebar.slider('creatinine_phosphokinase',47, 7702, 1)
    diabetes = st.sidebar.selectbox( 'diabetes',(0,1))
    ejection_fraction = st.sidebar.slider('ejection_fraction',14,60,1)
    high_blood_pressure = st.sidebar.selectbox( 'high_blood_pressure',(0,1))
    serum_creatinine = st.sidebar.slider('serum_creatinine',0,9,1)
    gender = st.sidebar.selectbox('gender',(0,1))
    smoking= st.sidebar.selectbox('smoking',(0,1))
    time = st.sidebar.slider( 'time',0,256,8)


    data={'age':age,'anaemia':anaemia,'creatinine_phosphokinase':creatinine_phosphokinase,'diabetes':diabetes,
          'ejection_fraction':ejection_fraction,'high_blood_pressure':high_blood_pressure,'serum_creatinine':serum_creatinine,
          'gender':gender,'smoking':smoking,'time':time}
    features=pd.DataFrame(data,index=[0])
    return features
input_f=input_values()
st.write(input_f)
set_as_arrays=np.asarray(input_f)
reshape_data=set_as_arrays.reshape(1,-1)
print(reshape_data)
scale_data=scalars.transform(reshape_data)
predictions=load.predict(scale_data)
st.write(predictions)
prob_prediction=load.predict_proba(scale_data) 
st.write(prob_prediction)