
import streamlit as at
import numpy as np
import pandas as pd
import joblib


# first lets load the instances that were created
import joblib 
with open('scaler.joblib','rb') as file:
    scale = joblib.load(file)

with open('pca.joblib','rb') as file:
    pca = joblib.load(file)

with open('final_model.joblib','rb') as file:
    model = joblib.load(file)


def prediction(input_list):

    scaled_input = scale.transform([input_list])
    pca_input = pca.transform(scaled_input)
    output = model.predict(pca_input)[0]

    if output==0:
        return 'developing'
    if output == 1:
        return 'developed'
    if output == 2:
        return 'under_developed'

def main():

    st.title('HELP NGO FOUNDATION')
    st.subheader('This application will give the status of the country based on socio-economic and health factors')

    gdp = st.text_input('Enter the GDP per Population of a Country')
    inc = st.text_input('Enter the Per capita Income of a Country')
    imp = st.text_input('Enter the imports in the terms of % of GDP')
    exp = st.text_input('Enter the exports in the terms of % of GDP')
    inf = st.text_input('Enter the inflation rate in a country (%)')
    
    hel = st.text_input('Enter the expenditure on helath in terms of % of GDP')
    ch_m = st.text_input('Enter the number of deaths per thousand(1000) births for <5 yrs')
    fer = st.text_input('Enter the avg children born to a women in a country')
    lf = st.text_input('Enter the avg life expectancy in a country')

    in_data = [ch_m,exp,hel,imp,inc,inf,lf,fer,gdb]

    if st.button('Predict'):
        response = prediction(in_data)
        st.success(response)

if __name__=='__main__':
    main()
