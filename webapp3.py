from os import write
from st_on_hover_tabs import on_hover_tabs # using on hover tabs custom components streamlit 
import streamlit as st
import plotly.express as px  
import pandas as pd 
import numpy as np 
import seaborn as sns
import pickle
import matplotlib.pyplot as plt
import urllib
import xgboost 
import joblib
from tensorflow.keras.models import load_model
# import keras
#import tensorflow as tf
import pandas_profiling

import sklearn
from streamlit_pandas_profiling import st_profile_report

from st_aggrid import AgGrid




sns.set_style('whitegrid')
st.set_option('deprecation.showPyplotGlobalUse', False)




df = pd.read_csv('src/data/insurance.csv')

st.image('assets/Banner_2.png')


with st.sidebar:
    
    st.title("Please Select")
    choice = st.radio("Navigation",["Overview","Exploratory Data Analysis","Sensitivity Analysis","Try Prediction with ANN Model"])

if choice == "Overview":
    st.title("Overview")
    url = 'https://raw.githubusercontent.com/ktchan33GBC/WIP_team55_demo/main/README.md?token=GHSAT0AAAAAACCGCJAXC4WSFYVHT6OKPP5WZFQW5ZQ'
    def get_file_content_as_string(url):
        #for reading readme.md from github
        response = urllib.request.urlopen(url)
        return response.read().decode("utf-8")
    st.markdown(get_file_content_as_string(url),unsafe_allow_html=True)

if choice == "Exploratory Data Analysis":

    profile_report= df.profile_report()
    st_profile_report(profile_report)

if choice == "Sensitivity Analysis":
     
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(['age', 'sex', 'bmi', 'children', 'smoker',
        'northwest', 'southeast', 'southwest'], [  2.56975706e+02, -1.85916916e+01, 3.37092552e+02 ,4.25278784e+02,
  2.36511289e+04, -3.70677326e+02, -6.57864297e+02 ,-8.09799354e+02])
    ax.set_title("Sensitivity of Each Variable on Medical Insurance Premium")
    

    # plt.show()
    st.pyplot(fig)        

    st.write("""
             
             Sensitivity analysis is used in insurance company to illustrate the relevant importance of each variable in the change of insurance, it measures the movement of insurance premium with respect to the change in each variable. The above graph illustrate the sensitivity of each variable on insurance premium. 
             
             It is noted the insurance premium is most sensitivity to the smoker status. A smoker would be charged roughly 20000 annually in premium compared to a non-smoker if everything else are the same.
             
            The insurance premium is also positively correlated to BMI. 1 point increase in BMI results in approximately 337 dollar increase in the annual premium.The insurance premium is also positively correlated to age. 1 year increase in age results in approximately 257dollar  higher annual premium.
             
            From the negative coefficient in the 3 regions, we can interpret that northeast has a higher average insurance premium compared to the other 3 regions although it is not very significant.
             
            The sensitivity analysis provides us high level understanding on the relative importance of each variable, but is not the most accurate prediction approach in terms of the MAE/MSE/RMSE/R-square.
             
            Therefore, we will introduce the neural network approach in the next section
             
            """)
    

if choice == "Try Prediction with ANN Model":
        
    st.title(" To predict the medical insurance premium of your customer (Using ANN Model)")
    with st.form('my form') : 
        st.write('To predict how much your customer need to pay for his/her medical insurance premium.')
        age_form = st.number_input('Age',min_value=1)
        sex_form = st.selectbox('Sex',options=('female', 'male'))
        bmi_form = st.number_input('Please input his/her bmi',min_value=1.0)
        children_form = st.number_input('How many children he/she have ? ',min_value=1)
        smoker_form = st.selectbox('Do he/she smoke ? ',options=('yes','no'))
        region_form = st.selectbox('Which region does he/she live ? ',options=('northeast','northwest','southeast','southwest'))
        submit_btn = st.form_submit_button()

    if submit_btn : 
        st.write('Thank you for your input the models are about to tell you')
        response_data={'age':age_form, 'sex':sex_form, 
                    'bmi':bmi_form, 'children':children_form,
                    'smoker':smoker_form, 'region':region_form,
                }
        
        ######Data cleaning
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import GridSearchCV
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.preprocessing import StandardScaler
        from numpy import dtype

       
        response_data = pd.DataFrame([response_data])
        target_columns = ['sex', 'smoker']
        label_encoders = {}
        for column in target_columns:
            label_encoders[column] = LabelEncoder().fit(df[column].values)
            # Fit
            response_data[column] = label_encoders[column].transform(response_data[column].values)

        char_scaler = StandardScaler().fit(df[['charges']].values.reshape(-1, 1))
      
        bmi_scaler = StandardScaler().fit(df['bmi'].values.reshape(-1, 1))
        response_data['bmi_scaled'] = bmi_scaler.transform(response_data['bmi'].values.reshape(-1, 1))

        chi_scaler = StandardScaler().fit(df['children'].values.reshape(-1, 1))
        response_data['children_scaled'] = chi_scaler.transform(response_data['children'].values.reshape(-1, 1))

        response_data= pd.DataFrame(response_data)
        
        ohe_region = OneHotEncoder()
        region = ohe_region.fit(df[['region']])
        for region_lab, num in zip(ohe_region.categories_[0], range(0,4)):
            response_data[region_lab] = region.transform(response_data[['region']]).toarray()[:,num]
        
        model_input =response_data[['age', 'sex', 'bmi_scaled', 'children_scaled', 'smoker',
        'northwest', 'southeast', 'southwest']]

        #loading the trained_model 
        # path = os.path.join(current_dir,')
        filename = r'src/ann_trained.h5'
        # clf_model = keras.models.load_model(filename)
        # filename = r'src/linear_regression_trained.pkl'
        # clf_model = joblib.load(filename)
        clf_model = load_model(filename)


        pred = clf_model.predict(model_input)
        
        pred = char_scaler.inverse_transform(np.array(pred).reshape(-1, 1))
        st.write("The predicted medical insurance premium of your customer annually : $USD " ,str(np.round(pred[0],0)[0]))



