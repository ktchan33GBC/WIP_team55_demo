from os import write
from src.visualization import visualize_numerical_data,visualize_categorical_data
from src.button import download_button
from src.forms import FormFlow
from src.custom_model import start_training,generate_chart
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
#import pandas_profiling
#import ydata_profiling

import sklearn
#from streamlit_pandas_profiling import st_profile_report

from st_aggrid import AgGrid

# import keras
# from keras.models import Sequential
# from keras.layers import Dense


sns.set_style('whitegrid')
st.set_option('deprecation.showPyplotGlobalUse', False)




df = pd.read_csv('src/data/insurance.csv')
# file_id = '1e2z2mVkAhRvZ9JGcTuJFg99mvJwoQdyv'
# url = 'https://drive.google.com/uc?id={}'.format(file_id)
# df = pd.read_csv(url)

# tabs = st.sidebar.radio('Page Selector',('Project Explanation','Exploratory Data Analysis','Create your own model','Try Prediction'))
# with st.sidebar:
#     tabs = on_hover_tabs(tabName=['Project Explanation', 'Exploratory Data Analysis', 'Create your own model','Try Prediction'], 
#                          iconName=['Project Explanation', 'Exploratory Data Analysis', 'Create your own model','Try Prediction'], default_choice=0
# )
# tab1, tab2, tab3,tab4, = st.tabs(['Project Introduction', 'Exploratory Data Analysis', '唔做：Create your own model','Try Prediction'])
# with tab1 :
# with tab2 :
# with tab3 : 
# with tab4 : 
st.image('assets/Banner_2.jpeg')


with st.sidebar:
    
    st.title("Please Select")
    choice = st.radio("Navigation",["Project Introduction","Exploratory Data Analysis","唔做：Create your own model","Try Prediction"])
    #st.info(" This is the best Travel Insurance Predictor in the market")

if choice == "Project Introduction":
    st.title("Project Introduction")
    url = 'https://raw.githubusercontent.com/ktchan33GBC/WIP_team55_demo/main/README.md?token=GHSAT0AAAAAACCGCJAXC4WSFYVHT6OKPP5WZFQW5ZQ'
    def get_file_content_as_string(url):
        #for reading readme.md from github
        response = urllib.request.urlopen(url)
        return response.read().decode("utf-8")
    st.markdown(get_file_content_as_string(url),unsafe_allow_html=True)

# if choice == "Exploratory Data Analysis":

#     st.title('Exploratory Data Analysis')
#     ## ------ Original Coding Begins----------
#     # dataset_desc_exp = st.expander('Dataset Description',expanded=True)
#     # dataset_desc_exp.markdown('''
#     # |Column Name  |Description  |
#     # |---------|---------|
#     # |Age     |   Age Of The Customer      |
#     # |Employment     | The Sector In Which Customer Is Employed        |
#     # |GraduateOrNot     |  Whether The Customer Is College Graduate Or Not       |
#     # |AnnualIncome     |   The Yearly Income Of The Customer In Indian Rupees[Rounded To Nearest 50 Thousand Rupees      |
#     # |FamilyMembers     |     Number Of Members In Customer's Family    |
#     # |ChronicDisease     |Whether The Customer Suffers From Any Major Disease Or Conditions Like Diabetes/High BP or Asthama,etc         |
#     # |FrequentFlyer     |  Derived Data Based On Customer's History Of Booking Air Tickets On Atleast 4 Different Instances In The Last 2 Years[2017-2019]       |
#     # |EverTravelledAbroad     | Has The Customer Ever Travelled To A Foreign Country[Not Necessarily Using The Company's Services]        |
#     # |TravelInsurance     |   Did The Customer Buy Travel Insurance Package During Introductory Offering Held In The Year 2019      |
                    
#     #             ''',unsafe_allow_html=True)
#     # data_expander = st.expander('Data Inspection')
#     # col1,col2 = data_expander.columns(2)
#     # with col1 : 
#     #     st.subheader('Missing Values Inspection: ')
#     #     missing = pd.DataFrame({"columns":[x for x in data.columns],
#     #                             "% missing data" :[(x/data.shape[0])*100 for x in data.isnull().sum()] })
#     #     AgGrid(missing)
#     # with col2 : 
#     #     st.subheader('Dataset Statistics: ')
#     #     AgGrid(data.describe())
#     # col3,col4 = data_expander.columns(2)
#     # with col3 : 
#     #     st.subheader('Exploratory Data Analysis : Categorical Data')
#     #     visualize_categorical_data(data)
#     # with col4 : 
        
#     #     visualize_numerical_data(data)

#     ## ------ Original Coding Ends ----------
#     profile_report= data.profile_report()
#     st_profile_report(profile_report)

    ## tab 3 will be deleted
if choice == "唔做：Create your own model":
     
    st.subheader("Custom Params for creating XGBoost Model in Travel Insurance dataset")
    with st.form("model_customization") : 
        st.write('please input this following value to customize model')
        xgb_param  = {
            'objective':'binary:logistic',
            'max_depth': 6,
            'alpha': 10,
            'learning_rate': 0.0001,
            'n_estimators':300
        }
        max_depth = st.slider("please input num of max_depth",min_value=1,max_value=50,step=1)
        alpha = st.slider("please input num of alpha",min_value=1,max_value=30,step=1)
        learning_rate = st.slider("please input num of learning_rate",min_value=0.0001,max_value=1.0,step=0.0001)
        n_estimators = st.slider("please input num of n_estimators",min_value=10,max_value=1000,step=10)
        model_name_to_save = st.text_input("Please input the name of your model to")
        custom_params = {
            'objective':'binary:logistic',
            'max_depth': max_depth,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'n_estimators':n_estimators
        }
        start_train_button = st.form_submit_button("Start Training Model")
        
    custom_model = xgboost.XGBClassifier(**custom_params)
    if start_train_button : 
        result,trained_model = start_training(custom_model)
        st.write("Model Result : ")
        AgGrid(result)
        figure = generate_chart(result)
        st.plotly_chart(figure)
        filename = f'{model_name_to_save}.pkl'
        with open(filename,'wb') as file : 
            model_download = joblib.dump(result,file)
        download_btn = download_button(model_download,download_filename=filename,button_text=f'Click here to download {filename}', pickle_it=False)
        st.markdown(download_btn, unsafe_allow_html=True)
        




if choice == "Try Prediction":
        
    st.title(" To Predict if your customer would buy Travel Insurance  (Using Voting_classifier)")
    with st.form('my form') : 
        st.write('To predict whether your customer is willing to buy our insurance product please input your customer candidate data below')
        age_form = st.number_input('Age',min_value=1)
        sex_form = st.selectbox('Sex',options=('female', 'male'))
        bmi_form = st.number_input('Please input your bmi',min_value=1.0)
        children_form = st.number_input('How many children you have ? ',min_value=1)
        smoker_form = st.selectbox('Do you smoke ? ',options=('yes','no'))
        region_form = st.selectbox('Which region do you live ? ',options=('northeast','northwest','southeast','southwest'))
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
        response_data.to_csv('response_data.csv')
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
        st.write("predicted charges : " ,np.round(pred[0],0))

