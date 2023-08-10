# Medical Insurance Premium Prediction Web App

## Team 55
Chuen Kei Ho 101410183 
Ka Tsun Chan 101420274 
Ka Ho Cheung 101422288 


## Brief Explanation 
This project's goal is to create a precise machine learning /neural network model to predict medical insurance premium by considering factors like age, BMI, no.of children, smokers….etc..Dataset was downloaded from [kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance). 

[Link to Github Repo](https://github.com/ktchan33GBC/WIP_team55_demo)

The Project itself has theese following features : 

1. Overview 
2. Exploratory Data Analysis
3. Sensitivity Analysis
4. Prediction with ANN Model


## Technology Stacks : 
1. Streamlit Community Cloud for Deployment 
2. Streamlit for web page creation 
3. Data processing libraries : Pandas, Numpy, seaborn , and etc. 
4. Machine /Deep Learning Model : sklearn, keras
   

## Project Workflow 
1. Data Exploration --> Model Building --> Creating interface --> Deploy on Streamlit

## Project Structure 
In order to create this project i create several files including jupyter notebook and python scripts. 
1. [insurance_regression.ipynb](https://github.com/ktchan33GBC/WIP_team55_demo/blob/main/insurance_regression.ipynb) for exploring the dataset from data cleaning to modelling 
2. [webapp3.py](https://github.com/ktchan33GBC/WIP_team55_demo/blob/main/webapp3.py) python script for creating streamlit based web app 

please share [this web app](https://wip-team55-demo-medical-insur-premium-predictor.streamlit.app/) to your friends.

## Metrics 


| Metric         | Linear Regression | Voting Model   | ANN         |
|----------------|-------------------|----------------|-------------|
| MSE            | 33,596,915.85     | 20,861,810.37  | 20,028,624.12 |
| RMSE           | 5,796.28          | 4,567.47       | 4,475.34    |
| MAE            | 4,181.19          | 2,528.41       | 2,960.15    |
| R-squared      | 0.78              | 0.83           | 0.87        |
| MAPE           | 46.89%            | 40.61%         | 37.29%      |


