# Travel Insurance Willingness Predictor Web App 

## Team 55
Chuen Kei Ho 101410183 
Ka Tsun Chan 101420274 



## Brief Explanation 
The objective of this Project is to forecast the willingness of a travel insurance company's customer to purchase its insurance product.Dataset was downloaded from [kaggle](https://www.kaggle.com/tejashvi14/travel-insurance-prediction-data). 

The Project itself has theese following features : 

1. Exploratory Data Analysis Overview 
2. Prediction 


## Technology Stacks : 
1. Streamlit Community Cloud for Deployment 
2. streamlit for web page creation 
3. Data processing packages / libraries such as Pandas, Numpy, seaborn , and etc. 
4. Machine Learning Model : sklearn, xgboost
   

## Project Workflow 
1. Data Exploration --> Model Building --> Creating interface --> Deploy on Streamlit

## Project Structure 
In order to create this project i create several files including jupyter notebook and python scripts. 
1. [modelling.ipynb]() for exploring the dataset from data cleaning to modelling 
2. [webapp.py]() python script for creating streamlit based web app 
3. [poetry.lock]() for dependency management 
4. [Dockerfile]() for containerizing app 


## Metrics 

Since the dataset's target class is not well distributed we attempt to use ROC AUC Scoring. 
The Model is using 
````
xgb_param  = {
            'objective':'binary:logistic',
            'max_depth': 6,
            'alpha': 6,
            'learning_rate': 0.01,
            'n_estimators':400
        }  

voting_classifier_params = 
{'estimators': [('lgb', LGBMClassifier()),
  ('rf', RandomForestClassifier()),
  ('gbc', GradientBoostingClassifier()),
  ('cat', <catboost.core.CatBoostClassifier at 0x2ae0594bdf0>)],
 'flatten_transform': True,
 'n_jobs': None,
 'verbose': False,
 'voting': 'soft',
 'weights': None,
 'lgb': LGBMClassifier(),
 'rf': RandomForestClassifier(),
 'gbc': GradientBoostingClassifier(),
 'cat': <catboost.core.CatBoostClassifier at 0x2ae0594bdf0>,
 'lgb__boosting_type': 'gbdt',
 'lgb__class_weight': None,
 'lgb__colsample_bytree': 1.0,
 'lgb__importance_type': 'split',
 'lgb__learning_rate': 0.1,
 'lgb__max_depth': -1,
 'lgb__min_child_samples': 20,
 'lgb__min_child_weight': 0.001,
 'lgb__min_split_gain': 0.0,
 'lgb__n_estimators': 100,
 'lgb__n_jobs': -1,
 'lgb__num_leaves': 31,
 'lgb__objective': None,
 'lgb__random_state': None,
 'lgb__reg_alpha': 0.0,
 'lgb__reg_lambda': 0.0,
 'lgb__silent': True,
 'lgb__subsample': 1.0,
 'lgb__subsample_for_bin': 200000,
 'lgb__subsample_freq': 0,
 'rf__bootstrap': True,
 'rf__ccp_alpha': 0.0,
 'rf__class_weight': None,
 'rf__criterion': 'gini',
 'rf__max_depth': None,
 'rf__max_features': 'auto',
 'rf__max_leaf_nodes': None,
 'rf__max_samples': None,
 'rf__min_impurity_decrease': 0.0,
 'rf__min_impurity_split': None,
 'rf__min_samples_leaf': 1,
 'rf__min_samples_split': 2,
 'rf__min_weight_fraction_leaf': 0.0,
 'rf__n_estimators': 100,
 'rf__n_jobs': None,
 'rf__oob_score': False,
 'rf__random_state': None,
 'rf__verbose': 0,
 'rf__warm_start': False,
 'gbc__ccp_alpha': 0.0,
 'gbc__criterion': 'friedman_mse',
 'gbc__init': None,
 'gbc__learning_rate': 0.1,
 'gbc__loss': 'deviance',
 'gbc__max_depth': 3,
 'gbc__max_features': None,
 'gbc__max_leaf_nodes': None,
 'gbc__min_impurity_decrease': 0.0,
 'gbc__min_impurity_split': None,
 'gbc__min_samples_leaf': 1,
 'gbc__min_samples_split': 2,
 'gbc__min_weight_fraction_leaf': 0.0,
 'gbc__n_estimators': 100,
 'gbc__n_iter_no_change': None,
 'gbc__presort': 'deprecated',
 'gbc__random_state': None,
 'gbc__subsample': 1.0,
 'gbc__tol': 0.0001,
 'gbc__validation_fraction': 0.1,
 'gbc__verbose': 0,
 'gbc__warm_start': False}  
````
if you want to customize on your own feel free to download model in my [this web app](https://share.streamlit.io/fakhrirobi/travel_insurance_webapp/main/webapp.py)
|Model Name |Accuracy Score  |ROC AUC Score|
|---------|---------|---------|
|XGBClassifier     |   0.849246      |0.849135|
|VotingClassifier     |   0.853162      |0.849246|
