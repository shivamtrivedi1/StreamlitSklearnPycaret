import pandas as pd
from pycaret.classification import *
import streamlit as st

def app():
    st.title('PYCARET')
    st.write('Welcome to pycaret training')

    #train = pd.read_csv('train.csv')
    #test = pd.read_csv('test.csv')
    file_upload = st.file_uploader("Upload csv file for X_train", type=["csv"])

    if file_upload is not None:
        train = pd.read_csv(file_upload)

    train["Survived"]=train["Survived"].apply(lambda x:"Survived" if x==1 else "Dead")
    clf1 = setup(data = X_train, 
                target = 'Survived',
                numeric_imputation = 'mean',
                categorical_features = ['Sex','Embarked'], 
                ignore_features = ['Name','Ticket','Cabin'],
                silent = True,
                log_experiment = True, 
                experiment_name = 'titanic'
                )
    g_boost  = create_model('gbc') 
    tuned_gb = tune_model(g_boost)
    rand_for=create_model('rf') 
    log_reg=create_model('lr')
    best = compare_models(n_select = 15)
    compare_model_results = pull() 
    evaluate_model(tuned_gb)
    predictions=predict_model(tuned_gb,train)
            
    test_predictions=predict_model(tuned_gb,test)
    save_model(g_boost , 'deploy_gboost')
    save_model(rand_for,'deploy_rand_for')
            
    save_model(log_reg,'deploy_log_reg')