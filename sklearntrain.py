import streamlit as st
import pandas as pd

def app():
    import numpy as np
    from importlib import import_module
    import joblib

    st.title('Streamlit Example')

    st.write("Titanic Dataset")

    classifier_name = st.text_input("Classifier_name")

    file_upload = st.file_uploader("Upload csv file for X_train", type=["csv"])

    if file_upload is not None:
        X_train = pd.read_csv(file_upload)

    file_upload = st.file_uploader("Upload csv file for Y_train", type=["csv"])

    if file_upload is not None:
        Y_train= pd.read_csv(file_upload)    

    if st.button("Train"):

        module_name, model_name = classifier_name.split(".")
        model = getattr(import_module(f"sklearn.{module_name}"), model_name)()
        model.fit(X_train, Y_train)
        filename = model_name + "trained.sav"
        joblib.dump(model,filename)

        """directory = os.path.abspath(os.getcwd())
        with open(os.path.join(directory,model_name),"wb") as f: 
          f.write(model_name.getbuffer())"""
        st.text("Model saved")


    

    

    
  





