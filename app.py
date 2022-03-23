import sklearntrain
import app2
import app1
#import app3 # create app3 file
import streamlit as st
PAGES = {
    "Sklearn Training": sklearntrain,
    "Sklearn Testing": app2,
    #"Pyacret Training": pycaret_training,
    "Pycaret Testing": app1
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()