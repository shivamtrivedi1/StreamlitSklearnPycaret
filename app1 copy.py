import streamlit as st
from pycaret.classification import *
import pandas as pd
  

def app():
    st.title('PYCARET')
    st.write('Welcome to pycaret testing')
   

    model_gr =load_model('deploy_gboost')
    model_rf=load_model('deploy_rand_for')
    model_lr=load_model('deploy_log_reg')


    def predict(model, input_df):
        predictions_df = predict_model(estimator=model, data=input_df)
        predictions = predictions_df['Label'][0]
        return predictions


    def run():

        from PIL import Image
        st.title('Streamlit Example')
        st.write("""
        # Explore different classifier 
        """)
        st.write("Titanic Dataset")
        classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('Gradient Boost', 'Random Forest', 'Logistic Regression')
        )


        st.title("Titanic Prediction App")
        Age = st.number_input('Age', min_value=1, max_value=100, value=25)
        Sex = st.selectbox('Sex', ['male', 'female'])
        Pclass= st.number_input('P Class', 1,3)
        SibSp=  st.multiselect('Number of Siblings And Spouse',[0,1,2,3,4,5,8])
        Parch= st.multiselect('Parch',[0,1,2,3,4,5,6])
        Fare=  st.slider('Fare', 0,600)
        Embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])
        
        output=""

        input_dict = {'Age' : Age, 'Sex' : Sex, 'Pclass':Pclass,'SibSp':SibSp,'Parch':Parch,'Fare':Fare,'Embarked':Embarked}
        input_df = pd.DataFrame([input_dict])
        st.dataframe(input_df) 
        if st.button("Predict"):
            if classifier_name=='Gradient Boost':
                output = predict(model=model_gr, input_df=input_df)
                output = '$' + str(output)
                st.success('The output is {}'.format(output))
            elif classifier_name=='Random Forest':
                output = predict(model=model_rf, input_df=input_df)
                output = '$' + str(output)
                st.success('The output is {}'.format(output))
            else:
                output = predict(model=model_lg, input_df=input_df)
                output = '$' + str(output)
                st.success('The output is {}'.format(output))
                
    if __name__ == '__main__':
        run()

