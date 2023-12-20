import streamlit as st 
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Simple Iris Flower Prediction App") 
st.write("This app predicts the **Iris flower** type!")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) #(min,max, default value)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,  #susunan kena ikut column from your data
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0]) #convert data frame as index zero [0] as save as features
    return features

df = user_input_features() #call funtion, then will do all process from above

st.subheader('User Input parameters') #bole guna subheadeer == dengan #
st.write(df) #can do training in data application

data = sns.load_dataset('iris') 
X = data.drop(['species'],axis=1)
Y = data.species.copy()

modelGaussianIris = GaussianNB() #terus generate 
modelGaussianIris.fit(X, Y)

prediction = modelGaussianIris.predict(df) #terus buat prediction 
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Class labels and their corresponding index number') #subheader yg lain
st.write(Y.unique()) #untuk dapat unique value guna function unique

st.subheader('Prediction') #result probability from input towards all these
st.write(prediction)

st.subheader('Prediction Probability') #result probability from input towards all these
st.write(prediction_proba)
