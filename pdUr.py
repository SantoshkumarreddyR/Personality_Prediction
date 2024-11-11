import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = pd.read_csv('train.csv')

# Preprocess the data
data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})

# Features and target variable
X = data[['Gender', 'Age', 'openness', 'neuroticism', 'conscientiousness', 'agreeableness', 'extraversion']]
y = data['Personality (Class label)']

# Train a simple model (Random Forest Classifier)
model = RandomForestClassifier()
model.fit(X, y)

# Streamlit app
st.title('Personality Prediction')

st.sidebar.header('User Input Parameters')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 17, 30, 19)
    openness = st.sidebar.slider('Openness', 1, 10, 5)
    neuroticism = st.sidebar.slider('Neuroticism', 1, 10, 5)
    conscientiousness = st.sidebar.slider('Conscientiousness', 1, 10, 5)
    agreeableness = st.sidebar.slider('Agreeableness', 1, 10, 5)
    extraversion = st.sidebar.slider('Extraversion', 1, 10, 5)
    
    gender = 1 if gender == 'Male' else 0
    
    data = {
        'Gender': gender,
        'Age': age,
        'openness': openness,
        'neuroticism': neuroticism,
        'conscientiousness': conscientiousness,
        'agreeableness': agreeableness,
        'extraversion': extraversion
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Prediction
prediction = model.predict(input_df)

st.subheader('Prediction')
st.write(prediction[0])
