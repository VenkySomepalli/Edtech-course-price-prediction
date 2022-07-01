import streamlit as st
import pickle
import numpy as np
import pandas as pd

#load the model and dataframe
df = pd.read_csv("/home/venky/Desktop/Datascience_360/Real_Project_costprediction/Cost_Model/data.csv")
pipe = pickle.load(open("/home/venky/Desktop/Datascience_360/Real_Project_costprediction/Cost_Model/data.pkl", "rb"))

st.title("Cost prediction of edtech company course")

#Now we will take user input one by one as per our dataframe

#Institute_Name
#company = st.selectbox('Brand', df['Company'].unique())
Institute = st.selectbox('Institute_name', df['Institute'].unique())

#Subject_Name
Subject = st.selectbox("Subject_name", df['Subject'].unique())

#Location
Location = st.selectbox("Location", df['Location'].unique())

## Trainer
Trainer_Qualification = st.selectbox("Trainer_Qualification", df['Trainer_Qualification'].unique())
Trainer_experiance = st.sidebar.number_input('Trainer_experiance')
## Classes
Online_classes = st.selectbox("Online_classes", df['Online_classes'].unique())
Offline_classes = st.selectbox("Offline_classes", df['Offline_classes'].unique())

# Hours
Course_hours = st.sidebar.number_input('Course_hours')
Course_level = st.selectbox("Course_level", df['Course_level'].unique())
Course_rating = st.sidebar.number_input('Course_rating')
## Maintaince cost
Rental_permises = st.sidebar.number_input('Rental_permises')
Trainer_slary = st.sidebar.number_input('Trainer_slary')
Maintaince_cost = st.sidebar.number_input('Maintaince_cost')
Non_teaching_staff_salary = st.sidebar.number_input('Non_teaching_staff_salary')

# placements
Placements =  st.sidebar.number_input('Placements')

# Certificate
#Certificate = st.sidebar.number_input('Certificate')

#Prediction
if st.button('Price'):
    query = np.array([Institute,Subject,Location, Trainer_Qualification, Trainer_experiance, Online_classes, Offline_classes, Course_hours,
                      Course_level,Course_rating,Rental_permises, Trainer_slary,Maintaince_cost,Non_teaching_staff_salary, Placements])
    query = query.reshape(1, 15)
    prediction = str(int(np.exp(pipe.predict(query)[0])))
    st.title("The predicted price of the course is " + prediction)
    