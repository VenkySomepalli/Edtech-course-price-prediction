#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 20:30:11 2022

@author: venky
"""

import uvicorn
from fastapi import FastAPI
import pickle

app = FastAPI()

@app.get('/')
def home():
    return{'text' :'Price prediction'}

@app.get('/predict')
def predict(
    Institute: int,
    Subject: int,
    Location: int,
    Trainer_Qualification: int,
    Online_classes: int,
    Offline_classes: int,
    Course_level: int,
    Placements : int,
    Trainer_experiance: int,
    Course_hours:int,
    Course_rating: int,
    Rental_permises: int,
    Trainer_slary:int,
    Maintaince_cost:int,
    Non_teaching_staff_salary:int):
    
    model = pickle.load(open("/home/venky/Desktop/Datascience_360/Real_Project_costprediction/VS-Edtech_CourseCost_Model-Deployment/data_model.pkl","rb"))
    
    makeprediction = model.predict([[Institute,Subject,Location,Trainer_Qualification,Online_classes,
                Offline_classes, Course_level,Placements,Trainer_experiance,
                Course_hours,Course_rating, Rental_permises,Trainer_slary, Maintaince_cost,
                Non_teaching_staff_salary]])
    output = round(makeprediction[0])
    return {'Edtech course cost prediction per hour: {}'.format(output) + ' INR'} 

if __name__ == '__main__':
    uvicorn.run(app)





    



















    
    
    
    
    
    
    
    
    