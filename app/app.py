import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from app.models.database import CropPrediction
from app.models.database import Prediction
import pickle
import numpy as np
from typing import Dict
from logging import getLogger
import logging

# Create the FastAPI app
app = FastAPI()

getLogger().setLevel('INFO')

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the model
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'agriculture_model.pkl')

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app/static")), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "app/templates"))

classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', \
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', \
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', \
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

async def predict_pipeline(factors: Dict):
    model = await get_model()
    print("Model loaded! Predicting...")
    try:
        prediction = model.predict([[factors['N'], factors['P'], factors['K'], factors['temperature'], factors['humidity'], factors['ph'], factors['rainfall']]])
        logging.info(f"Prediction: {prediction}")
        logging.info(f"Predicted class: {classes[prediction[0]]}")

    except:
        logging.error("Error predicting the crop")
        return "Error predicting the crop"
    return prediction, classes[prediction[0]]

async def get_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("home/index.html", {"request": request})

@app.post("/predict", response_model=Prediction)
async def predict_crop(request: Request, crop: CropPrediction):
    prediction, class_name = await predict_pipeline(crop.dict())
    
    return {"prediction": prediction[0].tolist(), "class_name": class_name}