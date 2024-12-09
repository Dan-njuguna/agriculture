import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
from typing import Dict

# Create the FastAPI app
app = FastAPI()

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the model
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'agriculture_model.pkl')

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "/static")), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

classes = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee', \
       'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize', \
       'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya', \
       'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

async def predict_pipeline(factors: Dict[np.number]):
    model = await get_model()

    prediction = model.predict([[factors['N'], factors['P'], factors['K'], factors['temperature'], factors['humidity'], factors['ph'], factors['rainfall']]])
    return prediction, classes[prediction[0]]

async def get_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

@app.get("/", response_class=HTMLResponse)
async def read_root():

    return "Hello World"