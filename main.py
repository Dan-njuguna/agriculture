#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
import os
import pickle
import pandas as pd

# Basic Logging configuration
logging.basicConfig(level="INFO")
file_handler = logging.FileHandler("runs/api.log")
formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

# Constants
ENVIRONMENT = os.getenv("ENVIRONMENT")
MODEL_PATH: str = os.path.join("model", "agriculture_model.pkl")

# Initialize FastAPI instance
app = FastAPI(
    title="AgriPredict",
    description="API for doing predictions for Agricultural crops given climatic and chemical conditions",
    version="0.1.0"
)

class UserInput(BaseModel):
    N: int | float = Field(description="Nitrogen concentation in the soil.(mg/Kg)")
    P: int | float = Field(description="Phosphorus component in the soil. (mg/Kg)")
    K: int | float = Field(description="Potassium component n the soil. (mg/kg)")
    temperature: int | float = Field(description="Temperature in degree celcius.")
    humidity: int | float = Field(description="The humidity of a place as a percentage(%)")
    ph: int | float = Field(description="The level of acidity in a soil.")
    rainfall: int | float = Field(description="Rainfall in mm")


# NOTE: The classes to be predicted by the model
prediction_classes = [
    'apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
    'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
    'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
    'pigeonpeas', 'pomegranate', 'rice', 'watermelon'
]


async def load_model(path: str = MODEL_PATH):
    """This function load the model from the pickle file
    -----
    Args:
        path (str) - the path to the model being loaded
    --------
    Returns:
        The model instance loaded
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    
    logger.info("Model data type: %s", type(model))
    
    if model:
        logger.info("Model Loaded successfully!")
    else:
        logger.error("Model loading failed!")

    return model


@app.post(path="/")
async def predict(request: UserInput):
    """This function is used to predict the best crop to be planted in a given region
    ------
    Args:
        request (BaseModel) - This is the request body to be input by user.
    Return:
        JSON object containing the prediction made
    """
    model = await load_model()
    try:
        data = pd.DataFrame({
            "N": [request.N],
            "P": [request.P],
            "K": [request.K],
            "temperature": [request.temperature],
            "humidity": [request.humidity],
            "ph": [request.ph],
            "rainfall": [request.rainfall]
        })

        prediction = model.predict(data)
        logger.info(f"Model predictions ready! Predictions: {prediction}")

        result = ""
        prediction_index = int(prediction[0])
        # Check if the prediction index produced by the model is in the classes.
        if 0 <= prediction_index < len(prediction_classes):
            result = prediction_classes[prediction_index]
            logger.info("The prediction class has been found: %s", result)
            status = "success"
        else:
            logger.error(f"Prediction index not available in the classes! {prediction_index}")
            status = "failure"

        logger.info(f"Prediction status is {status}")

        return {
            "status": status,
            "index": [prediction_index],
            "class": [result]
        }

    except Exception as e:
        logger.error(f"Failed to make prediction for the model! Error {e}")
        return {
            "status": "failure",
            "index": None,
            "class": ""
        }
