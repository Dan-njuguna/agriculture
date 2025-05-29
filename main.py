#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os
from pathlib import Path
import joblib # Not directly used, but kept as it was in original
import pandas as pd
sys.path.append(str(Path(__file__).parent))
from src.utils import setup_logging, load_pickle

logger = setup_logging("api.log")

# Constants
ENVIRONMENT = os.getenv("ENVIRONMENT")
MODEL_PATH = os.path.join("model", "LogisticRegression_best_model.pkl")
LABEL_ENC = os.path.join("model", "encoders", "label_encoder.pkl")


# Initialize FastAPI instance
app = FastAPI(
    title="AgriPredict",
    description="API for doing predictions for Agricultural \
    crops given climatic and chemical conditions",
    version="1.1.0"
)


class UserInput(BaseModel):
    N: int | float = Field(
        description="Nitrogen concentation in the soil.(mg/Kg)"
    )
    P: int | float = Field(
        description="Phosphorus component in the soil. (mg/Kg)"
    )
    K: int | float = Field(description="Potassium component n the soil. (mg/kg)")
    temperature: int | float = Field(
        description="Temperature in degree celcius."
    )
    humidity: int | float = Field(
        description="The humidity of a place as a percentage(%)"
    )
    ph: int | float = Field(description="The level of acidity in a soil.")
    rainfall: int | float = Field(description="Rainfall in mm")

# Load Label encoder that has already been fit
encoder = load_pickle(LABEL_ENC)

# Get the classes from the encoder. These are the human-readable labels.
prediction_classes = encoder.classes_

@app.post("/predict")
async def predict(request: UserInput):
    """This function is used to predict the best crop
    to be planted in a given region
    ------
    Args:
        request (BaseModel) - This is the request body to be input by user.
    Return:
        JSON object containing the prediction made
    """
    model = load_pickle(MODEL_PATH)
    try:
        # Prepare the input data as a pandas DataFrame
        data = pd.DataFrame({
            "N": [request.N],
            "P": [request.P],
            "K": [request.K],
            "temperature": [request.temperature],
            "humidity": [request.humidity],
            "ph": [request.ph],
            "rainfall": [request.rainfall]
        })

        # Make the prediction.
        # The model.predict(data) now returns an array like [13] (integer index)
        prediction = model.predict(data)
        logger.info(f"Model predictions ready! Predictions: {prediction}")

        # Extract the predicted index directly from the model's output
        prediction_index = int(prediction[0]) # Ensure it's an integer
        logger.info(f"Model prediction index: {prediction_index}")

        result = "" # Initialize result
        predicted_probability_value = None # Initialize probability as None

        # Check if the prediction index produced by the model is in the classes.
        if 0 <= prediction_index < len(prediction_classes):
            result = prediction_classes[prediction_index] # Get the human-readable label
            logger.info("The prediction class has been found: %s", result)
            status = "success"

            # Attempt to get probabilities only if the model supports it
            if hasattr(model, 'predict_proba'):
                probability = model.predict_proba(data)
                predicted_probability_value = probability[0][prediction_index]
                logger.info(f"Prediction probability: {predicted_probability_value}")
            else:
                logger.warning("Model does not have 'predict_proba' attribute. Probability will be None.")
        else:
            logger.error(
                f"Prediction index not available in the classes! {prediction_index}. "
                f"Available classes length: {len(prediction_classes)}"
            )
            status = "failure"
            result = "" # Ensure result is empty if prediction failed
            prediction_index = -1 # Keep index as sentinel for failure

        logger.info(f"Prediction status is {status}")

        return {
            "status": status,
            "index": [prediction_index], # Return the numerical index
            "class": [result], # Return the string class name
            "probability": [predicted_probability_value]
        }

    except Exception as e:
        logger.error(f"Failed to make prediction for the model! Error {e}")
        return {
            "status": "failure",
            "index": None,
            "class": "",
            "probability": None
        }
