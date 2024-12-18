import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from pydantic import BaseModel
from typing import List, Optional

class CropPrediction(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class Prediction(BaseModel):
    prediction: List[float]
    class_name: str