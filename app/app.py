from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np
import os

# Create the FastAPI app
app = FastAPI()

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the model
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'agriculture_model.pkl')

# Configuring CORS
allowed_origins = [
    "http://localhost:3000",
    "http://localhost",
]
allowed_methods = ["post", "get"]
allowed_headers = ["*"]
app.add_middleware(allow_origins=allowed_origins,
                   allow_credentials=True,
                   allow_methods=allowed_methods,
                   allow_headers=allowed_headers
                )
# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "app/crop-app/public")), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=allowed_methods,
    allow_headers=allowed_headers,
)


async def get_model():
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    return model

@app.get("/", response_class=HTMLResponse)
async def read_root():
    model = await get_model()
    return "Hello World"