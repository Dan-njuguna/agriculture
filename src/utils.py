#!/usr/bin/env python3

"""
AUTHOR: Dan Njuguna
DATE: 2025-05-25

DESCRIPTION:
    This module defines helper functions and utilities to be used globally
"""

import os
import logging
from pathlib import Path

import pandas as pd
import pickle

# TODO: Define constants
LOGS_DIR = os.path.join(Path(__file__).parent.parent, "logs", "utils.log")
if not os.path.exists(Path(LOGS_DIR).resolve()):
    os.makedirs(Path(LOGS_DIR).parent, exist_ok=True)


# TODO: Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler(
    LOGS_DIR,
    encoding="utf-8"
)
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


# TODO: Define function to save data to csv
def save_to_csv(
        data: pd.DataFrame,
        path: str
) -> bool:
    """Saves a DataFrame to a CSV file."""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    logger.info(f"Saving data to csv in path {path} ...")
    try:
        data.to_csv(path, index=False)

    except Exception as e:
        logger.error(f"Failed to save data to {path}: {e}")
        return False

    logging.info(f"Data saved to {path}")
    return True


# TODO: Save pickle files to path for reuse
def save_pickle(
        model,
        path: str
) -> bool:
    """Saves pickle file to a given path"""
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    
    logger.info(f"Saving model to pickle in path {path} ...")
    try:
        with open(path, "wb") as f:
            pickle.dump(model, f)
    
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        return False
    
    logger.info(f"Model saved to {path}")
    return True


# TODO: Load pickle file
def load_pickle(
        path: str
):
    """Loads a pickle file in a give path and returns the loaded object"""
    pass