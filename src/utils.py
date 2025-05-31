#!/usr/bin/env python3

"""
AUTHOR: Dan Njuguna
DATE: 2025-05-25

DESCRIPTION:
    This module defines helper functions and utilities to be used globally
"""

import os
import sys
import logging
from pathlib import Path

import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Configure logging
def setup_logging(
    log_file: str
):
    """Setup logging configuration"""
    log_file = LOGS_DIR / f"{log_file}"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging("utils.log")


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
        if "index" in data.columns:
            data.drop(columns=["index"], inplace=True)

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
            joblib.dump(model, f)

    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        return False

    logger.info(f"Model saved to {path}")
    return True


# # TODO: Load pickle file
def load_pickle(
        path: str
):
    """Loads a pickle file in a give path and returns the loaded object"""
    try:
        logger.info(f"Starting to load pickle file from path {path}")

        with open(path, "rb") as f:
            model = joblib.load(f)

        logger.info(f"Successfully loaded file of type {type(model)}")

        return model

    except Exception as e:
        logger.error(f"Failed to load pickle file: {e}")
        raise
