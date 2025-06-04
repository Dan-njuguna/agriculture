#!/usr/bin/env python3

"""
AUTHOR: Dan Njuguna
DATE: 2025-05-25

DESCRIPTION:
    This module loads and preprocesses data for use. Also creates a
    test set data for use in evaluating the model accuracy in prod/
    dev.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Import local custom modules
sys.path.append(str(Path(__file__).parent.parent))
from src.utils import save_to_csv, save_pickle, setup_logging

# TODO: Define constants
logger = setup_logging("preprocess.log")


# TODO: Define a data loader
class IDataLoader(ABC):
    """Interface for loading data"""
    @abstractmethod
    def load() -> pd.DataFrame:
        raise NotImplementedError


# TODO: Define a data splitter for train and test set
class IDataSplitter(ABC):
    """Interface for data splitter"""
    @abstractmethod
    def split(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Splits data into train and test sets"""
        raise NotImplementedError


# TODO: Create a feature Engineer
class IFeatureEngineer(ABC):
    """Interface for all feature engineering steps"""
    @abstractmethod
    def engineer(data: pd.DataFrame) -> pd.DataFrame:
        """This runs the Feature engineer and performs engineering on target column"""
        raise NotImplementedError


# TODO: Create a datasaver
class IDataSaver(ABC):
    """Interface for the data saver (saves data to csv files)"""
    @abstractmethod
    def save(data: pd.DataFrame, path: str) -> bool:
        raise NotImplementedError


# TODO: XLSX Data Loader
class XLSXDataLoader(IDataLoader):
    """Loads XLSX data from local path"""
    def __init__(
            self,
            path: str,
            engine: str = "openpyxl"
    ) -> None:
        super().__init__()
        self.path = path
        self.engine = engine

    def load(self) -> pd.DataFrame:
        """Loads XLSX data"""
        try:
            logger.info(f"✅ Loading XLSX File from path {self.path} ...")
            data = pd.read_excel(self.path, engine=self.engine)
            return data

        except Exception as e:
            logger.error(f"❌ Loading data failed ...")
            raise


# TODO: CSV Data loader for use in the training pipeline
class CSVDataLoader(IDataLoader):
    """This class loads csv data from a given path"""
    def __init__(
            self,
            path: str
    ):
        super().__init__()
        self.path = path

    def load(self) -> pd.DataFrame:
        """loads csv data from given path"""
        try:
            logger.info(f"✅ Loading CSV File from path {self.path} ...")
            data = pd.read_csv(self.path)
            logger.info(f"✅ Successfully loaded CSV data ...\n{data.head()}")
            return data

        except Exception as e:
            logger.error(f"❌ Loading data failed ...")
            raise


class TrainTestSplitter(IDataSplitter):
    """This class splits data into train and tests"""
    def __init__(
            self,
            test_size: float = 0.2,
            stratify: bool = False,
            stratify_column: str = "label",
    ):
        super().__init__()
        self.test_size = test_size
        self.stratify = stratify
        self.column = stratify_column

    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Splits data to train and test sets"""
        try:
            if self.stratify:
                train, test = train_test_split(
                    data, 
                    stratify=data[self.column],
                    test_size=self.test_size,
                    random_state=42
                )

                return {
                    "train": train.reset_index(),
                    "test": test.reset_index()
                }

            else:
                train, test = train_test_split(
                    data,
                    test_size=self.test_size,
                    random_state=42
                )

                return {
                    "train": train.reset_index(),
                    "test": test.reset_index()
                }

        except Exception as e:
            logger.error(f"Failed to split data {e}")
            raise


# TODO: Scaler for numerical columns
class BasicScaler(IFeatureEngineer):
    """Perform min-max scaling on the dataframe"""
    def __init__(
            self,
            column: str
    ) -> None:
        super().__init__()
        self.scaler = MinMaxScaler()
        self.column = column
        self.fitted = False

    def fit(
            self,
            data: pd.DataFrame
    ) -> None:
        """Fit the data to the scaler"""
        if self.fitted:
            logger.warning(f"Column {self.column} has been fitted already ...")
            return

        try:
            self.scaler.fit(data[[self.column]])
            self.fitted = True

        except Exception as e:
            logger.error(f"Failed to fit the scaler on the column {self.column}, error {e}")

    def transform(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """transforms data using a fitted scaler and returns the fitted dataframe"""
        if self.fitted:
            data[self.column] = self.scaler.transform(data[[self.column]]).flatten()
            logger.info(f"Successfully transformed column {self.column}")
            return data
        else:
            logger.error(f"❌ No data fitted to scaler object ...")
            raise NotFittedError

    def engineer(
            self,
            data: pd.DataFrame,
            fit: bool = False,
            save_scaler: bool = True
    ) -> pd.DataFrame:
        """This function performs the scaling of numerical columns"""
        if fit:
            self.fit(data)
            if save_scaler:
                scaler_path = os.path.join(
                        Path(__file__).parent.parent, 
                        "model/scalers", 
                        f"{self.column}_scaler.pkl"
                    )

                save_pickle(
                    self.scaler,
                    scaler_path
                )
                data = self.transform(data)

                logger.info(f"Scaler for column {self.column} has been saved to {scaler_path}")
                return data

        else:
            data = self.transform(data)
            return data


# TODO: Create a Label Encoder for categorical columns
class BasicLabelEncoder(IFeatureEngineer):
    """Performs label encoding on categorical columns"""
    def __init__(
            self,
            column: str
    ) -> None:
        super().__init__()
        self.column = column
        self.encoder = LabelEncoder()
        self.fitted = False

    def fit(
            self,
            data: pd.DataFrame
    ) -> None:
        """Fit the data to the encoder"""
        if self.fitted:
            logger.warning(f"Column {self.column} has been fitted already ...")
            return

        try:
            self.encoder.fit(data[self.column])
            self.fitted = True


        except Exception as e:
            logger.error(f"Failed to fit the encoder on the column {self.column}, error {e}")
            raise


    def transform(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """transforms data using a fitted encoder and returns the fitted dataframe"""
        if self.fitted:
            data[self.column] = self.encoder.transform(data[self.column])
            logger.info(f"Successfully transformed column {self.column}")
            return data
        else:
            raise NotFittedError

    def engineer(
            self,
            data: pd.DataFrame,
            fit: bool = False,
            save_encoder: bool = True
    ) -> pd.DataFrame:
        """This function performs the label encoding of categorical columns"""
        if fit:
            self.fit(data)
            if save_encoder:
                encoder_path = os.path.join(
                        Path(__file__).parent.parent,
                        "model/encoders",
                        f"{self.column}_encoder.pkl"
                )

                status = save_pickle(
                    self.encoder,
                    encoder_path
                )
                if not status:
                    logger.warning(f"Failed to save encoder for column {self.column} to {encoder_path}")
                else:
                    logger.info(f"Encoder for column {self.column} has been saved to {encoder_path}")

                data = self.transform(data)

                logger.info(f"Encoder for column {self.column} has been saved to {encoder_path}")
                return data

        else:
            data = self.transform(data)
            return data


class CSVDataSaver(IDataSaver):
    def save(self, data: pd.DataFrame, path: str) -> bool:
        try:
            save_to_csv(data, path)
            return True
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return False


# TODO: Create the DataProcessor
class DataProcessor:
    """This class combines all feature processing steps"""
    def __init__(
            self,
            loader: IDataLoader,
            splitter: IDataSplitter,
            feature_engineers: List[IFeatureEngineer],
            saver: IDataSaver,
    ) -> None:
        self.loader = loader
        self.splitter = splitter
        self.feature_engineers = feature_engineers
        self.saver = saver

    def process(self) -> Dict[str, pd.DataFrame]:
        """This function loads the data, splits it, and applies feature engineering"""
        try:
            # Load the data
            data = self.loader.load()
            logger.info("Data loaded successfully!")

            # Split the data into train and test sets
            split_data = self.splitter.split(data)
            logger.info("Data split into train and test sets successfully!")

            # Apply feature engineering
            for engineer in self.feature_engineers:
                if isinstance(engineer, BasicScaler):
                    split_data["train"] = engineer.engineer(
                        split_data["train"],
                        fit=True,
                        save_scaler=True
                    )
                    split_data["test"] = engineer.engineer(
                        split_data["test"],
                        fit=False,
                        save_scaler=True
                    )
                elif isinstance(engineer, BasicLabelEncoder):
                    split_data["train"] = engineer.engineer(
                        split_data["train"],
                        fit=True,
                        save_encoder=True
                    )
                    split_data["test"] = engineer.engineer(
                        split_data["test"],
                        fit=False,
                        save_encoder=True
                    )
                else:
                    split_data["train"] = engineer.engineer(split_data["train"])
                    split_data["test"] = engineer.engineer(split_data["test"])
            processed_path = os.path.join(Path(__file__).parent.parent, "data", "processed")

            save_status = self.saver.save(split_data["train"], f"{processed_path}/train.csv")
            if save_status:
                logger.info(f"Successfully saved to csv")

            self.saver.save(split_data["test"], f"{processed_path}/test.csv")
            logger.info("Feature engineering applied successfully!")
            return split_data

        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            raise


# TODO: Run the module as CLI Tool
def main():
    """This function compiles the module as a CLI tool"""
    argsparser = argparse.ArgumentParser(
        prog="Data Preprocess Pipeline",
        usage="Module runs the feature engineering pipeline and saves the train and test sets as CSV."
    )

    scale_default = [
        "N",
        "P",
        "K",
        "humidity",
        "temperature",
        "ph",
        "rainfall"
    ]

    encode_default = [
        "label"
    ]

    argsparser.add_argument(
        "--data_path",
        required=True,
        help="Path to the XLSX data file"
    )
    argsparser.add_argument(
        "--scale",
        action="append",
        default=scale_default,
        help="Name(s) of the numerical column(s) to scale"
    )
    argsparser.add_argument(
        "--encode",
        action="append",
        default=encode_default,
        help="Name(s) of the categorical column(s) to encode"
    )
    args = argsparser.parse_args()

    loader = XLSXDataLoader(args.data_path)

    stratify_column = "label"
    splitter = TrainTestSplitter(
        stratify=True,
        stratify_column=stratify_column
    )

    feature_engineers = []
    for col in args.scale:
        feature_engineers.append(BasicScaler(col))
    for col in args.encode:
        feature_engineers.append(BasicLabelEncoder(col))
    saver = CSVDataSaver()

    processor = DataProcessor(
        loader=loader,
        splitter=splitter,
        feature_engineers=feature_engineers,
        saver=saver
    )

    try:
        processor.process()
        logger.info("Data processing completed successfully!")
    except Exception as e:
        logger.info("Data processing failed: %s", e)


if __name__ == '__main__':
    main()
