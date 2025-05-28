#!/usr/bin/env python3

"""
AUTHOR: Dan Njuguna
DATE: 2025-05-26

DESCRIPTION:
    This module defines the training pipeline for the machine learning model.
"""

import sys
import logging
import joblib
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import sklearn.metrics as metrics
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import save_to_csv
from src.preprocess import CSVDataLoader

# TODO: Define constants
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "model"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed"

# Create directories if they don't exist
for directory in [LOGS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# TODO: Configure logging
def setup_logging():
    """Setup logging configuration"""
    log_file = LOGS_DIR / "train.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)  # Also log to console
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# TODO: Create a trainer interface
class ITrainer(ABC):
    """Interface for the training method"""
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.DataFrame, params: Dict) -> Any:
        """Train the model with given data and parameters"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the trained model"""
        raise NotImplementedError


class BaseTrainer(ITrainer):
    """Base trainer class with common functionality"""

    def __init__(self, model, model_name: str = "model", use_scaling: bool = True):
        self.model = model
        self.model_name = model_name
        self.use_scaling = use_scaling
        self.best_model = None
        self.scaler = StandardScaler() if use_scaling else None

    def _create_pipeline(self) -> Pipeline:
        """Create a pipeline with optional scaling"""
        steps = []
        if self.use_scaling:
            steps.append(('scaler', StandardScaler()))
        steps.append(('classifier', self.model))
        return Pipeline(steps)

    def _cross_validation(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        params: Dict,
        cv: int = 5,
        n_iter: int = 5
    ) -> Any:
        """Cross validate the model to find the best hyperparameters"""
        try:
            logger.info(f"✅ Starting cross validation for {self.model_name}...")

            # Fix target variable shape
            y_fixed = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()

            # Create pipeline
            pipeline = self._create_pipeline()

            # Update parameter names for pipeline
            pipeline_params = {}
            for key, value in params.items():
                pipeline_params[f'classifier__{key}'] = value

            # Suppress warnings during cross-validation
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                clf = RandomizedSearchCV(
                    estimator=pipeline,
                    param_distributions=pipeline_params,
                    n_iter=n_iter,
                    n_jobs=-1,  # Use all available cores
                    scoring="accuracy",
                    cv=cv,
                    refit=True,
                    return_train_score=True,
                    random_state=42
                )

                clf.fit(X, y_fixed)

            logger.info(f"✅ Best model parameters: {clf.best_params_}")
            logger.info(f"✅ Best CV score: {clf.best_score_:.6f}")

            return clf.best_estimator_

        except Exception as e:
            logger.error(f"❌ Failed to cross validate the {self.model_name}: {str(e)}")
            raise

    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.DataFrame) -> Dict[str, float]:
        """Evaluate the trained model"""
        try:
            y_fixed = y.values.ravel() if hasattr(y, 'values') else np.array(y).ravel()
            y_pred = model.predict(X)

            # Calculate metrics
            accuracy = metrics.accuracy_score(y_fixed, y_pred)
            precision = metrics.precision_score(y_fixed, y_pred, average='weighted', zero_division=0)
            recall = metrics.recall_score(y_fixed, y_pred, average='weighted', zero_division=0)
            f1 = metrics.f1_score(y_fixed, y_pred, average='weighted', zero_division=0)

            # Cross-validation score
            cv_scores = cross_val_score(model, X, y_fixed, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()

            results = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }

            logger.info(f"✅ {self.model_name} Evaluation Results:")
            for metric, value in results.items():
                logger.info(f"   {metric}: {value:.6f}")

            return results

        except Exception as e:
            logger.error(f"❌ Failed to evaluate {self.model_name}: {str(e)}")
            raise

    def save_model(self, model: Any, filename: str = None) -> str:
        """Save the trained model"""
        try:
            if filename is None:
                filename = f"{self.model_name}_best_model.pkl"

            model_path = MODELS_DIR / filename
            joblib.dump(model, model_path)
            logger.info(f"✅ Model saved to: {model_path}")
            return str(model_path)
        except Exception as e:
            logger.error(f"❌ Failed to save model: {str(e)}")
            raise


class LogisticTrainer(BaseTrainer):
    """Logistic Regression trainer implementation"""
    def __init__(self):
        super().__init__(
            model=LogisticRegression(random_state=42),
            model_name="LogisticRegression",
            use_scaling=True
        )
    def train(self, X: pd.DataFrame, y: pd.DataFrame, params: Dict = None) -> Any:
        """Train a logistic regression model"""
        if params is None:
            params = {
                "C": [0.001, 0.01, 0.1, 1, 2, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear", "lbfgs", "newton-cg", "sag", "saga"]  # Fixed typo: sage -> saga
            }
        self.best_model = self._cross_validation(X, y, params)
        return self.best_model


class RandomForestTrainer(BaseTrainer):
    """Random Forest trainer implementation"""
    def __init__(self):
        super().__init__(
            model=RandomForestClassifier(random_state=42),
            model_name="RandomForest",
            use_scaling=False  # Random Forest doesn't need scaling
        )

    def train(self, X: pd.DataFrame, y: pd.DataFrame, params: Dict = None) -> Any:
        """Train a random forest model"""
        if params is None:
            params = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4]
            }

        self.best_model = self._cross_validation(X, y, params, n_iter=15)
        return self.best_model


class SVMTrainer(BaseTrainer):
    """SVM trainer implementation"""
    def __init__(self):
        super().__init__(
            model=SVC(random_state=42),
            model_name="SVM",
            use_scaling=True
        )

    def train(self, X: pd.DataFrame, y: pd.DataFrame, params: Dict = None) -> Any:
        """Train an SVM model"""
        if params is None:
            params = {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf", "poly"],
                "gamma": ["scale", "auto"]
            }
        self.best_model = self._cross_validation(X, y, params, n_iter=10)
        return self.best_model


class ModelTrainingPipeline:
    """Main training pipeline that orchestrates model training"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.loader = CSVDataLoader(data_path)
        self.trainers = {
            'logistic': LogisticTrainer(),
            'random_forest': RandomForestTrainer(),
            'svm': SVMTrainer()
        }
        self.results = {}

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare training data"""
        try:
            logger.info("📊 Loading training data...")
            data = self.loader.load()

            X = data.drop(["label"], axis=1)
            y = data[["label"]]

            logger.info(f"✅ Data loaded successfully. Shape: {data.shape}")
            logger.info(f"   Features: {X.shape[1]}")
            logger.info(f"   Samples: {X.shape[0]}")
            logger.info(f"   Unique labels: {y['label'].nunique()}")

            return X, y

        except Exception as e:
            logger.error(f"❌ Failed to load data: {str(e)}")
            raise

    def train_model(self, model_type: str, X: pd.DataFrame, y: pd.DataFrame, custom_params: Dict = None) -> Any:
        """Train a specific model type"""
        if model_type not in self.trainers:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.trainers.keys())}")

        logger.info(f"🚀 Training {model_type} model...")
        trainer = self.trainers[model_type]

        # Train the model
        model = trainer.train(X, y, custom_params)

        # Evaluate the model
        evaluation = trainer.evaluate(model, X, y)

        # Save the model
        model_path = trainer.save_model(model)

        # Store results
        self.results[model_type] = {
            'model': model,
            'evaluation': evaluation,
            'model_path': model_path,
            'trainer': trainer
        }

        return model

    def train_all_models(self, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Train all available models"""
        logger.info("🔄 Training all models...")

        for model_type in self.trainers.keys():
            try:
                self.train_model(model_type, X, y)
            except Exception as e:
                logger.error(f"❌ Failed to train {model_type}: {str(e)}")
                continue

        return self.results

    def get_best_model(self) -> Tuple[str, Any, Dict]:
        """Get the best performing model"""
        if not self.results:
            raise ValueError("No models have been trained yet!")

        best_model_type = None
        best_accuracy = 0

        for model_type, result in self.results.items():
            accuracy = result['evaluation']['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_type = model_type

        logger.info(f"🏆 Best model: {best_model_type} (Accuracy: {best_accuracy:.6f})")

        return (
            best_model_type,
            self.results[best_model_type]['model'],
            self.results[best_model_type]['evaluation']
        )

    def run_pipeline(self, model_types: list = None) -> Dict:
        """Run the complete training pipeline"""
        try:
            logger.info("🎯 Starting training pipeline...")

            # Load data
            X, y = self.load_data()

            # Train specified models or all models
            if model_types is None:
                results = self.train_all_models(X, y)
            else:
                results = {}
                for model_type in model_types:
                    results[model_type] = self.train_model(model_type, X, y)

            # Get best model
            best_model_type, best_model, best_evaluation = self.get_best_model()

            logger.info("✅ Training pipeline completed successfully!")

            return {
                'results': results,
                'best_model': {
                    'type': best_model_type,
                    'model': best_model,
                    'evaluation': best_evaluation
                }
            }

        except Exception as e:
            logger.error(f"❌ Training pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the training pipeline"""
    parser = argparse.ArgumentParser(description="Train machine learning models")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(TRAIN_PATH / "train.csv"),
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic", "random_forest", "svm"],
        default=None,
        help="Models to train (default: all models)"
    )

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = ModelTrainingPipeline(args.data_path)

        # Run training
        results = pipeline.run_pipeline(args.models)

        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)

        for model_type, result in results['results'].items():
            eval_results = result['evaluation']
            print(f"\n{model_type.upper()}:")
            print(f"  Accuracy: {eval_results['accuracy']:.6f}")
            print(f"  F1-Score: {eval_results['f1_score']:.6f}")
            print(f"  CV Mean: {eval_results['cv_mean']:.6f} ± {eval_results['cv_std']:.6f}")

        best = results['best_model']
        print(f"\n🏆 BEST MODEL: {best['type'].upper()}")
        print(f"   Accuracy: {best['evaluation']['accuracy']:.6f}")

    except Exception as e:
        logger.error(f"❌ Main execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
