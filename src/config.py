
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Raw dataset
RAW_DATA_PATH = RAW_DATA_DIR / "day.csv"

# Model directory
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "bike_rental_model.joblib"

# Reports directory
REPORT_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORT_DIR / "model_metrics.json"

# Experiment tracking
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlruns.db'}"
MLFLOW_EXPERIMENT_NAME = "Bike Rental Demand Forecasting"

# Target configuration
TARGET_COLUMN = "cnt"
DATE_COLUMN = "dteday"

# Reproducibility
RANDOM_STATE = 42
