
import joblib
import pandas as pd

from src.config import MODEL_PATH, MODEL_DIR, DATE_COLUMN, TARGET_COLUMN
from src.features import create_features


# Raw columns required for next-day prediction.

NEXT_DAY_COLUMNS = [
    "season", "mnth", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
]



def load_model_and_schema():
    """Load the trained model and its feature schema."""

    schema_path = MODEL_DIR / "feature_schema.json"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not schema_path.exists():
        raise FileNotFoundError(
            f"Feature schema not found: {schema_path}"
        )

    model = joblib.load(MODEL_PATH)

    import json

    with open(schema_path, "r", encoding="utf-8") as file:
        schema = json.load(file)

    return model, schema


def validate_history(history_df):
    """Validate historical data required for forecasting."""

    if not isinstance(history_df, pd.DataFrame):
        raise TypeError("History must be a pandas DataFrame.")

    required = [DATE_COLUMN, TARGET_COLUMN]

    missing = [
        col for col in required
        if col not in history_df.columns
    ]

    if missing:
        raise ValueError(f"Missing history columns: {missing}")

    if len(history_df) < 28:
        raise ValueError("At least 28 days of history are required.")

    history = history_df.copy()

    history[DATE_COLUMN] = pd.to_datetime(
        history[DATE_COLUMN],
        errors="raise"
    )

    history = history.sort_values(DATE_COLUMN).reset_index(drop=True)

    if history[DATE_COLUMN].duplicated().any():
        raise ValueError("Duplicate dates found in history.")

    if history[TARGET_COLUMN].isnull().any():
        raise ValueError("Historical rental counts contain missing values.")

    if (history[TARGET_COLUMN] < 0).any():
        raise ValueError("Historical rental counts cannot be negative.")

    date_differences = history[DATE_COLUMN].diff().dropna()

    if not (date_differences == pd.Timedelta(days=1)).all():
        raise ValueError("History must contain consecutive daily records.")

    return history


def validate_next_day_input(next_day_data):
    """Validate the known information for the forecast date."""

    if not isinstance(next_day_data, dict):
        raise TypeError("Next-day input must be a dictionary.")

    required = [DATE_COLUMN] + NEXT_DAY_COLUMNS

    missing = [
        col for col in required
        if col not in next_day_data
    ]

    if missing:
        raise ValueError(
            f"Missing next-day input fields: {missing}"
        )

    next_day = pd.DataFrame([next_day_data])

    next_day[DATE_COLUMN] = pd.to_datetime(
        next_day[DATE_COLUMN],
        errors="raise"
    )

    for col in NEXT_DAY_COLUMNS:
        next_day[col] = pd.to_numeric(
            next_day[col],
            errors="raise"
        )

    if next_day[NEXT_DAY_COLUMNS].isnull().any().any():
        raise ValueError("Next-day input contains missing values.")

    return next_day.iloc[0].to_dict()


def predict_next_day(history_df, next_day_data):
    """
    Predict rental demand for the next day.

    Parameters
    ----------
    history_df : pd.DataFrame
        Historical daily dataset containing dteday and cnt.

    next_day_data : dict
        Next-day date and known weather/calendar features.

    Returns
    -------
    dict
        Forecast date and predicted rental count.
    """

    history = validate_history(history_df)

    next_day_data = validate_next_day_input(next_day_data)

    last_date = history[DATE_COLUMN].iloc[-1]
    forecast_date = next_day_data[DATE_COLUMN]

    if forecast_date != last_date + pd.Timedelta(days=1):
        raise ValueError(
            "Forecast date must be exactly one day after "
            "the latest historical date."
        )

    # Build the future row without knowing its rental count.
    future_row = {col: None for col in history.columns}

    future_row.update(next_day_data)

    future_row[DATE_COLUMN] = forecast_date
    future_row[TARGET_COLUMN] = float("nan")

    # Target-derived columns are unavailable for the forecast date.
    for col in ["casual", "registered", "instant"]:
        if col in future_row:
            future_row[col] = float("nan")

    future_df = pd.DataFrame([future_row])

    # Combine history and the future row.
    combined_df = pd.concat(
        [history, future_df],
        ignore_index=True
    )

    # Generate features without dropping the future row.
    feature_df = create_features(
        combined_df,
        drop_missing=False
    )

    forecast_features = feature_df.iloc[[-1]]

    model, schema = load_model_and_schema()

    feature_columns = schema["feature_columns"]

    missing_features = [
        col for col in feature_columns
        if col not in forecast_features.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    X_forecast = forecast_features[feature_columns]

    if X_forecast.isnull().any().any():
        raise ValueError(
            "Forecast features contain missing values."
        )

    prediction = float(model.predict(X_forecast)[0])

    return {
        "forecast_date": forecast_date.strftime("%Y-%m-%d"),
        "predicted_rentals": max(0, round(prediction)),
    }
