
import numpy as np
import pandas as pd

from src.config import DATE_COLUMN, TARGET_COLUMN
from src.features import create_features
from src.predict import predict_next_day


class DummyModel:
    def predict(self, X):
        assert len(X) == 1
        assert not X.isna().any().any()
        return np.array([1234.6])


def test_complete_prediction_pipeline(monkeypatch):
    # Create 35 days of historical data
    dates = pd.date_range("2022-01-01", periods=35)

    history = pd.DataFrame({
        DATE_COLUMN: dates,
        TARGET_COLUMN: np.arange(1000, 1035),
    })

    # Prepare the next day's known information
    forecast_date = dates[-1] + pd.Timedelta(days=1)

    next_day_data = {
        DATE_COLUMN: forecast_date,
        "season": 1,
        "mnth": 2,
        "holiday": 0,
        "weekday": forecast_date.dayofweek,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.3,
        "atemp": 0.3,
        "hum": 0.6,
        "windspeed": 0.15,
    }

    # Generate the feature schema expected by the model
    combined = pd.concat([
        history,
        pd.DataFrame([{
            **{column: None for column in history.columns},
            **next_day_data,
            TARGET_COLUMN: np.nan,
        }])
    ], ignore_index=True)

    feature_df = create_features(combined, drop_missing=False)

    excluded = {
        TARGET_COLUMN, "casual", "registered",
        "instant", DATE_COLUMN, "yr",
    }

    feature_columns = [
        column for column in feature_df.select_dtypes(include="number").columns
        if column not in excluded
    ]

    # Replace model loading with a predictable test model
    monkeypatch.setattr(
        "src.predict.load_model_and_schema",
        lambda: (
            DummyModel(),
            {"feature_columns": feature_columns},
        ),
    )

    result = predict_next_day(history, next_day_data)

    assert result["forecast_date"] == forecast_date.strftime("%Y-%m-%d")
    assert result["predicted_rentals"] == 1235
