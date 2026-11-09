
import numpy as np
import pandas as pd

from src.features import create_features


def sample_data():
    dates = pd.date_range("2022-01-01", periods=35, freq="D")

    return pd.DataFrame({
        "dteday": dates,
        "cnt": np.arange(35, dtype=float),
    })


def test_feature_creation():
    df = sample_data()

    result = create_features(df)

    assert "year" in result.columns
    assert "month" in result.columns
    assert "day_of_week" in result.columns
    assert "cnt_lag_1" in result.columns
    assert "cnt_rolling_mean_7" in result.columns


def test_lag_features_use_previous_values():
    df = sample_data()

    result = create_features(df)

    first_valid_row = result.iloc[0]

    assert first_valid_row["cnt_lag_1"] == 27
    assert first_valid_row["cnt_lag_7"] == 21
    assert first_valid_row["cnt_lag_28"] == 0


def test_rolling_features_do_not_use_current_target():
    df = sample_data()

    result = create_features(df)

    first_valid_row = result.iloc[0]

    expected_mean = np.mean(np.arange(28))

    assert np.isclose(
        first_valid_row["cnt_rolling_mean_28"],
        expected_mean
    )


def test_original_dataframe_is_not_modified():
    df = sample_data()
    original_columns = df.columns.tolist()

    create_features(df)

    assert df.columns.tolist() == original_columns
