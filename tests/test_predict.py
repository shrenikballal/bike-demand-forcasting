
import pandas as pd
import pytest

from src.predict import validate_history, validate_next_day_input


def test_history_requires_at_least_28_rows():
    history = pd.DataFrame({
        "dteday": pd.date_range("2022-01-01", periods=20),
        "cnt": range(20),
    })

    with pytest.raises(ValueError):
        validate_history(history)


def test_history_rejects_duplicate_dates():
    history = pd.DataFrame({
        "dteday": pd.to_datetime([
            "2022-01-01",
            "2022-01-02",
            "2022-01-02",
        ] * 10),
        "cnt": range(30),
    })

    with pytest.raises(ValueError):
        validate_history(history)


def test_history_rejects_missing_target():
    history = pd.DataFrame({
        "dteday": pd.date_range("2022-01-01", periods=30),
        "cnt": [100] * 29 + [None],
    })

    with pytest.raises(ValueError):
        validate_history(history)


def test_next_day_input_requires_all_features():
    incomplete_input = {
        "dteday": "2022-02-01",
    }

    with pytest.raises(ValueError):
        validate_next_day_input(incomplete_input)
