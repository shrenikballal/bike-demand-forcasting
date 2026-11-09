
# Model Evaluation Report
## Bike Rental Demand Forecasting

**Model:** Linear Regression  
**Evaluation Type:** Chronological Holdout Testing  
**Target Variable:** `cnt` (Total Daily Rentals)

---

## 1. Objective

The objective of this project is to predict daily bike rental demand using historical rental patterns, calendar information, and weather conditions.

The model is evaluated on unseen historical data to assess its ability to generalize to future observations.

## 2. Dataset Overview

The Bike Sharing daily dataset contains 731 observations covering:

- Start date: 2011-01-01
- End date: 2012-12-31

The target variable is:

- `cnt`: Total number of daily bike rentals.

The following columns are excluded from model training:

| Column | Reason |
|---|---|
| `cnt` | Target variable |
| `casual` | Component of target variable |
| `registered` | Component of target variable |
| `instant` | Record identifier |
| `dteday` | Used for date-based feature engineering and splitting |
| `yr` | Redundant with the engineered `year` feature |

Excluding `casual` and `registered` prevents direct target leakage.

## 3. Feature Engineering

The following features are generated.

### 3.1 Calendar Features

- Year
- Month
- Day
- Day of week
- Day of year
- Week of year
- Weekend indicator

These features capture seasonal, monthly, and weekly demand patterns.

### 3.2 Lag Features

Historical rental counts are used to capture temporal dependencies.

| Feature | Description |
|---|---|
| `cnt_lag_1` | Previous day's demand |
| `cnt_lag_2` | Demand two days earlier |
| `cnt_lag_3` | Demand three days earlier |
| `cnt_lag_7` | Demand one week earlier |
| `cnt_lag_14` | Demand two weeks earlier |
| `cnt_lag_21` | Demand three weeks earlier |
| `cnt_lag_28` | Demand four weeks earlier |

### 3.3 Rolling Features

Rolling mean and standard deviation are calculated using windows of:

- 7 days
- 14 days
- 28 days

All rolling calculations use a one-day shift to ensure that the current day's target is not included.

## 4. Training and Testing Strategy

A chronological split is used instead of random sampling.

| Dataset | Date Range |
|---|---|
| Training | 2011-01-29 to 2012-09-30 |
| Testing | 2012-10-01 to 2012-12-31 |

After generating lag and rolling features, the first 28 observations are removed because sufficient historical data is not available for those rows.

Final dataset sizes:

- Training observations: 611
- Testing observations: 92

This strategy simulates forecasting on future observations using historical training data.

## 5. Selected Model

**Linear Regression**

Linear Regression was selected as the current model based on its performance in the evaluated experiments.

It provides a relatively simple and interpretable baseline for daily demand forecasting.

## 6. Evaluation Metrics

The model was evaluated using three regression metrics.

### Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted rental counts.

### Root Mean Squared Error (RMSE)

RMSE penalizes larger prediction errors more heavily than MAE.

### R-squared (R²)

R² measures the proportion of variation in the target variable explained by the model.

## 7. Evaluation Results

| Metric | Result |
|---|---:|
| MAE | 782.02 |
| RMSE | 1112.46 |
| R² | 0.66 |

### Interpretation

**MAE = 782.02**

The model's average absolute prediction error is approximately 782 rentals per day.

**RMSE = 1112.46**

The RMSE is higher than MAE, indicating that larger prediction errors have some influence on the overall error.

**R² = 0.66**

The model explains approximately 66% of the variation in daily rental demand on the held-out test dataset.

Overall, the model captures meaningful demand patterns but still has room for improvement.

## 8. Historical Prediction Validation

The prediction pipeline was additionally tested using historical data.

For 2012-10-01:

| Metric | Value |
|---|---:|
| Actual rentals | 6,778 |
| Predicted rentals | 6,724 |
| Absolute error | 54 |
| Absolute percentage error | 0.80% |

This example confirms that the inference pipeline can generate a prediction using historical data available before the forecast date.

However, this is one observation and should not be interpreted as the model's overall accuracy.

## 9. Automated Testing

The project includes nine automated tests.

| Test Area | Tests |
|---|---:|
| Feature engineering | 4 |
| Prediction input validation | 4 |
| Prediction pipeline integration | 1 |
| **Total** | **9** |

All nine tests passed successfully.

Testing covers:

- Feature generation
- Lag calculations
- Rolling feature leakage prevention
- Input validation
- Duplicate date detection
- Missing target detection
- End-to-end prediction execution

## 10. Limitations

1. The dataset covers only two years of historical observations.
2. The model may not generalize well to future years or changing rental patterns.
3. Linear Regression may not capture complex nonlinear relationships.
4. The held-out test period contains only 92 observations.
5. Weather information must be available in advance for operational forecasting.
6. The current model has not been validated using a longer-term or external dataset.

**Important:** The dataset ends on 2012-12-31. Predictions for dates beyond this period are demonstrations of the inference pipeline, not reliable current demand forecasts.

## 11. Future Improvements

Potential improvements include:

- Collecting updated bike rental data.
- Evaluating Random Forest and Gradient Boosting models.
- Using time-series cross-validation.
- Evaluating seasonal and weather-related interactions.
- Monitoring prediction errors over time.
- Integrating MLflow for experiment tracking.
- Deploying the model through a production API.
- Implementing automated model retraining.

## 12. Conclusion

The Linear Regression model provides a useful baseline for daily bike rental demand forecasting.

With an MAE of 782.02, RMSE of 1112.46, and R² of 0.66, it captures a meaningful portion of the demand variation in the historical test dataset.

The project also demonstrates a complete machine learning workflow, including data validation, feature engineering, model training, evaluation, prediction, and automated testing.

Further improvements should focus on updated data, stronger temporal validation, and model monitoring before operational deployment.
