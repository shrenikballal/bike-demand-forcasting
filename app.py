
import streamlit as st
import pandas as pd

from src.preprocess import load_and_validate_data
from src.predict import predict_next_day


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Bike Rental Demand Forecast",
    page_icon="🚲",
    layout="centered",
)

st.title("🚲 Bike Rental Demand Forecast")
st.write(
    "Predict next-day bike rental demand using historical "
    "rental patterns and weather conditions."
)


# --------------------------------------------------
# Load historical data
# --------------------------------------------------

@st.cache_data
def load_history():
    return load_and_validate_data()


try:
    history = load_history()
except Exception as error:
    st.error(f"Unable to load historical data: {error}")
    st.stop()


history["dteday"] = pd.to_datetime(history["dteday"])

last_date = history["dteday"].iloc[-1]
forecast_date = last_date + pd.Timedelta(days=1)


# --------------------------------------------------
# Display historical information
# --------------------------------------------------

st.subheader("Historical Data")

col1, col2 = st.columns(2)

with col1:
    st.metric("Historical Records", len(history))

with col2:
    st.metric(
        "Last Available Date",
        last_date.strftime("%d %b %Y"),
    )

st.info(
    f"Your next forecast date is "
    f"**{forecast_date.strftime('%d %B %Y')}**."
)

st.warning(
    "This dataset ends in 2012. The forecast is a demonstration "
    "outside the model's training period, not a reliable current "
    "operational forecast. Update the historical data and retrain "
    "the model before using it for real-world decisions."
)


# --------------------------------------------------
# Weather and calendar inputs
# --------------------------------------------------

st.subheader("Forecast Conditions")

st.write("Enter the expected weather conditions for the forecast date.")

with st.form("forecast_form"):

    holiday = st.checkbox("Is this date a holiday?")

    weather_options = {
        "Clear / Partly cloudy": 1,
        "Mist / Cloudy": 2,
        "Light rain / Light snow": 3,
        "Heavy rain / Snow": 4,
    }

    weather_label = st.selectbox(
        "Weather condition",
        options=list(weather_options.keys()),
    )

    temp = st.slider(
        "Temperature (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
    )

    atemp = st.slider(
        "Feels-like temperature (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.30,
        step=0.01,
    )

    hum = st.slider(
        "Humidity (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.60,
        step=0.01,
    )

    windspeed = st.slider(
        "Windspeed (normalized)",
        min_value=0.0,
        max_value=1.0,
        value=0.15,
        step=0.01,
    )

    submitted = st.form_submit_button(
        "Predict Rental Demand",
        type="primary",
    )


# --------------------------------------------------
# Generate prediction
# --------------------------------------------------

if submitted:

    month = forecast_date.month
    weekday = forecast_date.dayofweek

    # Bike Sharing dataset season encoding:
    # 1 = Spring, 2 = Summer, 3 = Fall, 4 = Winter
    season = ((month % 12) // 3) + 1

    workingday = int(weekday < 5 and not holiday)

    next_day_data = {
        "dteday": forecast_date,
        "season": season,
        "mnth": month,
        "holiday": int(holiday),
        "weekday": weekday,
        "workingday": workingday,
        "weathersit": weather_options[weather_label],
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
    }

    try:
        result = predict_next_day(
            history_df=history,
            next_day_data=next_day_data,
        )

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Bike Rentals",
            value=f"{result['predicted_rentals']:,}",
        )

        st.success(
            f"Forecast generated for {result['forecast_date']}"
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# Historical demand chart
# --------------------------------------------------

st.divider()
st.subheader("Recent Historical Demand")

recent_history = history.tail(30).copy()

recent_history = recent_history.set_index("dteday")

st.line_chart(recent_history["cnt"])

st.caption(
    "Historical daily rental counts for the most recent 30 records."
)
