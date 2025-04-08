# app.py

import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Constants
DATA_FOLDER_LOGINS = "C:/Users/Desktop/Logins_Payments_LLM/ForecastX/Test_Login_Reports"
DATA_FOLDER_PAYMENTS = "C:/Users/Desktop/Logins_Payments_LLM/ForecastX/Test_Payment_Reports"
MODEL_FOLDER = "forecast_models"

# Helper functions
def calculate_accuracy(actual_values, forecast_values):
    actual_values = np.array(actual_values)
    forecast_values = np.array(forecast_values)
    min_length = min(len(actual_values), len(forecast_values))
    actual_values = actual_values[:min_length]
    forecast_values = forecast_values[:min_length]
    actual_values = np.where(actual_values == 0, 1e-10, actual_values)
    mask = actual_values != 0
    if not any(mask):
        return None
    mape = np.mean(np.abs((actual_values[mask] - forecast_values[mask]) / actual_values[mask])) * 100
    return round(mape, 2)

def get_actual_values(channel, type_, future_dates):
    folder_path = DATA_FOLDER_LOGINS if type_ == "logins" else DATA_FOLDER_PAYMENTS
    if channel == 'ny-ios' and type_ == 'payments':
        channel = 'ny_ios'

    all_actual_values = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_long = df.melt(id_vars=['Channel'], var_name='timestamp', value_name='count')
        df_long['timestamp'] = df_long['timestamp'].str.replace(':AM', ' AM').str.replace(':PM', ' PM')
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format="%m/%d/%y %I:%M:%S %p", errors='coerce')
        df_long = df_long[df_long["Channel"] == channel]
        df_filtered = df_long[df_long['timestamp'].dt.date == future_dates[0].date()]
        if not df_filtered.empty:
            all_actual_values.extend(df_filtered["count"].tolist())
    return all_actual_values if all_actual_values else None

def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

def forecast(channel, type_, future_date):
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{type_}_Prophet.pkl")
    if not os.path.exists(model_path):
        return None, None, None
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data['model']
    future_dates = pd.date_range(start=future_date, periods=24, freq='H')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast_values = model.predict(future_df)['yhat']
    forecast_values = [max(0, round(value)) for value in forecast_values]
    actual_values = get_actual_values(channel, type_, future_dates)
    if actual_values:
        mape = calculate_accuracy(actual_values, forecast_values)
        return actual_values, forecast_values, mape
    return None, None, None

# --- Streamlit UI ---
st.title("Forecast Accuracy Dashboard")

channel_list = ['ny-ios', 'ny-android', 'la-ios', 'la-android']  # Modify this as needed
selected_channel = st.selectbox("Select Channel", channel_list)
selected_type = st.radio("Select Type", ["logins", "payments"])
selected_date = st.date_input("Select Date")
submitted = st.button("Run Forecast")

if submitted:
    actual, predicted, mape = forecast(selected_channel, selected_type, pd.to_datetime(selected_date))
    if actual and predicted:
        st.subheader(f"MAPE Accuracy: {100 - mape:.2f}%")
        st.line_chart(pd.DataFrame({
            "Actual": actual,
            "Forecasted": predicted
        }))
    else:
        st.error("Could not retrieve forecast or actual values. Check model or data availability.")
