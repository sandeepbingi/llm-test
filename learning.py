import streamlit as st
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from datetime import datetime, timedelta

# Define folders
DATA_FOLDER = "data"
TRACKED_FILES_LOG = "tracked_files.txt"
MODEL_FOLDER = "saved_models"

# Function to load CSV files and reshape them
def load_and_preprocess_data():
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]

    all_data = []
    for file in files:
        file_path = os.path.join(DATA_FOLDER, file)
        df = pd.read_csv(file_path)

        # Reshape wide format to long format
        df_long = df.melt(id_vars=['channel'], var_name='timestamp', value_name='count')
        
        # Convert timestamp to datetime format
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], errors='coerce')

        # Fill missing values with 0
        df_long['count'].fillna(0, inplace=True)

        all_data.append(df_long)

    if all_data:
        return pd.concat(all_data, ignore_index=True), files
    else:
        return pd.DataFrame(), files

# Function to track processed files
def get_processed_files():
    if not os.path.exists(TRACKED_FILES_LOG):
        return set()
    with open(TRACKED_FILES_LOG, "r") as f:
        return set(f.read().splitlines())

def update_processed_files(new_files):
    with open(TRACKED_FILES_LOG, "a") as f:
        for file in new_files:
            f.write(file + "\n")

# Function to save model
def save_model(model, channel, method):
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{method}.pkl")
    joblib.dump(model, model_path)

# Function to load model
def load_model(channel, method):
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{method}.pkl")
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# Function to train ARIMA model
def train_arima(df):
    model = ARIMA(df['count'], order=(5,1,0))
    model_fit = model.fit()
    return model_fit

# Function to train Prophet model
def train_prophet(df):
    df = df.rename(columns={'timestamp': 'ds', 'count': 'y'})
    model = Prophet()
    model.fit(df)
    return model

# Function to forecast
def forecast(model, future_dates, method):
    if method == 'arima':
        return model.forecast(steps=len(future_dates))
    else:
        future = pd.DataFrame({'ds': future_dates})
        return model.predict(future)['yhat']

# Function to evaluate model performance
def evaluate_model(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted)
    return rmse, mape

# Streamlit UI
st.title("Automated Time Series Forecasting")

# Load and process data
df, all_files = load_and_preprocess_data()
processed_files = get_processed_files()
new_files = set(all_files) - processed_files

if new_files:
    st.warning(f"New files detected! Retraining models... ({len(new_files)} new files)")
    update_processed_files(new_files)

if df.empty:
    st.error("No data found in the folder. Please check your CSV files.")
    st.stop()

# Select channel
channels = df['channel'].unique().tolist()
selected_channels = st.multiselect("Select Channels", channels)

# Select forecasting method
method = st.radio("Select Model", ["ARIMA", "Prophet"]).lower()

# Select future date
future_date = st.date_input("Select Future Date", min_value=datetime.today())

if st.button("Generate Forecast"):
    results = []

    for channel in selected_channels:
        channel_df = df[df['channel'] == channel]

        # Load or train model
        model = load_model(channel, method)

        if model is None or new_files:
            st.info(f"Training model for {channel}")
            if method == 'arima':
                model = train_arima(channel_df)
            else:
                model = train_prophet(channel_df)
            save_model(model, channel, method)

        # Generate future dates (24-hour forecast)
        future_dates = pd.date_range(start=channel_df['timestamp'].max() + timedelta(hours=1), periods=24, freq='H')

        # Forecast
        forecast_values = forecast(model, future_dates, method)

        # Evaluate
        if len(channel_df) > 24:
            rmse, mape = evaluate_model(channel_df['count'][-24:], forecast_values[:24])
        else:
            rmse, mape = None, None

        # Store results
        for i in range(len(future_dates)):
            results.append([channel, future_dates[i], forecast_values[i], rmse, mape])

    # Convert to DataFrame
    forecast_df = pd.DataFrame(results, columns=['Channel', 'Datetime', 'Forecasted Count', 'RMSE', 'MAPE'])

    # Display Results
    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    # Download Button
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast as CSV", csv, "forecast_results.csv", "text/csv")

    # Plot Forecast
    st.subheader("Trend Visualization")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=forecast_df, x='Datetime', y='Forecasted Count', hue='Channel', marker='o')
    plt.xlabel("Datetime")
    plt.ylabel("Count")
    plt.title("Forecasted Count per Channel")
    plt.xticks(rotation=45)
    st.pyplot(plt)
