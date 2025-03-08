import os
import pandas as pd
import pickle
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet

# Define folders
data_folder_logins = "data/logins"
data_folder_payments = "data/payments"
model_folder = "models"
processed_files_log = "processed_files.txt"

# Create model folder if not exists
os.makedirs(model_folder, exist_ok=True)

def load_data(data_folder):
    """
    Load all CSV files from a specified folder, reshape them into long format, 
    and parse timestamps correctly.
    """
    all_files = os.listdir(data_folder)
    data_frames = []

    for file in all_files:
        file_path = os.path.join(data_folder, file)
        df = pd.read_csv(file_path)

        # Reshape wide format to long format
        df_long = df.melt(id_vars=['channel'], var_name='timestamp', value_name='count')

        # Convert timestamp format
        df_long['timestamp'] = df_long['timestamp'].str.replace(':AM', ' AM').str.replace(':PM', ' PM', regex=False)
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format="%m/%d/%y %I:%M:%S %p", errors='coerce')

        # Treat empty or null values as 0
        df_long['count'] = df_long['count'].fillna(0)

        data_frames.append(df_long)

    return pd.concat(data_frames, ignore_index=True) if data_frames else pd.DataFrame()

def train_or_update_model(channel, data_type, data, model_type):
    """
    Train or update the existing model with new data.
    """
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    
    if model_type == "ARIMA":
        model = ARIMA(data['count'].fillna(0), order=(5,1,0)).fit()
    else:
        df = data.rename(columns={'timestamp': 'ds', 'count': 'y'})
        model = Prophet()
        model.fit(df)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return model

def forecast(channel, data_type, model_type, periods=24):
    """
    Load the trained model and forecast future values.
    """
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    if not os.path.exists(model_path):
        return None, None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    future_dates = pd.date_range(start=pd.Timestamp.now(), periods=periods, freq='H')

    if model_type == "ARIMA":
        forecast_values = model.forecast(periods)
    else:
        future_df = pd.DataFrame({'ds': future_dates})
        forecast_values = model.predict(future_df)['yhat']

    return future_dates, forecast_values

def main():
    st.title("Logins & Payments Forecasting")

    # Load data
    logins_df = load_data(data_folder_logins)
    payments_df = load_data(data_folder_payments)

    # Get unique channels
    channels = set(logins_df['channel'].unique()).union(set(payments_df['channel'].unique()))

    selected_channel = st.selectbox("Select Channel", list(channels))
    options = st.multiselect("Select Data Type", ["Logins", "Payments", "Both"], default="Both")
    model_type = st.selectbox("Select Model", ["ARIMA", "Prophet"])

    if st.button("Train/Update Model"):
        if "Logins" in options or "Both" in options:
            train_or_update_model(selected_channel, "logins", logins_df[logins_df['channel'] == selected_channel], model_type)
        if "Payments" in options or "Both" in options:
            train_or_update_model(selected_channel, "payments", payments_df[payments_df['channel'] == selected_channel], model_type)
        st.success("Model Updated Successfully!")

    if st.button("Forecast"):
        logins_forecast, payments_forecast = None, None

        if "Logins" in options or "Both" in options:
            logins_dates, logins_values = forecast(selected_channel, "logins", model_type)
            if logins_dates is not None:
                logins_forecast = pd.DataFrame({'Date': logins_dates, 'Logins': logins_values})
                st.line_chart(logins_forecast.set_index('Date'))

        if "Payments" in options or "Both" in options:
            payments_dates, payments_values = forecast(selected_channel, "payments", model_type)
            if payments_dates is not None:
                payments_forecast = pd.DataFrame({'Date': payments_dates, 'Payments': payments_values})
                st.line_chart(payments_forecast.set_index('Date'))

        if logins_forecast is not None and payments_forecast is not None and "Both" in options:
            combined_forecast = pd.merge(logins_forecast, payments_forecast, on='Date', how='outer')
            st.line_chart(combined_forecast.set_index('Date'))

if __name__ == "__main__":
    main()
