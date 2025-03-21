import os
import pickle
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
import matplotlib.pyplot as plt
import seaborn as sns

# Define directories
DATA_FOLDER_LOGINS = "data/logins"
DATA_FOLDER_PAYMENTS = "data/payments"
MODEL_FOLDER = "models"
PROCESSED_FILES_TRACKER = "processed_files.txt"

# Ensure model directory exists
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Static channel list
CHANNELS = ["ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "QWE", "XYZ"]  # Add your actual channel names

# Function to load processed files
def load_processed_files():
    if os.path.exists(PROCESSED_FILES_TRACKER):
        with open(PROCESSED_FILES_TRACKER, "r") as f:
            return set(f.read().splitlines())
    return set()

# Function to save processed files
def save_processed_files(processed_files):
    with open(PROCESSED_FILES_TRACKER, "w") as f:
        f.write("\n".join(processed_files))

# Function to load new data
def load_data():
    processed_files = load_processed_files()
    
    logins_files = [f for f in os.listdir(DATA_FOLDER_LOGINS) if f not in processed_files]
    payments_files = [f for f in os.listdir(DATA_FOLDER_PAYMENTS) if f not in processed_files]
    
    logins_df = pd.concat([pd.read_csv(os.path.join(DATA_FOLDER_LOGINS, f)) for f in logins_files], ignore_index=True) if logins_files else pd.DataFrame()
    payments_df = pd.concat([pd.read_csv(os.path.join(DATA_FOLDER_PAYMENTS, f)) for f in payments_files], ignore_index=True) if payments_files else pd.DataFrame()

    if not logins_df.empty:
        logins_df = logins_df.melt(id_vars=['channel'], var_name='timestamp', value_name='count')
        logins_df['timestamp'] = pd.to_datetime(logins_df['timestamp'], errors='coerce')

    if not payments_df.empty:
        payments_df = payments_df.melt(id_vars=['channel'], var_name='timestamp', value_name='count')
        payments_df['timestamp'] = pd.to_datetime(payments_df['timestamp'], errors='coerce')

    return logins_df, payments_df, logins_files, payments_files, processed_files

# Function to train or update model incrementally
def train_or_update_model(channel, data_type, data, model_type):
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{data_type}_{model_type}.pkl")
    
    # Load existing model if available
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            saved_model = pickle.load(f)
        existing_data = saved_model['data']
        data = pd.concat([existing_data, data]).drop_duplicates().sort_values('timestamp')

    # Train or update model
    if model_type == "ARIMA":
        model_instance = ARIMA(data['count'].fillna(0), order=(5,1,0)).fit()
    else:  # Prophet Model
        df = data.rename(columns={'timestamp': 'ds', 'count': 'y'})
        model_instance = Prophet()
        model_instance.fit(df)
    
    # Save the updated model
    with open(model_path, "wb") as f:
        pickle.dump({'model': model_instance, 'data': data}, f)

# Function to forecast future values
def forecast(channel, data_type, model_type, future_date):
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{data_type}_{model_type}.pkl")
    if not os.path.exists(model_path):
        return None, None

    with open(model_path, "rb") as f:
        saved_model = pickle.load(f)
    
    model_instance = saved_model['model']

    future_dates = pd.date_range(start=future_date, periods=24, freq='H')
    
    if model_type == "ARIMA":
        forecast_values = model_instance.forecast(24)
    else:  # Prophet
        future_df = pd.DataFrame({'ds': future_dates})
        forecast_values = model_instance.predict(future_df)['yhat']

    return future_dates, forecast_values

# Main Streamlit UI
def main():
    st.title("Logins & Payments Forecasting")

    logins_df, payments_df, logins_files, payments_files, processed_files = load_data()

    selected_channel = st.selectbox("Select Channel", sorted(CHANNELS), index=0, key="channel_dropdown")
    data_type = st.radio("Select Data Type", ["Logins", "Payments", "Both"])
    model_type = st.radio("Select Model", ["ARIMA", "Prophet"])
    future_date = st.date_input("Select Future Date for Forecast")

    if st.button("Forecast"):
        if data_type in ["Logins", "Both"] and not logins_df.empty:
            train_or_update_model(selected_channel, "logins", logins_df[logins_df['channel'] == selected_channel], model_type)
        
        if data_type in ["Payments", "Both"] and not payments_df.empty:
            train_or_update_model(selected_channel, "payments", payments_df[payments_df['channel'] == selected_channel], model_type)

        # Update processed files tracker only after successful training
        processed_files.update(logins_files + payments_files)
        save_processed_files(processed_files)

        # Forecasting
        forecast_data = []
        
        if data_type in ["Logins", "Both"]:
            dates, values = forecast(selected_channel, "logins", model_type, future_date)
            if dates is not None:
                forecast_data.extend(zip(dates, values, ["Logins"] * len(dates)))

        if data_type in ["Payments", "Both"]:
            dates, values = forecast(selected_channel, "payments", model_type, future_date)
            if dates is not None:
                forecast_data.extend(zip(dates, values, ["Payments"] * len(dates)))

        if forecast_data:
            forecast_df = pd.DataFrame(forecast_data, columns=["Datetime", "Forecasted Count", "Channel"])
            st.subheader("Forecast Results")
            st.dataframe(forecast_df)

            # Download button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast as CSV", csv, "forecast_results.csv", "text/csv")

            # Plotting
            st.subheader("Trend Visualization")
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=forecast_df, x='Datetime', y='Forecasted Count', hue='Channel', marker='o')
            plt.xlabel("Datetime")
            plt.ylabel("Count")
            plt.title("Forecasted Count per Channel")
            plt.xticks(rotation=45)
            st.pyplot(plt)
        else:
            st.warning("No data available to display the forecast.")

if __name__ == "__main__":
    main()
