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
os.makedirs(model_folder, exist_ok=True)

# Track processed files
def get_processed_files():
    if os.path.exists(processed_files_log):
        with open(processed_files_log, "r") as f:
            return set(f.read().splitlines())
    return set()

def update_processed_files(new_files):
    with open(processed_files_log, "a") as f:
        for file in new_files:
            f.write(file + "\n")

# Load data from CSV files
def load_data():
    processed_files = get_processed_files()
    logins_files = [f for f in os.listdir(data_folder_logins) if f not in processed_files]
    payments_files = [f for f in os.listdir(data_folder_payments) if f not in processed_files]

    logins_df = pd.concat([pd.read_csv(os.path.join(data_folder_logins, f)) for f in logins_files], ignore_index=True) if logins_files else pd.DataFrame()
    payments_df = pd.concat([pd.read_csv(os.path.join(data_folder_payments, f)) for f in payments_files], ignore_index=True) if payments_files else pd.DataFrame()

    for df in [logins_df, payments_df]:
        if not df.empty:
            df['timestamp'] = df['timestamp'].str.replace(':AM', ' AM').str.replace(':PM', ' PM', regex=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%y %I:%M:%S %p", errors='coerce')

    update_processed_files(logins_files + payments_files)
    return logins_df, payments_df

# Train or update model per channel and type
def train_or_update_model(channel, data_type, data, model_type):
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    
    if data.empty:
        return None

    data = data[data['channel'] == channel].sort_values('timestamp')
    
    if model_type == "ARIMA":
        model = ARIMA(data['count'].fillna(0), order=(5,1,0)).fit()
    else:
        df = data.rename(columns={'timestamp': 'ds', 'count': 'y'})
        model = Prophet()
        model.fit(df)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    return model

# Forecast function
def forecast(channel, data_type, model_type, periods=24):
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

# Streamlit UI
def main():
    st.title("Logins & Payments Forecasting")
    logins_df, payments_df = load_data()
    
    channels = set(logins_df['channel'].unique()).union(set(payments_df['channel'].unique()))
    
    selected_channel = st.selectbox("Select Channel", list(channels))
    options = st.multiselect("Select Data Type", ["Logins", "Payments", "Both"], default="Both")
    model_type = st.selectbox("Select Model", ["ARIMA", "Prophet"])

    if st.button("Train/Update Model"):
        if "Logins" in options or "Both" in options:
            train_or_update_model(selected_channel, "logins", logins_df, model_type)
        if "Payments" in options or "Both" in options:
            train_or_update_model(selected_channel, "payments", payments_df, model_type)
        st.success("Model Updated Successfully!")

    if st.button("Forecast"):
        logins_dates, logins_values = (None, None)
        payments_dates, payments_values = (None, None)

        if "Logins" in options or "Both" in options:
            logins_dates, logins_values = forecast(selected_channel, "logins", model_type)
        if "Payments" in options or "Both" in options:
            payments_dates, payments_values = forecast(selected_channel, "payments", model_type)

        if logins_dates is not None and logins_values is not None:
            st.line_chart(pd.DataFrame({'Date': logins_dates, 'Logins': logins_values}))
        
        if payments_dates is not None and payments_values is not None:
            st.line_chart(pd.DataFrame({'Date': payments_dates, 'Payments': payments_values}))

        if logins_values is not None and payments_values is not None:
            combined_df = pd.DataFrame({'Date': logins_dates, 'Logins': logins_values, 'Payments': payments_values})
            st.line_chart(combined_df.set_index('Date'))

if __name__ == "__main__":
    main()
