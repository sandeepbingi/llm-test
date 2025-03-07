import os
import pandas as pd
import pickle
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet

data_folder = "data"
model_folder = "models"
processed_files_log = "processed_files.txt"
os.makedirs(model_folder, exist_ok=True)

def load_data():
    files = os.listdir(data_folder)
    logins_files = [f for f in files if "Login" in f]
    payments_files = [f for f in files if "Payments" in f]
    
    logins_df = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in logins_files], ignore_index=True)
    payments_df = pd.concat([pd.read_csv(os.path.join(data_folder, f)) for f in payments_files], ignore_index=True)
    
    logins_df['timestamp'] = pd.to_datetime(logins_df['timestamp'], errors='coerce')
    payments_df['timestamp'] = pd.to_datetime(payments_df['timestamp'], errors='coerce')
    
    return logins_df, payments_df

def get_processed_files():
    if not os.path.exists(processed_files_log):
        return set()
    with open(processed_files_log, "r") as f:
        return set(f.read().splitlines())

def update_processed_files(new_files):
    with open(processed_files_log, "a") as f:
        for file in new_files:
            f.write(file + "\n")

def train_or_update_model(channel, data_type, data, model_type):
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    
    if model_type == "ARIMA":
        model = ARIMA(data['count'], order=(5,1,0)).fit()
    else:
        df = data.rename(columns={'timestamp': 'ds', 'count': 'y'})
        model = Prophet()
        model.fit(df)
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return model

def forecast(channel, data_type, model_type, periods=24):
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    if not os.path.exists(model_path):
        return None
    
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
    logins_df, payments_df = load_data()
    channels = set(logins_df['channel'].unique()).union(set(payments_df['channel'].unique()))
    
    selected_channel = st.selectbox("Select Channel", list(channels))
    options = st.multiselect("Select Data Type", ["Logins", "Payments", "Both"], default="Both")
    model_type = st.selectbox("Select Model", ["ARIMA", "Prophet"])
    
    processed_files = get_processed_files()
    new_files = set(os.listdir(data_folder)) - processed_files
    
    if new_files:
        st.write("New data detected, updating model...")
        if "Logins" in options or "Both" in options:
            train_or_update_model(selected_channel, "logins", logins_df[logins_df['channel'] == selected_channel], model_type)
        if "Payments" in options or "Both" in options:
            train_or_update_model(selected_channel, "payments", payments_df[payments_df['channel'] == selected_channel], model_type)
        update_processed_files(new_files)
        st.success("Model Updated Successfully!")
    
    if st.button("Forecast"):
        if "Logins" in options or "Both" in options:
            dates, values = forecast(selected_channel, "logins", model_type)
            st.line_chart(pd.DataFrame({'Date': dates, 'Logins': values}))
        if "Payments" in options or "Both" in options:
            dates, values = forecast(selected_channel, "payments", model_type)
            st.line_chart(pd.DataFrame({'Date': dates, 'Payments': values}))

if __name__ == "__main__":
    main()
