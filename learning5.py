import os
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet

# Define folder paths
data_folder_logins = "data/logins"
data_folder_payments = "data/payments"
model_folder = "models"
processed_files_log = "processed_files.txt"

# Ensure model directory exists
os.makedirs(model_folder, exist_ok=True)

# Load processed files list
def get_processed_files():
    if not os.path.exists(processed_files_log):
        return set()
    with open(processed_files_log, "r") as f:
        return set(f.read().splitlines())

# Save processed files list
def update_processed_files(files):
    with open(processed_files_log, "a") as f:
        for file in files:
            f.write(file + "\n")

# Load and process new data files
def load_new_data():
    processed_files = get_processed_files()
    
    # Identify new files
    logins_files = [f for f in os.listdir(data_folder_logins) if f not in processed_files]
    payments_files = [f for f in os.listdir(data_folder_payments) if f not in processed_files]

    # Load new data
    def process_files(files, folder):
        df_list = []
        for file in files:
            file_path = os.path.join(folder, file)
            df = pd.read_csv(file_path)
            df_long = df.melt(id_vars=['channel'], var_name='timestamp', value_name='count')
            df_long['timestamp'] = df_long['timestamp'].str.replace(':AM', ' AM').str.replace(':PM', ' PM', regex=False)
            df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format="%m/%d/%y %I:%M:%S %p", errors='coerce')
            df_long['count'] = df_long['count'].fillna(0)
            df_list.append(df_long)
        
        return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

    logins_df = process_files(logins_files, data_folder_logins)
    payments_df = process_files(payments_files, data_folder_payments)

    # Update processed file tracking
    update_processed_files(logins_files + payments_files)
    
    return logins_df, payments_df

# Train or update model incrementally
def train_or_update_model(channel, data_type, data, model_type):
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")

    # Load existing model if available
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        existing_data = model['data']
        data = pd.concat([existing_data, data]).drop_duplicates().sort_values('timestamp')

    # Train new or update existing model
    if model_type == "ARIMA":
        model_instance = ARIMA(data['count'], order=(5,1,0)).fit()
    else:
        df = data.rename(columns={'timestamp': 'ds', 'count': 'y'})
        model_instance = Prophet()
        model_instance.fit(df)

    # Save model with updated data
    with open(model_path, "wb") as f:
        pickle.dump({'model': model_instance, 'data': data}, f)

# Forecast future values
def forecast(channel, data_type, model_type, periods=24):
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    if not os.path.exists(model_path):
        return None, None
    
    with open(model_path, "rb") as f:
        saved_model = pickle.load(f)
    
    model = saved_model['model']
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

    # Load new data files
    logins_df, payments_df = load_new_data()

    # Get all unique channels
    channels = set(logins_df['channel'].unique()).union(set(payments_df['channel'].unique()))
    sorted_channels = sorted(channels)

    selected_channel = st.selectbox("Select Channel", sorted_channels)
    data_type = st.radio("Select Data Type", ["Logins", "Payments", "Both"])
    model_type = st.radio("Select Model", ["ARIMA", "Prophet"])

    # Forecasting trigger
    if st.button("Forecast"):
        results = []
        if data_type in ["Logins", "Both"]:
            train_or_update_model(selected_channel, "logins", logins_df[logins_df['channel'] == selected_channel], model_type)
            dates, values = forecast(selected_channel, "logins", model_type)
            if dates is not None:
                results.append(pd.DataFrame({'Datetime': dates, 'Forecasted Count': values, 'Channel': selected_channel, 'Type': 'Logins'}))

        if data_type in ["Payments", "Both"]:
            train_or_update_model(selected_channel, "payments", payments_df[payments_df['channel'] == selected_channel], model_type)
            dates, values = forecast(selected_channel, "payments", model_type)
            if dates is not None:
                results.append(pd.DataFrame({'Datetime': dates, 'Forecasted Count': values, 'Channel': selected_channel, 'Type': 'Payments'}))

        # Combine results
        if results:
            forecast_df = pd.concat(results, ignore_index=True)

            # Display results
            st.subheader("Forecast Results")
            st.dataframe(forecast_df)

            # Download Button
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Forecast as CSV", csv, "forecast_results.csv", "text/csv")

            # Plot Forecast
            st.subheader("Trend Visualization")
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=forecast_df, x='Datetime', y='Forecasted Count', hue='Type', marker='o')
            plt.xlabel("Datetime")
            plt.ylabel("Count")
            plt.title(f"Forecasted Count for {selected_channel}")
            plt.xticks(rotation=45)
            st.pyplot(plt)

if __name__ == "__main__":
    main()
