import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet

# Define data and model folders
data_folder_logins = "data/logins"
data_folder_payments = "data/payments"
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

# Load data from CSV files
def load_data():
    logins_files = [os.path.join(data_folder_logins, f) for f in os.listdir(data_folder_logins) if f.endswith('.csv')]
    payments_files = [os.path.join(data_folder_payments, f) for f in os.listdir(data_folder_payments) if f.endswith('.csv')]

    logins_df = pd.concat([process_csv(f) for f in logins_files], ignore_index=True) if logins_files else pd.DataFrame()
    payments_df = pd.concat([process_csv(f) for f in payments_files], ignore_index=True) if payments_files else pd.DataFrame()

    return logins_df, payments_df

# Process CSV files
def process_csv(file_path):
    df = pd.read_csv(file_path)
    df_long = df.melt(id_vars=['channel'], var_name='timestamp', value_name='count')
    df_long['timestamp'] = df_long['timestamp'].str.replace(':AM', ' AM').str.replace(':PM', ' PM', regex=False)
    df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format="%m/%d/%y %I:%M:%S %p", errors='coerce')
    df_long['count'] = df_long['count'].fillna(0)  # Treat empty/null as 0
    return df_long

# Train or update model
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
    channels = sorted(set(logins_df['channel'].unique()).union(set(payments_df['channel'].unique())))

    selected_channel = st.selectbox("Select Channel", channels)
    data_type = st.radio("Select Data Type", ["Logins", "Payments", "Both"])
    model_type = st.radio("Select Model", ["ARIMA", "Prophet"])

    if st.button("Forecast"):
        forecast_results = []
        if data_type in ["Logins", "Both"]:
            if not logins_df.empty:
                channel_data = logins_df[logins_df['channel'] == selected_channel]
                train_or_update_model(selected_channel, "logins", channel_data, model_type)
                dates, values = forecast(selected_channel, "logins", model_type)
                if dates is not None:
                    forecast_results.append(pd.DataFrame({'Datetime': dates, 'Forecasted Count': values, 'Channel': f"{selected_channel} (Logins)"}))

        if data_type in ["Payments", "Both"]:
            if not payments_df.empty:
                channel_data = payments_df[payments_df['channel'] == selected_channel]
                train_or_update_model(selected_channel, "payments", channel_data, model_type)
                dates, values = forecast(selected_channel, "payments", model_type)
                if dates is not None:
                    forecast_results.append(pd.DataFrame({'Datetime': dates, 'Forecasted Count': values, 'Channel': f"{selected_channel} (Payments)"}))

        if forecast_results:
            forecast_df = pd.concat(forecast_results, ignore_index=True)
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

if __name__ == "__main__":
    main()
