import os
import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet

# Folders for data and models
data_folder_logins = "data/logins"
data_folder_payments = "data/payments"
model_folder = "models"
processed_files_log = "processed_files.txt"

# Static channel list
CHANNELS = ["QWE", "ABC", "XYZ", "LMN"]  

# Create necessary directories
os.makedirs(model_folder, exist_ok=True)

# Load processed files tracker
def load_processed_files():
    if os.path.exists(processed_files_log):
        with open(processed_files_log, "r") as f:
            return set(f.read().splitlines())
    return set()

# Save processed files tracker
def save_processed_files(processed_files):
    with open(processed_files_log, "w") as f:
        f.write("\n".join(processed_files))

# Load new data while avoiding duplicate processing
def load_new_data():
    processed_files = load_processed_files()
    
    logins_files = [f for f in os.listdir(data_folder_logins) if f not in processed_files]
    payments_files = [f for f in os.listdir(data_folder_payments) if f not in processed_files]
    
    logins_df = pd.concat([pd.read_csv(os.path.join(data_folder_logins, f)) for f in logins_files], ignore_index=True) if logins_files else pd.DataFrame()
    payments_df = pd.concat([pd.read_csv(os.path.join(data_folder_payments, f)) for f in payments_files], ignore_index=True) if payments_files else pd.DataFrame()
    
    # Process timestamps
    for df in [logins_df, payments_df]:
        if not df.empty:
            df['timestamp'] = df['timestamp'].astype(str).str.replace(':AM', ' AM').str.replace(':PM', ' PM', regex=False)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format="%m/%d/%y %I:%M:%S %p", errors='coerce')

    # Update processed files tracker
    processed_files.update(logins_files + payments_files)
    save_processed_files(processed_files)

    return logins_df, payments_df

# Train or update model incrementally
def train_or_update_model(channel, data_type, data, model_type):
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    
    # Load existing model if available
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
            existing_data = model_data['data']
        
        # Merge new data, remove duplicates, and sort
        data = pd.concat([existing_data, data]).drop_duplicates().sort_values('timestamp')

    # Train/update the model
    if model_type == "ARIMA":
        model_instance = ARIMA(data['count'].fillna(0), order=(5,1,0)).fit()
    else:
        df = data.rename(columns={'timestamp': 'ds', 'count': 'y'})
        model_instance = Prophet()
        model_instance.fit(df)

    # Save updated model and data
    with open(model_path, "wb") as f:
        pickle.dump({'model': model_instance, 'data': data}, f)

    return model_instance

# Forecast future data
def forecast(channel, data_type, model_type, future_date):
    model_path = os.path.join(model_folder, f"{channel}_{data_type}_{model_type}.pkl")
    
    if not os.path.exists(model_path):
        st.warning(f"No trained model found for {channel} - {data_type} - {model_type}. Train the model first.")
        return None

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
        model = model_data['model']

    # Generate future timestamps
    future_dates = pd.date_range(start=future_date, periods=24, freq='H')

    if model_type == "ARIMA":
        forecast_values = model.forecast(24)
    else:
        future_df = pd.DataFrame({'ds': future_dates})
        forecast_values = model.predict(future_df)['yhat']

    # Create Forecast DataFrame
    forecast_df = pd.DataFrame({
        'Datetime': future_dates,
        'Forecasted Count': forecast_values,
        'Channel': channel,
        'Data Type': data_type
    })

    # Display Forecast Table
    st.subheader("Forecast Results")
    st.dataframe(forecast_df)

    # Download Button
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Forecast as CSV", csv, "forecast_results.csv", "text/csv")

    return forecast_df

# Main function to run Streamlit UI
def main():
    st.title("Logins & Payments Forecasting")

    # Load new data
    logins_df, payments_df = load_new_data()

    # Select Channel (Static List)
    selected_channel = st.selectbox("Select Channel", sorted(CHANNELS), index=0, key="channel_select", help="Choose a channel to forecast.")

    # Select Data Type (Radio Button)
    data_type = st.radio("Select Data Type", ["Logins", "Payments", "Both"], key="data_type_radio")

    # Select Model Type (Radio Button)
    model_type = st.radio("Select Model", ["ARIMA", "Prophet"], key="model_type_radio")

    # Input Future Date
    future_date = st.date_input("Select Future Date", help="Pick a date for forecasting.")

    if st.button("Forecast"):
        # Train/update models automatically
        if "Logins" in data_type or "Both" in data_type:
            if not logins_df.empty:
                train_or_update_model(selected_channel, "logins", logins_df[logins_df['channel'] == selected_channel], model_type)
        
        if "Payments" in data_type or "Both" in data_type:
            if not payments_df.empty:
                train_or_update_model(selected_channel, "payments", payments_df[payments_df['channel'] == selected_channel], model_type)

        # Forecast results
        forecast_results = []
        if "Logins" in data_type or "Both" in data_type:
            forecast_results.append(forecast(selected_channel, "logins", model_type, future_date))

        if "Payments" in data_type or "Both" in data_type:
            forecast_results.append(forecast(selected_channel, "payments", model_type, future_date))

        # Plot results if data exists
        if any(forecast_results):
            st.subheader("Trend Visualization")
            plt.figure(figsize=(12, 6))
            for result in forecast_results:
                if result is not None:
                    sns.lineplot(data=result, x='Datetime', y='Forecasted Count', hue='Data Type', marker='o')

            plt.xlabel("Datetime")
            plt.ylabel("Count")
            plt.title(f"Forecasted Trend for {selected_channel}")
            plt.xticks(rotation=45)
            st.pyplot(plt)

            # Download Plot
            plot_path = f"{selected_channel}_forecast_plot.png"
            plt.savefig(plot_path)
            with open(plot_path, "rb") as file:
                st.download_button("Download Plot", file, file_name=plot_path, mime="image/png")
        else:
            st.warning("No data available for plotting.")

if __name__ == "__main__":
    main()
