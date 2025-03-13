import os
import pickle
import io
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from prophet import Prophet

# Define Paths
DATA_FOLDER_LOGINS = "C:/Users/Desktop/Logins_Payments_LLM/Login_Reports"
DATA_FOLDER_PAYMENTS = "C:/Users/Desktop/Logins_Payments_LLM/Payment_Reports"
MODEL_FOLDER = "saved_models_v2"
TRACKED_FILES_PATH = "processed_files_v2.txt"

# Ensure directories exist
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Define Holidays for Prophet
holidays = pd.DataFrame({
    'holiday': [
        'new_year', 'mlk_day', 'presidents_day', 'memorial_day', 'independence_day', 'labor_day',
        'thanksgiving', 'christmas'
    ] * 2,  # Repeat for 2 years (2024-2025)
    'ds': pd.to_datetime([
        # 2024
        '2024-01-01', '2024-01-15', '2024-02-19', '2024-05-27',
        '2024-07-04', '2024-09-02', '2024-11-28', '2024-12-25',
        # 2025
        '2025-01-01', '2025-01-20', '2025-02-17', '2025-05-26',
        '2025-07-04', '2025-09-01', '2025-11-27', '2025-12-25'
    ]),
    'lower_window': -1,  # Effect starts 1 day before
    'upper_window': 2    # Effect lasts up to 2 days after
})

# Static channel list
CHANNELS_LIST = ['QWE', 'ASD']

# Load processed files tracking
def load_processed_files():
    if os.path.exists(TRACKED_FILES_PATH):
        with open(TRACKED_FILES_PATH, "rb") as f:
            return pickle.load(f)
    return {"logins": set(), "payments": set()}

# Save processed files tracking
def save_processed_files(processed_files):
    with open(TRACKED_FILES_PATH, "wb") as f:
        pickle.dump(processed_files, f)

# Process new data files
def process_folder(folder, data_type, processed_files):
    files = os.listdir(folder)
    new_files = [f for f in files if f not in processed_files.get(data_type, set())]

    if not new_files:
        return pd.DataFrame(), new_files  # No new data

    df_list = []
    for file in new_files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)
        df_long = df.melt(id_vars=['Channel'], var_name='timestamp', value_name='count')
        df_long['timestamp'] = df_long['timestamp'].str.replace(':AM', ' AM').str.replace(':PM', ' PM')
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format="%m/%d/%y %I:%M %p", errors='coerce')
        df_list.append(df_long)

    return (pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()), new_files

# Load new data
def load_data():
    processed_files = load_processed_files()

    logins_df, new_login_files = process_folder(DATA_FOLDER_LOGINS, "logins", processed_files)
    payments_df, new_payment_files = process_folder(DATA_FOLDER_PAYMENTS, "payments", processed_files)

    channels = pd.concat([logins_df, payments_df])['Channel'].unique()

    for channel in channels:
        if not logins_df.empty:
            train_or_update_model(channel, "logins", logins_df[logins_df['Channel'] == channel], "Prophet")
        if not payments_df.empty:
            train_or_update_model(channel, "payments", payments_df[payments_df['Channel'] == channel], "Prophet")

    # Update processed files only if new files were found
    if new_login_files:
        processed_files["logins"].update(new_login_files)
    if new_payment_files:
        processed_files["payments"].update(new_payment_files)

    save_processed_files(processed_files)
    return logins_df, payments_df

# Train or update model
def train_or_update_model(channel, data_type, new_data, model_type):
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{data_type}_{model_type}.pkl")

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        existing_data = model_data['data']

        # Merge only truly new data
        combined_data = pd.concat([existing_data, new_data]).drop_duplicates().sort_values('timestamp')

        if combined_data.equals(existing_data):
            print(f"No new data for {channel} ({data_type}). Skipping training.")
            return
    else:
        combined_data = new_data

    if combined_data.empty:
        print(f"No historical data for {channel} ({data_type}). Skipping training.")
        return

    df = combined_data.rename(columns={'timestamp': 'ds', 'count': 'y'})

    model_instance = Prophet(
        holidays=holidays,
        changepoint_prior_scale=0.8 if data_type == "logins" else 0.5,
        seasonality_mode="multiplicative",
        seasonality_prior_scale=15,
        holidays_prior_scale=15
    )
    model_instance.add_seasonality(name='weekly', period=7, fourier_order=10)
    model_instance.fit(df)

    with open(model_path, "wb") as f:
        pickle.dump({'model': model_instance, 'data': combined_data}, f)

    print(f"Model updated for {channel} ({data_type})")

# Forecast function
def forecast(channel, data_type, model_type, future_date):
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{data_type}_{model_type}.pkl")

    if not os.path.exists(model_path):
        return None, None

    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    model = model_data['model']

    future_dates = pd.date_range(start=future_date, periods=24, freq='H')
    future_df = pd.DataFrame({'ds': future_dates})
    forecast_values = model.predict(future_df)['yhat']
    forecast_values = [max(0, round(value)) for value in forecast_values]

    return future_dates, forecast_values

# Streamlit UI
def main():
    st.markdown("## 📊 Logins & Payments Forecasting")

    selected_channel = st.selectbox("Select Channel", sorted(CHANNELS_LIST))
    data_type = st.radio("Select Type", ["Logins", "Payments", "Both"], horizontal=True)
    future_date = st.date_input("Select Future Date for Forecast")

    if st.button("Forecast"):
        with st.spinner("Processing... Please wait..."):
            logins_df, payments_df = load_data()

            forecast_results = []
            if data_type in ["Logins", "Both"]:
                dates, values = forecast(selected_channel, "logins", "Prophet", future_date)
                if dates is not None and values is not None:
                    forecast_results.append(pd.DataFrame({
                        'Channel': selected_channel, 'Type': 'Logins',
                        'Datetime': dates, 'Forecasted Count': values
                    }))

            if data_type in ["Payments", "Both"]:
                dates, values = forecast(selected_channel, "payments", "Prophet", future_date)
                if dates is not None and values is not None:
                    forecast_results.append(pd.DataFrame({
                        'Channel': selected_channel, 'Type': 'Payments',
                        'Datetime': dates, 'Forecasted Count': values
                    }))

            if forecast_results:
                forecast_df = pd.concat(forecast_results, ignore_index=True)
                st.dataframe(forecast_df)

                fig, ax = plt.subplots(figsize=(12, 6))
                sns.lineplot(data=forecast_df, x='Datetime', y='Forecasted Count', hue='Type', marker='o', ax=ax)
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                plt.xticks(rotation=45)
                st.pyplot(fig)

if __name__ == "__main__":
    main()
