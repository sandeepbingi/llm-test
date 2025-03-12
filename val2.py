import os
import pickle
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Define Paths
MODEL_FOLDER = "saved_models_v2"
DATA_FOLDER_LOGINS = "C:/Users/Desktop/Logins_Payments_LLM/Login_Reports"
DATA_FOLDER_PAYMENTS = "C:/Users/Desktop/Logins_Payments_LLM/Payment_Reports"

# Load test data from separate folders
def load_test_data(folder, data_type):
    files = os.listdir(folder)
    data_list = []

    for file in files:
        file_path = os.path.join(folder, file)
        df = pd.read_csv(file_path)

        df_long = df.melt(id_vars=['Channel'], var_name='timestamp', value_name='count')
        df_long['timestamp'] = df_long['timestamp'].str.replace(':AM', ' AM').str.replace(':PM', ' PM')
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format="%m/%d/%y %I:%M %p", errors='coerce')

        df_long['Type'] = data_type  # Tag data type (logins/payments)
        data_list.append(df_long)

    if data_list:
        return pd.concat(data_list, ignore_index=True)
    return pd.DataFrame()

# Load trained model
def load_model(channel, data_type):
    model_path = os.path.join(MODEL_FOLDER, f"{channel}_{data_type}_Prophet.pkl")
    
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        return model_data["model"]
    
    return None

# Forecast using Prophet model
def forecast(model, future_dates):
    future_df = pd.DataFrame({"ds": future_dates})
    forecast = model.predict(future_df)
    return forecast[["ds", "yhat"]]

# Evaluate model
def evaluate_model(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse

# Streamlit UI
def main():
    st.title("Model Validation - Logins & Payments")

    # Load test data
    logins_test_data = load_test_data(DATA_FOLDER_LOGINS, "logins")
    payments_test_data = load_test_data(DATA_FOLDER_PAYMENTS, "payments")

    if logins_test_data.empty and payments_test_data.empty:
        st.error("No test data available. Please add test datasets to the respective folders.")
        return

    # Combine test data
    test_data = pd.concat([logins_test_data, payments_test_data], ignore_index=True)
    channels = test_data["Channel"].unique().tolist()

    # UI Inputs
    selected_channel = st.selectbox("Select Channel", sorted(channels))
    data_type = st.radio("Select Type", ["Logins", "Payments", "Both"], horizontal=True)

    if st.button("Validate Model"):
        results = []

        for dtype in ["logins", "payments"]:
            if data_type in ["Both", dtype.capitalize()]:  
                model = load_model(selected_channel, dtype)
                test_subset = test_data[(test_data["Channel"] == selected_channel) & (test_data["Type"] == dtype)]

                if test_subset.empty:
                    st.warning(f"No test data available for {dtype}.")
                    continue

                test_subset = test_subset.rename(columns={"timestamp": "ds", "count": "y"})

                if model:
                    forecasted = forecast(model, test_subset["ds"])
                    mae, rmse = evaluate_model(test_subset["y"], forecasted["yhat"])
                    results.append({"Type": dtype.capitalize(), "MAE": mae, "RMSE": rmse})

                    # Plot Actual vs Predicted
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.plot(test_subset["ds"], test_subset["y"], label="Actual", marker="o")
                    ax.plot(forecasted["ds"], forecasted["yhat"], label="Predicted", linestyle="dashed")
                    ax.set_title(f"{dtype.capitalize()} - Actual vs Predicted")
                    ax.set_xlabel("Time")
                    ax.set_ylabel("Count")
                    ax.legend()
                    st.pyplot(fig)
                else:
                    st.warning(f"No trained model found for {dtype}.")

        if results:
            results_df = pd.DataFrame(results)
            st.subheader("Validation Metrics")
            st.dataframe(results_df)

if __name__ == "__main__":
    main()
