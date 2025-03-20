import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from glob import glob

# Set Streamlit page configuration
st.set_page_config(page_title="Logins & Payments Dashboard", layout="wide")

# Define folder paths
LOGINS_FOLDER = "C:/Users/Desktop/Logins_Payments_LLM/Login_Reports/"
PAYMENTS_FOLDER = "C:/Users/Desktop/Logins_Payments_LLM/Payment_Reports/"

# Function to load and label data
def load_data(folder, type_label):
    files = glob(os.path.join(folder, "*.csv"))  # Get all CSV files
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp
        df['type'] = type_label  # Label as logins or payments
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()

# Load data from both folders
df_logins = load_data(LOGINS_FOLDER, "logins")
df_payments = load_data(PAYMENTS_FOLDER, "payments")

# Combine datasets
df = pd.concat([df_logins, df_payments], ignore_index=True)

# Check if data is available
if df.empty:
    st.error("No data found! Please check your CSV files.")
    st.stop()

# Extract date from timestamp
df['date'] = df['timestamp'].dt.date

# Aggregate total count per day per channel per type
daily_counts = df.groupby(['channel', 'date', 'type'])['count'].sum().reset_index()

# Sidebar - Select Channel
channels = sorted(df['channel'].unique())
selected_channel = st.sidebar.selectbox("Select Channel", channels)

# Filter data for selected channel
df_channel = daily_counts[daily_counts['channel'] == selected_channel]

# 📌 Interactive Time-Series Plot for Selected Channel
fig = go.Figure()
for data_type in df_channel['type'].unique():
    df_type = df_channel[df_channel['type'] == data_type]
    fig.add_trace(go.Scatter(
        x=df_type['date'], 
        y=df_type['count'], 
        mode='lines+markers', 
        name=data_type.capitalize()
    ))

fig.update_layout(
    title=f"Daily Logins & Payments for {selected_channel}",
    xaxis_title="Date",
    yaxis_title="Total Count",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# 📌 Peak Day Analysis (Which days have the most activity?)
df['day_of_week'] = df['timestamp'].dt.day_name()
peak_days = df.groupby('day_of_week')['count'].sum().reset_index()

fig_peak = px.bar(
    peak_days, 
    x='day_of_week', 
    y='count', 
    title="Total Logins & Payments by Day of the Week",
    labels={"count": "Total Count", "day_of_week": "Day of the Week"},
    color='count',
    color_continuous_scale='viridis'
)
st.plotly_chart(fig_peak, use_container_width=True)

# 📌 Interactive Heatmap for Peak Login/Payment Hours
df['hour'] = df['timestamp'].dt.hour  # Extract hour
heatmap_data = df.groupby(['day_of_week', 'hour'])['count'].sum().reset_index()

fig_heatmap = px.density_heatmap(
    heatmap_data, 
    x='hour', 
    y='day_of_week', 
    z='count', 
    title="Hourly Activity by Day of Week",
    labels={'hour': 'Hour of the Day', 'day_of_week': 'Day of the Week', 'count': 'Total Count'},
    color_continuous_scale='Blues'
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# Add a footer
st.markdown("**Developed for Logins & Payments Analysis 🚀**")
