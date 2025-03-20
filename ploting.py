import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from glob import glob

# Define folder paths
logins_folder = "C:/Users/Desktop/Logins_Payments_LLM/Login_Reports/"
payments_folder = "C:/Users/Desktop/Logins_Payments_LLM/Payment_Reports/"

# Function to load and label data
def load_data(folder, type_label):
    files = glob(os.path.join(folder, "*.csv"))  # Get all CSV files
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Convert timestamp
        df['type'] = type_label  # Label as logins or payments
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

# Load data from both folders
df_logins = load_data(logins_folder, "logins")
df_payments = load_data(payments_folder, "payments")

# Combine both datasets
df = pd.concat([df_logins, df_payments], ignore_index=True)

# Extract date from timestamp
df['date'] = df['timestamp'].dt.date

# Aggregate total count per day per channel per type
daily_counts = df.groupby(['channel', 'date', 'type'])['count'].sum().reset_index()

# 📌 Interactive Time-Series Plots for Each Channel
channels = df['channel'].unique()

for channel in channels:
    df_channel = daily_counts[daily_counts['channel'] == channel]

    # Create separate traces for logins and payments
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
        title=f"Daily Logins & Payments for {channel}",
        xaxis_title="Date",
        yaxis_title="Total Count",
        hovermode="x unified"
    )
    fig.show()

# 📌 Interactive Peak Day Analysis (Which days have the most activity?)
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
fig_peak.show()

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
fig_heatmap.show()
