import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.title("Hourly Event Analysis Dashboard")

st.write("Upload your Excel file with Days + hourly buckets")

# -------------------------
# UPLOAD FILE
# -------------------------
file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

if file:
    df = pd.read_excel(file)

    st.subheader("Raw Data Preview")
    st.dataframe(df)

    # Ensure first column is Days
    df = df.rename(columns={df.columns[0]: "Days"})

    # Melt to long format
    long_df = df.melt(id_vars="Days", var_name="Hour", value_name="Count")

    # ----------------------------------------------------
    # 1️⃣ LINE PLOT — each row is one line
    # ----------------------------------------------------
    st.subheader("Hourly Line Plot (Each Day as Separate Line)")

    fig_line = px.line(
        long_df,
        x="Hour",
        y="Count",
        color="Days",
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ----------------------------------------------------
    # 2️⃣ BAR PLOT — total for each row
    # ----------------------------------------------------
    st.subheader("Total Count per Day (Bar Chart)")

    df["Total"] = df.drop(columns=["Days"]).sum(axis=1)

    fig_bar = px.bar(
        df,
        x="Days",
        y="Total",
        text="Total"
    )
    fig_bar.update_traces(textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ----------------------------------------------------
    # 3️⃣ DOWNLOAD SECTION
    # ----------------------------------------------------
    st.subheader("Download Files")

    download_path = st.text_input("Enter folder path to save files (e.g., C:/Users/Downloads/myfolder):")

    if st.button("Save Plots & Data to Path"):
        if download_path.strip() == "":
            st.error("Please enter a valid path!")
        else:
            try:
                os.makedirs(download_path, exist_ok=True)

                # Save CSV
                csv_path = os.path.join(download_path, "processed_data.csv")
                df.to_csv(csv_path, index=False)

                # Save plots as PNG
                fig_line.write_image(os.path.join(download_path, "line_plot.png"))
                fig_bar.write_image(os.path.join(download_path, "bar_plot.png"))

                st.success(f"Files saved to: {download_path}")

            except Exception as e:
                st.error(f"Error saving files: {e}")
