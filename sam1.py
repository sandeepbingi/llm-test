 if st.session_state.get("forecast_clicked", False) and st.session_state.get("forecast_df") is not None:
        col1, col2 = st.columns([0.8, 1.15])
        with col1:
            st.subheader("Forecast Results")
        with col2:
            csv = st.session_state.forecast_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, f"forecast_results_{selected_channel}_{future_date}.csv", "text/csv")

        show_actual = st.toggle("Show Actual Data (if available)", value=False)

        forecast_df = st.session_state.forecast_df.copy()

        if show_actual:
            actual_map = {}
            for t in forecast_df["Type"].unique():
                dates = forecast_df[forecast_df["Type"] == t]["Datetime"].tolist()
                actual_vals = get_actual_values(selected_channel, t.lower(), dates)
                if actual_vals:
                    for i, dt in enumerate(dates[:len(actual_vals)]):
                        actual_map[(t, dt)] = actual_vals[i]
            forecast_df["Actual"] = forecast_df.apply(lambda r: actual_map.get((r["Type"], r["Datetime"]), None), axis=1)

        st.dataframe(forecast_df)

        col3, col4 = st.columns([0.8, 0.2])
        with col3:
            st.subheader("Trend Visualization")
            fig = px.line(
                forecast_df,
                x="Datetime",
                y="Forecasted Count",
                color="Type",
                markers=True,
                title=f"Forecasting Trend - {selected_channel} : {future_date}",
                color_discrete_sequence=["#4169E1", "#FFA500", "#90EE90", "#FF6347"]
            )
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=40, r=40, t=40, b=40),
                xaxis_title="Time", yaxis_title="Count",
                xaxis=dict(tickmode="linear", dtick=3 * 3600 * 1000,
                           showgrid=True, gridcolor="lightgray", tickangle=60),
                yaxis=dict(showgrid=True, gridcolor="lightgray"),
                hovermode="x unified", template="plotly_dark"
            )
            fig.update_traces(mode="lines+markers", hovertemplate="Time %{x}<br>Count: %{y}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            buf = io.BytesIO()
            fig.write_image(buf, format="png")
            st.download_button("Download Plot", buf.getvalue(),
                               file_name=f"forecast_plot_{selected_channel}_{future_date}.png", mime="image/png")
