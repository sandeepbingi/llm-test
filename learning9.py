def load_data(selected_channel, data_type):
    processed_files = load_processed_files()

    new_files = []
    if data_type in ["Logins", "Both"]:
        logins_df, new_login_files = process_folder(DATA_FOLDER_LOGINS, "logins", processed_files)
        logins_df = logins_df[logins_df["Channel"] == selected_channel]  # Filter only selected channel

        if not logins_df.empty:
            train_or_update_model(selected_channel, "logins", logins_df, "Prophet")
            new_files.extend(new_login_files)

    if data_type in ["Payments", "Both"]:
        payments_df, new_payment_files = process_folder(DATA_FOLDER_PAYMENTS, "payments", processed_files)
        payments_df = payments_df[payments_df["Channel"] == selected_channel]  # Filter only selected channel

        if not payments_df.empty:
            train_or_update_model(selected_channel, "payments", payments_df, "Prophet")
            new_files.extend(new_payment_files)

    # Update processed files tracking only if new files were used
    if new_files:
        if data_type in ["Logins", "Both"]:
            processed_files["logins"].update(new_files)
        if data_type in ["Payments", "Both"]:
            processed_files["payments"].update(new_files)
        save_processed_files(processed_files)
