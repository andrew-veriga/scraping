import pandas as pd
import logging

def load_and_preprocess_data(file_path):
    """Loads and preprocesses the data from the Excel file."""
    # Load the Excel data
    messages_df = pd.read_excel(file_path, dtype={'Referenced Message ID': str, 'Message ID': str, 'Author ID': str})

    # Fill any potential NaN values in 'Referenced Message ID' with empty strings
    messages_df['Referenced Message ID'] = messages_df['Referenced Message ID'].fillna('')

    # Convert 'Message ID' in CSV to string for easier lookup and merging
    messages_df['DateTime'] = pd.to_datetime( messages_df['Unix Timestamp'],unit='s', utc=True)
    messages_df['DatedMessage'] = messages_df.DateTime.astype(str) +" - " + messages_df.Content
    # Remove the 'Unix Timestamp' column
    messages_df = messages_df.drop(columns=['Unix Timestamp'])
    logging.info(f"{len(messages_df)} rows of data loaded and preprocessed successfully.")
    return messages_df
