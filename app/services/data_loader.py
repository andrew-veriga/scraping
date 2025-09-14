import pandas as pd
import logging
from typing import Dict, Any

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
    
    messages_df.set_index('Message ID',inplace=True, drop=False)

    logging.info(f"{len(messages_df)} rows of data loaded and preprocessed successfully.")
    return messages_df


def load_messages_to_database(messages_df: pd.DataFrame) -> int:
    """
    Load messages from DataFrame into database.
    Returns the number of new messages created.
    """
    try:
        from app.services.database import get_database_service
        
        db_service = get_database_service()
        
        # Convert DataFrame to list of dictionaries for database insertion
        messages_data = []
        for _, row in messages_df.iterrows():
            message_data = {
                'message_id': str(row['Message ID']),
                'author_id': str(row['Author ID']),
                'content': str(row['Content']),
                'datetime': row['DateTime'],
                'dated_message': str(row['DatedMessage']),
                'referenced_message_id': str(row['Referenced Message ID']) if row['Referenced Message ID'] else ''
            }
            messages_data.append(message_data)
        
        # Bulk create messages in database
        created_count = db_service.bulk_create_messages(messages_data)
        logging.info(f"Loaded {created_count} new messages into database (out of {len(messages_data)} total messages)")
        
        return created_count
        
    except Exception as e:
        logging.error(f"Failed to load messages to database: {e}")
        return 0
