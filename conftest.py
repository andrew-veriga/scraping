import pytest
import pandas as pd
from datetime import datetime

@pytest.fixture
def sample_messages_df():
    """Provides a sample DataFrame of messages for testing."""
    data = {
        'Message ID': ['1', '2', '3', '4', '5'],
        'Author ID': ['101', '102', '101', '103', '102'],
        'Content': [
            'How do I stake SUI?',
            'You can use a wallet like Sui Wallet.',
            'Thanks!',
            'I have a problem with a transaction.',
            'What is the transaction hash?'
        ],
        'Unix Timestamp': [
            1672531200,  # 2023-01-01 00:00:00
            1672531260,  # 2023-01-01 00:01:00
            1672531320,  # 2023-01-01 00:02:00
            1672531380,  # 2023-01-01 00:03:00
            1672531440,  # 2023-01-01 00:04:00
        ],
        'Referenced Message ID': ['', '1', '', '', '4']
    }
    df = pd.DataFrame(data)
    # Replicate preprocessing from data_loader
    df['DateTime'] = pd.to_datetime(df['Unix Timestamp'], unit='s')
    df['DatedMessage'] = df.DateTime.astype(str) + " - " + df.Content
    df = df.drop(columns=['Unix Timestamp'])
    df['Referenced Message ID'] = df['Referenced Message ID'].fillna('')
    return df