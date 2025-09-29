import pandas as pd
import logging
from typing import Dict, Any, List, Tuple

def is_admin(message_ID):
    if message_ID in [
        '862550907349893151',
        '466815633347313664',
        '997105563123064892',
        '457962750644060170'
    ]:
        return True
    return False

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses Discord data from Excel file.
    Enhanced to handle hierarchical message relationships.
    """
    logging.info(f"Loading Discord data from {file_path}")
    
    # Load the Excel data with proper data types
    messages_df = pd.read_excel(
        file_path, 
        dtype={
            'Referenced Message ID': str, 
            'Message ID': str, 
            'Author ID': str,
            'Content': str
        }, 
        engine='openpyxl',  # Use openpyxl engine for better Unicode support
        na_values=['', 'nan', ' ', '\u200b', '\u200c', '\u200d'],  # Treat zero-width spaces as NaN
        keep_default_na=True,   # Keep default NaN handling
    )
       # Clean zero-width spaces from all string columns
    unicode_chars_to_remove = ['\u200b', '\u200c', '\u200d', '\ufeff']  # Zero-width spaces and BOM
    for col in messages_df.columns:
        if messages_df[col].dtype == 'object':  # String columns
            for char in unicode_chars_to_remove:
                messages_df[col] = messages_df[col].astype(str).str.replace(char, '', regex=False)

    # Clean and preprocess data
    messages_df['Referenced Message ID'] = messages_df['Referenced Message ID'].fillna('')
    messages_df['Message ID'] = messages_df['Message ID'].astype(str)
    messages_df['Author ID'] = messages_df['Author ID'].astype(str)
    
    # Convert timestamps
    if 'Unix Timestamp' in messages_df.columns:
        messages_df['DateTime'] = pd.to_datetime(messages_df['Unix Timestamp'], unit='s', utc=True)
        # Remove Unix Timestamp column (no longer needed)
        messages_df = messages_df.drop(columns=['Unix Timestamp'], errors='ignore')
        # else DataTime already exists
    # Set Message ID as index for easy lookup
    messages_df.set_index('Message ID', inplace=True, drop=False)
    
    # Sort by timestamp for proper processing order
    messages_df = messages_df.sort_values('DateTime')
    
    logging.info(f"{len(messages_df)} messages loaded and preprocessed successfully")
    authors_df = extract_authors_from_messages(messages_df)
    

    return messages_df, authors_df


def load_messages_to_database(messages_df: pd.DataFrame) -> Dict[str, int]:
    """
    Load messages from DataFrame into database with enhanced statistics.
    Returns detailed statistics about the loading process.
    """
    try:
        from app.services.database import get_database_service
        
        db_service = get_database_service()
        stats = {
            'total_messages': len(messages_df),
            'new_messages_created': 0,
            'existing_messages_skipped': 0,
            'threads_created': 0
        }
        
        logging.info(f"Loading {len(messages_df)} messages into database...")
        
        # Convert DataFrame to list of dictionaries for database insertion
        messages_data = []
        for _, row in messages_df.iterrows():
            # Prepare message data with proper null handling and hierarchical fields
            message_data = {
                'message_id': str(row['Message ID']).replace('\u200b', ''),
                'parent_id': str(row.get('parent_id','')) if pd.notna(row.get('parent_id')) and row.get('parent_id') else None,
                'author_id': str(row['Author ID']).replace('\u200b', ''),
                'content': str(row['Content']),
                'datetime': row['DateTime'],
                'referenced_message_id': str(row['Referenced Message ID']) if pd.notna(row['Referenced Message ID']) and row['Referenced Message ID'] else None,
                'attachments': str(row.get('Attachments', '')) if pd.notna(row.get('Attachments')) and row.get('Attachments') else None,
                'thread_id': str(row['thread_id']) if pd.notna(row.get('thread_id')) and row.get('thread_id') else None
            }
            messages_data.append(message_data)
        
        # Try hierarchical method first, fallback to basic method
        try:
            # Use hierarchical method if available
            created_count = db_service.bulk_create_messages_hierarchical(messages_data)
        except AttributeError:
            # Fallback to basic method if hierarchical method doesn't exist
            logging.info("Hierarchical method not available, using basic method")
            # Remove hierarchical fields for basic method
            basic_messages_data = []
            for msg in messages_data:
                basic_msg = {k: v for k, v in msg.items() if k not in ['parent_id', 'thread_id']}
                basic_messages_data.append(basic_msg)
            created_count = db_service.bulk_create_messages(basic_messages_data)
        
        stats['new_messages_created'] = created_count
        stats['existing_messages_skipped'] = stats['total_messages'] - created_count
        stats['threads_created'] = 0  # No threads created in raw loading phase
        
        logging.info("Message loading complete:")
        logging.info(f"  ✅ New messages created: {stats['new_messages_created']}")
        logging.info(f"  ⏭️  Existing messages skipped: {stats['existing_messages_skipped']}")
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to load messages to database: {e}")
        # Return stats with error indication
        return {
            'total_messages': len(messages_df) if 'messages_df' in locals() else 0,
            'new_messages_created': 0,
            'existing_messages_skipped': 0,
            'threads_created': 0,
            'error': str(e)
        }


def load_authors_to_database(authors_df: pd.DataFrame) -> Dict[str, int]:
    """
    Load authors from DataFrame into database with enhanced statistics.
    Returns detailed statistics about the loading process.
    """
    try:
        from app.services.database import get_database_service
        
        db_service = get_database_service()
        stats = {
            'total_authors': len(authors_df),
            'new_authors_created': 0,
            'existing_authors_skipped': 0
        }
        
        logging.info(f"Loading {len(authors_df)} authors into database...")
        
        # Validate and clean author data
        cleaned_authors_data = []
        for _, row in authors_df.iterrows():
            # Ensure required fields are present
            
            required_fields = ['author_id', 'author_name', 'author_type']
            if not all(field in authors_df.columns for field in required_fields):
                logging.warning(f"DataFrame missing required columns: {required_fields}")
                break
            
            # Clean author data
            cleaned_author = {
                'author_id': str(row['author_id']).strip().replace('\u200b', ''),
                'author_name': str(row['author_name']).strip(),
                'author_type': str(row['author_type']).strip()
            }
            
            # Skip if author_id is empty
            if not cleaned_author['author_id']:
                logging.warning(f"Skipping author with empty author_id: {row.to_dict()}")
                continue
                
            cleaned_authors_data.append(cleaned_author)
        
        # Bulk create authors in database
        created_count = db_service.bulk_create_authors(cleaned_authors_data)
        
        stats['new_authors_created'] = created_count
        stats['existing_authors_skipped'] = stats['total_authors'] - created_count
        
        logging.info("Author loading complete:")
        logging.info(f"  ✅ New authors created: {stats['new_authors_created']}")
        logging.info(f"  ⏭️  Existing authors skipped: {stats['existing_authors_skipped']}")
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to load authors to database: {e}")
        # Return stats with error indication
        return {
            'total_authors': len(authors_df) if 'authors_df' in locals() else 0,
            'new_authors_created': 0,
            'existing_authors_skipped': 0,
            'error': str(e)
        }


def extract_authors_from_messages(messages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique authors from messages DataFrame.
    Returns a DataFrame with author data ready for database insertion.
    """
    try:
        # Get unique authors from the messages
        authors_df = messages_df[['Author ID', 'Author Name']].drop_duplicates()
        
        # Create new DataFrame with required columns
        authors_data = []
        for _, row in authors_df.iterrows():
            author_id = str(row['Author ID']).strip()
            author_name = str(row['Author Name']).strip()
            
            # Skip if author_id is empty
            if not author_id:
                continue
            
            # Determine author type
            author_type = 'admin' if is_admin(author_id) else 'user'
            
            authors_data.append({
                'author_id': author_id,
                'author_name': author_name,
                'author_type': author_type
            })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(authors_data)
        
        logging.info(f"Extracted {len(result_df)} unique authors from messages")
        return result_df
        
    except Exception as e:
        logging.error(f"Failed to extract authors from messages: {e}")
        return pd.DataFrame(columns=['author_id', 'author_name', 'author_type'])
