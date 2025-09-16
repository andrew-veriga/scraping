"""
Enhanced data loader for hierarchical Discord message structure
Handles parent-child relationships and depth calculation
"""

import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from app.models.db_models import Thread

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
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
        }
    )
    
    # Clean and preprocess data
    messages_df['Referenced Message ID'] = messages_df['Referenced Message ID'].fillna('')
    messages_df['Message ID'] = messages_df['Message ID'].astype(str)
    messages_df['Author ID'] = messages_df['Author ID'].astype(str)
    
    # Convert timestamps
    messages_df['DateTime'] = pd.to_datetime(messages_df['Unix Timestamp'], unit='s', utc=True)
    messages_df['DatedMessage'] = messages_df['DateTime'].astype(str) + " - " + messages_df['Content'].astype(str)
    
    # Remove Unix Timestamp column (no longer needed)
    messages_df = messages_df.drop(columns=['Unix Timestamp'], errors='ignore')
    
    # Set Message ID as index for easy lookup
    messages_df.set_index('Message ID', inplace=True, drop=False)
    
    # Sort by timestamp for proper processing order
    messages_df = messages_df.sort_values('DateTime')
    
    logging.info(f"{len(messages_df)} messages loaded and preprocessed successfully")
    return messages_df

def analyze_message_hierarchy(messages_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analyze Discord messages to build hierarchical parent-child relationships.
    Returns enriched DataFrame with hierarchy information and statistics.
    """
    logging.info("Analyzing message hierarchy and parent-child relationships...")
    
    # Create working copy
    df = messages_df.copy()
    
    # Initialize hierarchy fields
    df['parent_id'] = ''
    df['thread_id'] = ''
    
    # Build message lookup for fast access
    message_lookup = df.set_index('Message ID').to_dict('index')
    
    # Track thread relationships
    thread_stats = {
        'total_messages': len(df),
        'root_messages': 0,
        'reply_messages': 0,
        'threads_identified': 0,
        'orphaned_replies': 0
    }
    
    # First pass: Map Referenced Message ID to parent_id for hierarchical structure
    for idx, row in df.iterrows():
        message_id = row['Message ID']
        referenced_id = row['Referenced Message ID']
        
        if referenced_id and referenced_id != '' and referenced_id in message_lookup:
            # This message is a reply to another message (referenced_id becomes parent_id)
            df.loc[df['Message ID'] == message_id, 'parent_id'] = referenced_id
            thread_stats['reply_messages'] += 1
        else:
            # This is a root message (no parent, thread starter)
            df.loc[df['Message ID'] == message_id, 'parent_id'] = None
            df.loc[df['Message ID'] == message_id, 'thread_id'] = message_id  # Use message_id as thread_id
            thread_stats['root_messages'] += 1
    
    # Second pass: Assign thread IDs to all messages
    def find_thread_id(message_id: str, visited: set = None) -> str:
        """Recursively find root thread ID"""
        if visited is None:
            visited = set()
        
        if message_id in visited:
            # Circular reference detection
            return message_id
        
        visited.add(message_id)
        
        message_row = df[df['Message ID'] == message_id].iloc[0]
        parent_id = message_row['parent_id']
        
        if not parent_id or parent_id not in message_lookup:
            # This is a root message
            return message_id
        else:
            # This is a reply - get parent's thread
            return find_thread_id(parent_id, visited.copy())
    
    # Apply thread assignment
    for idx, row in df.iterrows():
        message_id = row['Message ID']
        thread_id = find_thread_id(message_id)
        df.loc[df['Message ID'] == message_id, 'thread_id'] = thread_id
    
    # Update statistics
    thread_stats['threads_identified'] = df['thread_id'].nunique()
    thread_stats['orphaned_replies'] = len(df[(df['parent_id'] != '') & (~df['parent_id'].isin(df['Message ID']))])
    
    # Log hierarchy analysis results
    logging.info("Message hierarchy analysis complete:")
    logging.info(f"  📊 Total messages: {thread_stats['total_messages']}")
    logging.info(f"  🌳 Root messages: {thread_stats['root_messages']}")
    logging.info(f"  💬 Reply messages: {thread_stats['reply_messages']}")
    logging.info(f"  📝 Threads identified: {thread_stats['threads_identified']}")
    logging.info(f"  ⚠️  Orphaned replies: {thread_stats['orphaned_replies']}")
    
    return df, thread_stats

def load_messages_to_database_hierarchical(messages_df: pd.DataFrame) -> Dict[str, int]:
    """
    Load messages with hierarchical structure into database.
    Returns statistics about the loading process.
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
        
        logging.info(f"Loading {len(messages_df)} messages with hierarchy into database...")
        
        # Convert DataFrame to list of dictionaries for database insertion
        messages_data = []
        threads_data = {}  # Keep track of threads to create
        
        for _, row in messages_df.iterrows():
            # Prepare message data with proper null handling
            message_data = {
                'message_id': str(row['Message ID']),
                'parent_id': str(row.get('parent_id','')) if pd.notna(row.get('parent_id')) and row.get('parent_id') else None,
                'author_id': str(row['Author ID']),
                'content': str(row['Content']),
                'datetime': row['DateTime'],
                'dated_message': str(row['DatedMessage']),
                'referenced_message_id': str(row['Referenced Message ID']) if pd.notna(row['Referenced Message ID']) and row['Referenced Message ID'] else None,
                'thread_id': str(row['thread_id']) if pd.notna(row.get('thread_id')) and row.get('thread_id') else None
            }
            messages_data.append(message_data)
            
            # Track thread data
        #     if row['is_root_message']:
        #         thread_id = str(row['thread_id'])
        #         if thread_id not in threads_data:
        #             threads_data[thread_id] = {
        #                 'topic_id': thread_id,
        #                 'header': f"Thread starting with: {str(row['Content'])[:100]}...",
        #                 'actual_date': row['DateTime'],
        #                 'status': 'new',
        #                 'is_technical': False,  # Will be determined later by LLM
        #                 'is_processed': False
        #             }
        #             stats['threads_created'] += 1
                
        #         stats['root_messages'] += 1
        #     else:
        #         stats['reply_messages'] += 1
        
        # # Skip thread creation - threads will be created later by LLM processing
        # logging.info(f"Skipping thread creation for {len(threads_data)} potential threads - will be created by LLM")
        
        # Now create messages with hierarchy
        created_count = db_service.bulk_create_messages_hierarchical(messages_data)
        stats['new_messages_created'] = created_count
        stats['existing_messages_skipped'] = stats['total_messages'] - created_count
        stats['threads_created'] = 0  # No threads created in raw loading phase
        
        logging.info("Hierarchical message loading complete:")
        logging.info(f"  ✅ New messages created: {stats['new_messages_created']}")
        logging.info(f"  ⏭️  Existing messages skipped: {stats['existing_messages_skipped']}")
        
        return stats
        
    except Exception as e:
        logging.error(f"Failed to load messages to database: {e}")
        raise

def validate_message_hierarchy(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate the integrity of the message hierarchy.
    Returns validation results and any issues found.
    """
    validation = {
        'valid': True,
        'issues': [],
        'warnings': [],
        'statistics': {}
    }
    
    logging.info("Validating message hierarchy integrity...")
    
    # Check 1: All parent_id references point to existing messages
    messages_with_parents = messages_df[
        (messages_df['parent_id'].notna()) & 
        (messages_df['parent_id'] != '') & 
        (messages_df['parent_id'] != 'None')
    ]
    for _, row in messages_with_parents.iterrows():
        parent_id = row['parent_id']
        if parent_id not in messages_df['Message ID'].values:
            validation['issues'].append(f"Message {row['Message ID']} references non-existent parent {parent_id}")
            validation['valid'] = False
    
    # Check 2: No circular references
    def has_circular_reference(message_id: str, visited: set = None) -> bool:
        if visited is None:
            visited = set()
        
        if message_id in visited:
            return True
        
        visited.add(message_id)
        message_row = messages_df[messages_df['Message ID'] == message_id]
        if message_row.empty:
            return False
        
        parent_id = message_row.iloc[0]['parent_id']
        if parent_id and parent_id != '':
            return has_circular_reference(parent_id, visited)
        
        return False
    
    circular_refs = []
    for message_id in messages_df['Message ID']:
        if has_circular_reference(message_id):
            circular_refs.append(message_id)
    
    if circular_refs:
        validation['issues'].append(f"Circular references detected in messages: {circular_refs}")
        validation['valid'] = False
    
    # Check 3: Thread IDs are consistent with parent relationships
    thread_inconsistencies = []
    for _, row in messages_df.iterrows():
        if row['parent_id']:
            parent_row = messages_df[messages_df['Message ID'] == row['parent_id']]
            if not parent_row.empty:
                parent_thread = parent_row.iloc[0]['thread_id']
                if row['thread_id'] != parent_thread:
                    thread_inconsistencies.append(f"Message {row['Message ID']} has thread {row['thread_id']}, expected {parent_thread}")
    
    if thread_inconsistencies:
        validation['warnings'].extend(thread_inconsistencies)
    
    # Gather statistics
    validation['statistics'] = {
        'total_messages': len(messages_df),
        'root_messages': len(messages_df[messages_df['parent_id'].isna() | (messages_df['parent_id'] == '')]),
        'threads': messages_df['thread_id'].nunique(),
        'orphaned_replies': len(messages_df[
            (messages_df['parent_id'] != '') & 
            (~messages_df['parent_id'].isin(messages_df['Message ID']))
        ])
    }
    
    if validation['valid']:
        logging.info("✅ Message hierarchy validation passed")
    else:
        logging.warning(f"⚠️ Message hierarchy validation failed with {len(validation['issues'])} issues")
    
    if validation['warnings']:
        logging.warning(f"⚠️ {len(validation['warnings'])} warnings found during validation")
    
    return validation