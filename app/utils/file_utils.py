import json
import os
from datetime import datetime
import pandas as pd
import logging
import re
from sqlalchemy import text

def create_dict_from_list(solutions_list):
    """Converts a list of solution dictionaries to a dictionary keyed by topic_id."""
    solutions_dict = {}
    for solution in solutions_list:
        # Support both old and new formats during transition
        key = solution.get('topic_id')
        if key:
            solutions_dict[key] = solution
    return solutions_dict

def add_or_update_solution(solutions_dict, new_solution):
    """Adds a new solution or updates an existing one in the solutions dictionary."""
    # Support both old and new formats during transition
    key = new_solution.get('topic_id')
    if key:
        solutions_dict[key] = new_solution
        logging.info(f"solution for topic_id {key} added or updated.")

def add_new_solutions_to_dict(solutions_dict, new_solutions):
    """Adds new solutions to the solutions dictionary."""
    if isinstance(new_solutions, list):
        for new_solution in new_solutions:
            add_or_update_solution(solutions_dict, new_solution)
    elif isinstance(new_solutions, dict):
        for _, new_solution in new_solutions.items():
            add_or_update_solution(solutions_dict, new_solution)
    else:
        logging.exception(f"Invalid type of new_solutions: {type(new_solutions)}")
    logging.info(f"Added/updated {len(new_solutions)} solutions to the dictionary.")

import yaml
def get_latest_normalized_date(config) -> datetime:
    """Get the latest datetime from config, normalized and incremented by one day."""
    latest_date = pd.Timestamp(config['LATEST_SOLUTION_DATE']).normalize() + pd.Timedelta(days=1)
    # Ensure timezone-aware datetime to match messages DataFrame
    if latest_date.tz is None:
        latest_date = latest_date.tz_localize('UTC')
    return latest_date.to_pydatetime()

def set_latest_solution_date(config, new_date: pd.Timestamp)->dict:
    """Set the latest solution date in config to the new_date, normalized."""
    config['LATEST_SOLUTION_DATE'] = new_date.isoformat()
    # with open("configs/config.yaml", "r") as f:
    #     saved_config = yaml.safe_load(f)
        # saved_config['LATEST_SOLUTION_DATE'] = new_date.isoformat()
    with open("configs/config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    logging.info(f"Updated LATEST_SOLUTION_DATE in config to {new_date.isoformat()}")
    return config

def save_solutions_dict(solutions_dict, config):
    """Saves the solutions dictionary to a JSON file."""
    filename = config['SOLUTIONS_DICT_FILENAME']
    save_path = config['SAVE_PATH']
    full_path = os.path.join(save_path, filename)
    with open(full_path, 'w') as f:
        json.dump(solutions_dict, f, indent=2, default=convert_Timestamp_to_str)
    logging.info(f"Solutions dictionary saved to {full_path}")
    # Update latest solution date in config
    latest_date = get_latest_date_from_solutions(solutions_dict)
    set_latest_solution_date(config, new_date=latest_date)
    return full_path

def convert_Timestamp_to_str(obj):
    """JSON serializer for objects not serializable by default json code, like datetime."""
    if isinstance(obj, pd.Timestamp) or isinstance(obj, datetime):
        obj = obj.isoformat(timespec='seconds')
        return obj
    # raise TypeError(f"Type {type(obj)} not serializable")

def convert_str_to_Timestamp(obj):
    """JSON deserializer for objects not deserializable by default json code, like datetime."""
    if obj.get('actual_date', None):
        obj['actual_date'] = pd.Timestamp(obj['actual_date']).tz_convert('UTC')
    return obj

def load_solutions_dict(config, from_date=None):
    """Loads the solutions dictionary from a JSON file with datetime conversion and optional date filtering."""
    filename = config['SOLUTIONS_DICT_FILENAME']
    save_path = config['SAVE_PATH']
    full_path = os.path.join(save_path, filename)
    try:
        with open(full_path, 'r') as f:
            solutions_dict = json.load(f,object_hook=convert_str_to_Timestamp)

        # TODO:Convert datetime strings back to pd.Timestamp objects
        if from_date:
            from_date = pd.Timestamp(from_date).tz_convert('UTC')
            filtered_solutions_dict = {}
            for topic_id, solution in solutions_dict.items():
                if solution['actual_date'] >= from_date:
                    filtered_solutions_dict[topic_id] = solution
            logging.info(f"Filtered to include {len(filtered_solutions_dict)} solutions from { convert_Timestamp_to_str(from_date)} onwards.")
            return filtered_solutions_dict

        logging.info(f"Solutions dictionary loaded from {full_path}")
        return solutions_dict
    except FileNotFoundError:
        logging.exception(f"Solutions file not found at {full_path}. Returning empty dictionary.")
        return {}
    except json.JSONDecodeError:
        logging.exception(f"Error decoding JSON from {full_path}. Returning empty dictionary.")
        return {}
    except ValueError as e:
        logging.exception(f"Error converting datetime string in {full_path}: {e}")
        return solutions_dict # Return partially loaded data if conversion fails for some entries

def get_latest_date_from_solutions(solutions_dict):
    """Extracts the latest actual_date from the solutions dictionary."""
    if not solutions_dict:
        return None
    # Support both old and new formats during transition
    dates = []
    for s in solutions_dict.values():
        actual_date = s.get('actual_date')
        if actual_date:
            dates.append(pd.Timestamp(actual_date))
    if dates:
        latest_date = max(dates)#.normalize() + pd.Timedelta(days=1)
        return latest_date
    return None



# Map Author IDs to User <N> using all unique author IDs from the entire Messages_df

def illustrated_message(message_ID, db_service, session=None):
    """
    Formats a message with user mapping and reply information using the database view.
    
    Args:
        message_ID: A string representing the message ID.
        db_service: Database service instance
        session: Optional existing database session (to avoid creating new connections)

    Returns:
        A formatted string representation of the message from the illustrated_messages view.
    """
    # Use provided session or create new one
    if session is not None:
        try:
            # Query the illustrated_messages view for the specific message
            result = session.execute(
                text("SELECT illustrated_message FROM illustrated_messages WHERE message_id = :message_id"),
                {"message_id": str(message_ID)}
            )
            
            row = result.fetchone()
            if row:
                return row[0]  # Return the illustrated_message field
            else:
                return "<empty>"
                
        except Exception as e:
            logging.error(f"Error querying illustrated_messages view for message {message_ID}: {e}")
            return "<empty>"
    else:
        # Fallback to creating new session (less efficient)
        with db_service.get_session() as new_session:
            return illustrated_message(message_ID, db_service, new_session)


def bulk_illustrated_messages(message_ids, db_service, session=None):
    """
    Retrieves illustrated messages for a list of message IDs using bulk query for better performance.
    
    Args:
        message_ids: List of message IDs to retrieve illustrated messages for
        db_service: Database service instance
        session: Optional existing database session (to avoid creating new connections)
    
    Returns:
        Dictionary mapping message_id to illustrated_message content
    """
    if not message_ids:
        return {}
    
    # Use provided session or create new one
    if session is not None:
        try:
            # Create a parameterized query with IN clause
            placeholders = ','.join([f':msg_id_{i}' for i in range(len(message_ids))])
            query = f"""
                SELECT message_id, illustrated_message 
                FROM illustrated_messages 
                WHERE message_id IN ({placeholders})
            """
            
            # Create parameters dictionary
            params = {f'msg_id_{i}': msg_id for i, msg_id in enumerate(message_ids)}
            
            # Execute the bulk query
            result = session.execute(text(query), params)
            
            # Create a dictionary mapping message_id to illustrated_message
            illustrated_messages_dict = {row[0]: row[1] for row in result}
            
            return illustrated_messages_dict
                
        except Exception as e:
            logging.error(f"Error bulk querying illustrated_messages: {e}")
            # Fallback to individual queries if bulk query fails
            illustrated_messages_dict = {}
            for msg_id in message_ids:
                message_content = illustrated_message(msg_id, db_service, session)
                illustrated_messages_dict[str(msg_id)] = message_content
            return illustrated_messages_dict
    else:
        # Fallback to creating new session (less efficient)
        with db_service.get_session() as new_session:
            return bulk_illustrated_messages(message_ids, db_service, new_session)


def illustrated_threads(threads_json_data, db_service):
    """
    Enriches thread data with full message content for better analysis by the LLM.
    Uses bulk query to the illustrated_messages view for better performance.
    """
    # Use a single database session for all message lookups to avoid connection pool exhaustion
    with db_service.get_session() as session:
        for thread in threads_json_data:
            # Support both old and new formats during transition
            whole_thread = thread.get('whole_thread',[]) 
            
            if not whole_thread or len(whole_thread) == 0:
                continue

            # Extract all message IDs from the thread
            message_ids = [msg['message_id'] for msg in whole_thread]
            
            # Use the new function to get illustrated messages
            illustrated_messages_dict = bulk_illustrated_messages(message_ids, db_service, session)
            thread['whole_thread'] = []
            
            # Fill the whole_thread with the results
            for msg in whole_thread:
                msg_id = str(msg['message_id'])
                message_content = illustrated_messages_dict.get(msg_id, "<empty>")
                if msg.get('image_summary', None):
                    message_content += f"\nAttachments: {msg.get('image_summary', None)}"
                thread['whole_thread'].append({
                    "message_id": msg_id,
                    "parent_id": str(msg.get('parent_id', None)),
                    "content": message_content
                })
                    
    return threads_json_data
    
if __name__ == "__main__":
    with open("configs/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    # os.system("copy C:\\VSCode\\withrag\\results\\solutions_dict.json .\\results\\solutions_dict.json")
    solution_dict = load_solutions_dict(config, pd.Timestamp("2025-07-23T06:57:02+00:00"))
    assert len(solution_dict) > 0, "Filtered solution dictionary is empty"
    print(f'filtered len: {len(solution_dict)}')
    solution_dict = load_solutions_dict(config)
    assert len(solution_dict) > 0, "Unfiltered solution dictionary is empty"
    print(f'unfiltered len: {len(solution_dict)}')
    save_solutions_dict(solution_dict, config)
    solution_dict = load_solutions_dict(config)
    assert len(solution_dict) > 0, "Saved solution dictionary is empty"
    print(f'saved len: {len(solution_dict)}')