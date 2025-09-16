import json
import os
from datetime import datetime
import pandas as pd
import logging
import re

def create_dict_from_list(solutions_list):
    """Converts a list of solution dictionaries to a dictionary keyed by topic_id."""
    solutions_dict = {}
    for solution in solutions_list:
        # Support both old and new formats during transition
        key = solution.get('topic_id') or solution.get('Topic_ID')
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

def add_new_solutions_to_dict(solutions_dict, new_solutions_list):
    """Adds a list of new solutions to the solutions dictionary."""
    for new_solution in new_solutions_list:
        add_or_update_solution(solutions_dict, new_solution)
    logging.info(f"Added/updated {len(new_solutions_list)} solutions to the dictionary.")

def save_solutions_dict(solutions_dict, filename, save_path):
    """Saves the solutions dictionary to a JSON file."""
    full_path = os.path.join(save_path,filename)
    with open(full_path, 'w') as f:
        json.dump(solutions_dict, f, indent=2, default=convert_datetime_to_str)
    logging.info(f"Solutions dictionary saved to {full_path}")
    return full_path

def convert_datetime_to_str(obj):
    """JSON serializer for objects not serializable by default json code, like datetime."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def load_solutions_dict(filename, save_path, from_date=None):
    """Loads the solutions dictionary from a JSON file with datetime conversion and optional date filtering."""
    full_path = os.path.join(save_path, filename)
    try:
        with open(full_path, 'r') as f:
            solutions_dict = json.load(f)

        # Convert datetime strings back to datetime objects and apply filtering
        filtered_solutions_dict = {}
        for topic_id, solution in solutions_dict.items():
           
            actual_date = solution.get('actual_date')
            if from_date is None or (isinstance(actual_date, datetime) and actual_date.replace(tzinfo=None) >= from_date.replace(tzinfo=None)):
                 filtered_solutions_dict[topic_id] = solution


        logging.info(f"Solutions dictionary loaded from {full_path}")
        if from_date:
             logging.info(f"Filtered to include solutions from {from_date.strftime('%Y-%m-%d')} onwards.")
        return filtered_solutions_dict
    except FileNotFoundError:
        logging.warning(f"Solutions file not found at {full_path}. Returning empty dictionary.")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {full_path}. Returning empty dictionary.")
        return {}
    except ValueError as e:
        logging.error(f"Error converting datetime string in {full_path}: {e}")
        return solutions_dict # Return partially loaded data if conversion fails for some entries

def get_end_date_from_solutions(solutions_dict):
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
        latest_date = max(dates).normalize() + pd.Timedelta(days=1)
        return latest_date
    return None

def is_admin(message_ID):
    if message_ID in [
        '862550907349893151',
        '466815633347313664',
        '997105563123064892',
        '457962750644060170'
    ]:
        return True
    return False

# Map Author IDs to User <N> using all unique author IDs from the entire Messages_df

def illustrated_message(message_ID, db_service, session=None):
    """
    Formats a message with user mapping and reply information.
    user_mapping: A dictionary mapping Author IDs to formatted user names.

    Args:
        message_ID: A pandas Series representing a message row from the dataframe.
        db_service: Database service instance
        session: Optional existing database session (to avoid creating new connections)

    Returns:
        A formatted string representation of the message.
    """
    # Use provided session or create new one
    if session is not None:
        message = db_service.get_message_by_message_id(message_ID, session)
        if message is None:
            return "<empty>"
        author_id = message.author_id
        referenced_message_id = message.referenced_message_id
        message_content = message.content
        user_mapping = lambda author_id: f'Admin {author_id}' if is_admin(author_id) else f'User {author_id}'
        formatted_msg = f"{message.datetime} {user_mapping(author_id)}: "

        # Handle replies
        if referenced_message_id:
            referenced_message = db_service.get_message_by_message_id(referenced_message_id, session)
            if referenced_message:
                referenced_author_id = referenced_message.author_id
                formatted_msg += f" reply to {user_mapping(referenced_author_id)} - "

        # Ensure message_content is a string before using re.sub
        if message_content is None:
            message_content = "<empty>"
        else:
            message_content = str(message_content)

        # Handle tagged users using regex to find and replace the pattern <@digits>
        def replace_tagged_users(match):
            # Extract the tagged user ID (digits) from the match
            tagged_user_id = match.group(1)
            # Replace with the formatted user name using the user_mapping dictionary
            return f"<tagged {user_mapping(tagged_user_id)}>"

        # Use re.sub with a raw string for the regex pattern
        message_content = re.sub(r'<@(\d+)>', replace_tagged_users, message_content)

        formatted_msg += message_content
        return formatted_msg
    else:
        # Fallback to creating new session (less efficient)
        with db_service.get_session() as new_session:
            return illustrated_message(message_ID, db_service, new_session)


def illustrated_threads(threads_json_data, db_service):
    """
    Enriches thread data with full message content for better analysis by the LLM.
    """
    # Use a single database session for all message lookups to avoid connection pool exhaustion
    with db_service.get_session() as session:
        for thread in threads_json_data:
            thread['whole_thread_formatted'] = []
            # Support both old and new formats during transition
            whole_thread = thread.get('whole_thread') or thread.get('whole_thread', [])
            for msg in whole_thread:
                msg_id = msg['message_id']
                message_content = illustrated_message(msg_id, db_service, session)
                thread['whole_thread_formatted'].append({
                    "message_id": str(msg_id),
                    "parent_id": str(msg.get('parent_id', None)),
                    "content": message_content
                })
    return threads_json_data
