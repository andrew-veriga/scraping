import json
import os
from datetime import datetime
import pandas as pd
import logging
import re

def create_dict_from_list(solutions_list):
    """Converts a list of solution dictionaries to a dictionary keyed by Topic_ID."""
    solutions_dict = {}
    for solution in solutions_list:
        solutions_dict[solution['Topic_ID']] = solution
    return solutions_dict

def add_or_update_solution(solutions_dict, new_solution):
    """Adds a new solution or updates an existing one in the solutions dictionary."""
    solutions_dict[new_solution['Topic_ID']] = new_solution
    logging.info(f"Solution for Topic_ID {new_solution['Topic_ID']} added or updated.")

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
            # if 'Actual_Date' in solution and isinstance(solution['Actual_Date'], str):
            #     solution['Actual_Date'] = datetime.fromisoformat(solution['Actual_Date'])
            #     if solution['Actual_Date'] > pd.Timestamp('2025-08-11 00:00:00+0000', tz='UTC'):
            #         continue
            if from_date is None or (isinstance(solution['Actual_Date'], datetime) and solution['Actual_Date'].replace(tzinfo=None) >= from_date.replace(tzinfo=None)):
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
    """Extracts the latest Actual_Date from the solutions dictionary."""
    if not solutions_dict:
        return None
    latest_date = max([pd.Timestamp(s['Actual_Date']) for s in solutions_dict.values()]).normalize() + pd.Timedelta(days=1)
    return latest_date#.tz_convert( tz=None)#.tz_localize(timezone.utc)

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

def illustrated_message(message_ID, df_indexed):
    """
    Formats a message with user mapping and reply information.
    user_mapping: A dictionary mapping Author IDs to formatted user names.

    Args:
        message_ID: A pandas Series representing a message row from the dataframe.

    Returns:
        A formatted string representation of the message.
    """

    message = df_indexed.loc[message_ID]
    author_id = message['Author ID']
    referenced_message_id = message['Referenced Message ID']
    message_content = message['Content']


    user_mapping = lambda author_id: f'Admin {author_id}' if is_admin(author_id) else f'User {author_id}'

    formatted_msg = f"{message['DateTime']} {user_mapping(author_id)}: "

    # Handle replies
    if referenced_message_id in df_indexed.index:
        referenced_message = df_indexed.loc[referenced_message_id]
        referenced_author_id = referenced_message['Author ID']
        formatted_msg += f" reply to {user_mapping(referenced_author_id)} - "

    # Ensure message_content is a string before using re.sub
    if pd.isna(message_content):
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

def illustrated_threads(threads_json_data, messages_df):
    """
    Enriches thread data with full message content for better analysis by the LLM.
    """
    messages_dict = messages_df.set_index('Message ID')['DatedMessage'].to_dict()
    authors_dict = messages_df.set_index('Message ID')['Author ID'].to_dict()

    for thread in threads_json_data:
        thread['Whole_thread_formatted'] = []
        for msg_id in thread['Whole_thread']:
            message_content = messages_dict.get(str(msg_id), "Message content not found")
            author_id = authors_dict.get(str(msg_id), "Author not found")
            thread['Whole_thread_formatted'].append({
                "Message_ID": str(msg_id),
                "Author_ID": str(author_id),
                "Content": message_content
            })
    return threads_json_data
