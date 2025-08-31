import json
import os
from datetime import datetime
import pandas as pd

def create_dict_from_list(solutions_list):
    """Converts a list of solution dictionaries to a dictionary keyed by Topic_ID."""
    solutions_dict = {}
    for solution in solutions_list:
        solutions_dict[solution['Topic_ID']] = solution
    return solutions_dict

def add_or_update_solution(solutions_dict, new_solution):
    """Adds a new solution or updates an existing one in the solutions dictionary."""
    solutions_dict[new_solution['Topic_ID']] = new_solution
    print(f"Solution for Topic_ID {new_solution['Topic_ID']} added or updated.")

def add_new_solutions_to_dict(solutions_dict, new_solutions_list):
    """Adds a list of new solutions to the solutions dictionary."""
    for new_solution in new_solutions_list:
        add_or_update_solution(solutions_dict, new_solution)
    print(f"Added/updated {len(new_solutions_list)} solutions to the dictionary.")

def save_solutions_dict(solutions_dict, filename, save_path):
    """Saves the solutions dictionary to a JSON file."""
    full_path = os.path.join(save_path,filename)
    with open(full_path, 'w') as f:
        json.dump(solutions_dict, f, indent=2, default=convert_datetime_to_str)
    print(f"Solutions dictionary saved to {filename}")
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
            if 'Actual_Date' in solution and isinstance(solution['Actual_Date'], str):
                solution['Actual_Date'] = datetime.fromisoformat(solution['Actual_Date'])
            if from_date is None or (isinstance(solution['Actual_Date'], datetime) and solution['Actual_Date'].replace(tzinfo=None) >= from_date.replace(tzinfo=None)):
                 filtered_solutions_dict[topic_id] = solution


        print(f"Solutions dictionary loaded from {filename}")
        if from_date:
             print(f"Filtered to include solutions from {from_date.strftime('%Y-%m-%d')} onwards.")
        return filtered_solutions_dict
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return {}
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filename}")
        return {}
    except ValueError as e:
        print(f"Error converting datetime string: {e}")
        return solutions_dict # Return partially loaded data if conversion fails for some entries

def get_end_date_from_solutions(solutions_dict):
    """Extracts the latest Actual_Date from the solutions dictionary."""
    if not solutions_dict:
        return None
    latest_date = max([pd.Timestamp(s['Actual_Date']) for s in solutions_dict.values()]).normalize() + pd.Timedelta(days=1)
    return latest_date.tz_convert( tz=None)#.tz_localize(timezone.utc)
