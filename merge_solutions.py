import os
import json
from app.utils.file_utils import save_solutions_dict,create_dict_from_list
from datetime import datetime
import yaml

def merge_solutions():
    base_path = '.\\results'
    solutions_dict = {}
    for filename in os.listdir(base_path):
        if filename.endswith('_solutions.json'):
            with open(os.path.join(base_path, filename), 'r') as f:
                next_solutions_list = json.load(f)
            new_solution_dict = create_dict_from_list(next_solutions_list)
            solutions_dict.update(new_solution_dict)
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    save_solutions_dict(solutions_dict, config)

def main():
    merge_solutions()

if __name__ == "__main__":
    main()