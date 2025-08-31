import json
from datetime import datetime, timezone
import os
import pandas as pd
from app.services import gemini_service
from app.utils.file_utils import save_solutions_dict, load_solutions_dict, create_dict_from_list, add_or_update_solution, add_new_solutions_to_dict

def first_thread_gathering(logs_df, save_path):
    """
    Group messages into threads using LLM from DataFrame provided flat message list
    """
    logs_csv = logs_df.to_csv(index=False)
    response = gemini_service.generate_content(
        contents=[
            logs_csv,
            gemini_service.system_prompt,
            gemini_service.prompt_start_step_1
            ],
        config=gemini_service.config_step1
    )
    output_filename = f'first_group_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'

    def convert_datetime_to_str(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError("Type not serializable")

    threads_list_dict = [thread.model_dump() for thread in response.parsed.threads]

    full_path = os.path.join(save_path, output_filename)
    with open(full_path, 'w') as f:
        json.dump(threads_list_dict, f, indent=4, default=convert_datetime_to_str)
    print(f"Start step 1. Processed {len(logs_df)} messages")
    print(f"Successfully saved {len(response.parsed.threads)} grouped threads to {output_filename}")
    return full_path

def filter_technical_topics(filename, startnext: str, messages_df, save_path):
    """
    Extract technical topics from JSON file using LLM with DataFrame logs provided full texts of messages
    return only technical threads
    """
    with open(filename, 'r') as f:
        threads_json_data = json.load(f)

    processed_threads = illustrated_threads(threads_json_data, messages_df)

    response = gemini_service.generate_content(
        model=gemini_service.model_name,
        contents=[
            json.dumps(processed_threads, indent=4),
            gemini_service.system_prompt,
            gemini_service.prompt_step_2
            ],
        config=gemini_service.config_step2
        )

    technical_threads = [thread for thread in processed_threads if thread['Topic_ID'] in response.parsed.technical_topics]
    print (f"{startnext} step 2. Selected {len(response.parsed.technical_topics)} technical topics from {len(processed_threads)} threads")

    output_filename = f'{startnext}_technical_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        f.write(json.dumps(technical_threads,  indent=2))

    print(f"Successfully saved {len(technical_threads)} filtered threads to {output_filename}")
    return full_path

def generalization_solution(filename,startnext: str, save_path):
    """
    Generalize technical topics from JSON file using LLM
    return only technical threads
    """
    with open(filename, 'r') as f:
        technical_threads = json.load(f)

    response_solutions = gemini_service.generate_content(
        model=gemini_service.model_name,
        contents=[
            json.dumps(technical_threads, indent=2),
            gemini_service.system_prompt,
            gemini_service.prompt_step_3
            ],
        config=gemini_service.solution_config
        )
    print(f"{startnext} step 3. Generalization of {len(response_solutions.parsed.threads)} technical threads")
    solutions_list = [thread.model_dump() for thread in response_solutions.parsed.threads]

    def convert_datetime_to_str(obj):
        """
        convert datetime objects to strings
        """

        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError("Type not serializable")


    output_filename = f'{startnext}_solutions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        json.dump(solutions_list, f, indent=2, default=convert_datetime_to_str)
    print(f"Successfully saved {len(solutions_list)} threads to {output_filename}")
    return full_path

def next_thread_gathering(next_bach_df, solutions_dict, lookback_date, next_start_date, save_path, messages_df):
    added_threads = {}
    previous_threads_json =[]

    def debug_print_modified(): #for debugging purposes
        existing_solutions = previous_threads_json

        flat_list = []
        for t in existing_solutions:
            flat_list.extend(t['Whole_thread_formatted'])

        exisiting_messages = [t['Message_ID'] for t in flat_list]

        list_last_topics = [ t['Topic_ID'] for t in existing_solutions]
        replies = [t.Topic_ID for t in added_threads if t.Topic_ID in list_last_topics ]

        existing_dict = {t['Topic_ID']: t for t in existing_solutions}

        new_dict = { t['Topic_ID']: t for t in illustrated_threads([tt.model_dump() for tt  in added_threads], messages_df)}
        print('replies:')
        for id in replies:
            ex_mes = [t['Message_ID'] for t in existing_dict[id]['Whole_thread_formatted']]
            new_mes = [t['Message_ID'] for t in new_dict[id]['Whole_thread_formatted']]
            print(id, ":",new_dict[id]['status'], [m for m in ex_mes if m not in new_mes], [m for m in new_mes if m not in ex_mes])

    def convert_datetime_to_str(obj):
        """
        convert datetime objects to strings
        """

        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError("Type not serializable")

    next_bach_csv = next_bach_df.to_csv(index=False)

    technical_threads_json = [t for t in solutions_dict.values() if pd.Timestamp(t['Actual_Date']) > lookback_date.tz_localize(timezone.utc)]
    previous_threads_json = illustrated_threads(technical_threads_json, messages_df)

    prmpt = gemini_service.prompt_addition_step1.format(JSON_prev=json.dumps(previous_threads_json, indent=2))

    print(f"Next step 1. Processing next {len(next_bach_df)} raw messages...")
    response = gemini_service.generate_content(
        model=gemini_service.model_name,
        contents=[
            gemini_service.system_prompt,
            next_bach_csv,
            prmpt
            ],
        config=gemini_service.config_addition_step1
        )
    added_threads = [t for t in response.parsed.threads if t.status !='persisted']
    cnt_modified = len([t for t in added_threads if t.status=='modified'])
    cnt_new = len([t for t in added_threads if t.status=='new'])

    print(f"Added {cnt_new} new threads. Modified {cnt_modified} threads created before {next_start_date.strftime("%Y-%m-%d_%H-%M-%S")}.