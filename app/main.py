from fastapi import FastAPI, HTTPException
from app.services import data_loader, thread_processor
from app.utils.file_utils import load_solutions_dict, save_solutions_dict, get_end_date_from_solutions
import pandas as pd
import yaml
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from google.api_core.exceptions import ServiceUnavailable
import json

app = FastAPI()

with open("configs/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

SAVE_PATH = config['SAVE_PATH']
INTERVAL_FIRST = config['INTERVAL_FIRST']
INTERVAL_NEXT = config['INTERVAL_NEXT']
INTERVAL_BACK = config['INTERVAL_BACK']

def create_dict_from_list(solutions_list):
    solutions_dict = {}
    for solution in solutions_list:
        solutions_dict[solution['solution_id']] = solution
    return solutions_dict

@app.post("/process-first-batch")
def process_first_batch():
    try:
        messages_df = data_loader.load_and_preprocess_data("c:\\Users\\LiveComp\\Documents\\Лекции в Peeranha\\Discord messages\\discord_messages.xlsx")
        start_date = messages_df['DateTime'].min().normalize()
        end_date = start_date + pd.Timedelta(days=INTERVAL_FIRST)
        first_some_days_df = messages_df[messages_df['DateTime'] < end_date].copy()
        step1_output_filename = thread_processor.first_thread_gathering(first_some_days_df, SAVE_PATH)
        first_technical_filename = thread_processor.filter_technical_topics(step1_output_filename, "first", messages_df, SAVE_PATH)
        first_solutions_filename = thread_processor.generalization_solution(first_technical_filename, "first", SAVE_PATH)
        solutions_list = json.load(open(first_solutions_filename, 'r'))
        solutions_dict = create_dict_from_list(solutions_list)
        save_solutions_dict(solutions_dict, 'solutions_dict.json', SAVE_PATH)
        return {"message": "First batch processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@retry(stop=stop_after_attempt(5), wait=wait_fixed(60), retry=retry_if_exception_type(ServiceUnavailable))
def process_batch(next_day_df, solutions_dict, lookback_date, next_start_date, next_end_date, messages_df):
    next_step1_output_filename = thread_processor.next_thread_gathering(next_day_df, solutions_dict, lookback_date, next_start_date, SAVE_PATH, messages_df)
    next_technical_filename = thread_processor.filter_technical_topics(next_step1_output_filename, "next", messages_df, SAVE_PATH)
    next_solutions_filename = thread_processor.generalization_solution(next_technical_filename, "next", SAVE_PATH)
    thread_processor.new_solutions_revision_and_add(next_solutions_filename,next_technical_filename, SAVE_PATH, lookback_date)

@app.post("/process-next-batch")
def process_next_batch():
    try:
        messages_df = data_loader.load_and_preprocess_data("c:\\Users\\LiveComp\\Documents\\Лекции в Peeranha\\Discord messages\\discord_messages.xlsx")
        solutions_dict = load_solutions_dict('solutions_dict.json', SAVE_PATH)
        latest_solution_date = get_end_date_from_solutions(solutions_dict)

        if latest_solution_date:
            next_end_date = latest_solution_date + pd.Timedelta(days=1)
        else:
            next_end_date = messages_df['DateTime'].min().normalize() + pd.Timedelta(days=INTERVAL_FIRST)

        while True:
            next_start_date = next_end_date
            lookback_date = next_start_date - pd.Timedelta(days=INTERVAL_BACK)
            next_end_date = next_start_date + pd.Timedelta(days=INTERVAL_NEXT)

            if next_start_date.tz_localize('UTC') > messages_df['DateTime'].max().tz_localize('UTC'):
                break

            print("-"*60)
            print(lookback_date,next_start_date, next_end_date)
            next_day_df = messages_df[(next_start_date <= messages_df['DateTime']) & (messages_df['DateTime'] < next_end_date)].copy()
            if not next_day_df.empty:
                process_batch(next_day_df, solutions_dict, lookback_date, next_start_date, next_end_date, messages_df)
            else:
                print(f"No messages in the interval: {next_start_date} to {next_end_date}")
        return {"message": "Next batch processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))