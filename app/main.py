from dotenv import load_dotenv

load_dotenv() # This loads variables from .env into the environment

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.services import data_loader, thread_processor
from app.utils.file_utils import load_solutions_dict, save_solutions_dict, get_end_date_from_solutions, create_dict_from_list
import pandas as pd
import yaml
import json
import os
import logging
from markdown_it import MarkdownIt
from app.utils.file_utils import illustrated_message, illustrated_threads #, save_solutions_dict, load_solutions_dict, create_dict_from_list, add_new_solutions_to_dict, add_or_update_solution, convert_datetime_to_str

with open("configs/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

SAVE_PATH = config['SAVE_PATH']
os.makedirs(SAVE_PATH, exist_ok=True)
LOG_FILE = os.path.join(SAVE_PATH, 'app.log')

# Configure logging to output to console and file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,  # In case uvicorn tries to configure logging as well
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

app = FastAPI()

logging.info(f"Config loaded: {config}")
MESSAGES_FILE_PATH = config['MESSAGES_FILE_PATH']
INTERVAL_FIRST = config['INTERVAL_FIRST']
INTERVAL_NEXT = config['INTERVAL_NEXT']
INTERVAL_BACK = config['INTERVAL_BACK']
SOLUTIONS_DICT_FILENAME = config['SOLUTIONS_DICT_FILENAME']


@app.post("/full-process")
def full_process():
    """
    Runs the full data processing pipeline:
    1. Processes the initial batch of messages to create a baseline solutions file.
    2. Processes all subsequent batches of messages until up-to-date.
    """
    try:
        logging.info("Starting full process...")
        # The called functions will return a dict on success and raise HTTPException on failure.
        # We are calling them as procedures and ignoring the return value.
        logging.info("Starting process_first_batch...")
        process_first_batch()
        logging.info("process_first_batch complete.")
        logging.info("Starting process_next_batch...")
        process_next_batch()
        logging.info("process_next_batch complete.")
        logging.info("Full process finished successfully.")
        return {"message": "Full process completed successfully."}
    except Exception as e:
        logging.error("An error occurred during the full process.", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during the full process: {getattr(e, 'detail', str(e))}")

@app.post("/process-first-batch")
def process_first_batch():
    try:
        logging.info(f"Processing first batch from {MESSAGES_FILE_PATH}")
        messages_df = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)
        start_date = messages_df['DateTime'].min().normalize()
        end_date = start_date + pd.Timedelta(days=INTERVAL_FIRST)
        str_interval = f"{start_date.date()}-{end_date.date()}"
        first_some_days_df = messages_df[messages_df['DateTime'] < end_date].copy()
        step1_output_filename = thread_processor.first_thread_gathering(first_some_days_df,  f"first_{str_interval}", SAVE_PATH)
        first_technical_filename = thread_processor.filter_technical_topics(step1_output_filename, f"first_{str_interval}", messages_df, SAVE_PATH) # messages_df is needed for illustrated_threads
        first_solutions_filename = thread_processor.generalization_solution(first_technical_filename,  f"first_{str_interval}", SAVE_PATH)
        solutions_list = json.load(open(first_solutions_filename, 'r'))
        solutions_dict = create_dict_from_list(solutions_list)
        save_solutions_dict(solutions_dict, SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        return {"message": "First batch processed successfully"}
    except Exception as e:
        logging.error("Error processing first batch", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def process_batch(solutions_dict, lookback_date:pd.Timestamp, next_start_date: pd.Timestamp, next_end_date: pd.Timestamp, messages_df):
    str_interval = f"{next_start_date.date()}-{next_end_date.date()}"
    next_batch_df = messages_df[(next_start_date <= messages_df['DateTime']) & (messages_df['DateTime'] < next_end_date)].copy()
    if next_batch_df.empty:
        logging.info(f"No messages found in the interval {str_interval}. Skipping this batch.")
        return False
    prev_threads = [t for t in solutions_dict.values() if pd.Timestamp(t['Actual_Date']) > lookback_date]
    next_step1_output_filename = thread_processor.next_thread_gathering(next_batch_df, prev_threads, str_interval, SAVE_PATH, messages_df)
    next_technical_filename = thread_processor.filter_technical_topics(next_step1_output_filename, f"next_{str_interval}", messages_df, SAVE_PATH)
    next_solutions_filename = thread_processor.generalization_solution(next_technical_filename, f"next_{str_interval}", SAVE_PATH)
    solutions_dict = thread_processor.new_solutions_revision_and_add(next_solutions_filename, next_technical_filename, solutions_dict, lookback_date)
    save_solutions_dict(solutions_dict, SOLUTIONS_DICT_FILENAME, save_path=SAVE_PATH)
    return True

@app.post("/process-next-batch")
def process_next_batch():
    try:
        logging.info(f"Processing next batch from {MESSAGES_FILE_PATH}")
        messages_df = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)
        solutions_dict = load_solutions_dict(SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        latest_solution_date = get_end_date_from_solutions(solutions_dict)

        if latest_solution_date:
            next_end_date = latest_solution_date #+ pd.Timedelta(days=1)
        else:
            next_end_date = messages_df['DateTime'].min().normalize() + pd.Timedelta(days=INTERVAL_FIRST)

        while True:
            next_start_date = next_end_date
            lookback_date = next_start_date - pd.Timedelta(days=INTERVAL_BACK)
            next_end_date = next_start_date + pd.Timedelta(days=INTERVAL_NEXT)

            if next_start_date > messages_df['DateTime'].max():
                break

            logging.info("-" * 60)
            logging.info(f"Processing batch. Lookback: {lookback_date}, Start: {next_start_date}, End: {next_end_date}")
            if not process_batch(solutions_dict, lookback_date, next_start_date, next_end_date, messages_df):
                logging.info(f"No messages in the interval: {next_start_date} to {next_end_date}")
        return {"message": "Next batch processed successfully"}
    except Exception as e:
        logging.error(f"Error processing next batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/solutions")
def get_solutions():
    """Retrieves the processed solutions dictionary."""
    try:
        solutions_dict = load_solutions_dict(SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        return solutions_dict
    except FileNotFoundError:
        logging.warning(f"Solutions file not found on request to /solutions endpoint.")
        raise HTTPException(status_code=404, detail="Solutions file not found. Please run a batch process first.")
    except Exception as e:
        logging.error("Error retrieving solutions from /solutions endpoint", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def iterate_final_threads(thread_data, messages_df):
    """
    Iterates over the final threads and yields formatted markdown output.
    """
    # Set 'Message ID' as the index for efficient message lookup.
    df_indexed = messages_df.set_index('Message ID')

    for i, thread in enumerate(thread_data):
        markdown_output = f"""## {i}. {thread.get('Actual_Date','no date')} {thread.get('Header', 'N/A')}

#### Whole Thread Messages:
"""
        whole_thread_ids = thread.get('Whole_thread', [])
        messages = []
        if whole_thread_ids:
            for message_id in whole_thread_ids:
                # Ensure message_id is a string for lookup
                # Assuming formatted_message function is defined and accessible
                message_content =illustrated_message(message_id,df_indexed)
                if message_id == thread.get('Topic_ID', 'N/A'):
                    messages.append(f"""- ({message_id}) - **Topic started** :{message_content} """)
                elif message_id == thread.get('Answer_ID', 'N/A'):
                    messages.append(f"- ({message_id}) **Answer** : {message_content}")
                else:
                    messages.append(f"- ({message_id}) {message_content} ")
            markdown_output += "\n".join(messages)
        else:
            markdown_output += "\n  N/A"
        label = thread.get('Label', 'N/A') # Check the label field
        solution = thread.get('Solution','')
        if label == 'unresolved':
            markdown_output += f"\n\n### Solution {label}"
        else:
            markdown_output += f"\n\n### Solution {label}: {solution}"
        yield markdown_output


@app.get("/markdown-report", response_class=HTMLResponse)
def generate_markdown_report():
    """
    Generates a markdown report from the solutions dictionary and returns it in the response.
    It also saves the report to a file.
    """
    try:
        solutions_dict = load_solutions_dict(SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        if not solutions_dict:
            raise HTTPException(status_code=404, detail="Solutions file not found or is empty. Please run a batch process first.")

        messages_df = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)

        # Generate the raw markdown report
        report_sections = list(iterate_final_threads(solutions_dict.values(), messages_df))
        full_report_md = "\n\n---\n\n".join(report_sections)

        # Save the raw markdown file as before
        output_file = os.path.join(SAVE_PATH, "solutions_report.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_report_md)
        logging.info(f"Markdown report generated and saved to {output_file}")

        # Convert markdown to HTML
        md = MarkdownIt()
        html_body = md.render(full_report_md)

        # Embed the HTML in a full page with some basic styling for a "pretty" look
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Solutions Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
            line-height: 1.5;
            color: #24292e;
            background-color: #fff;
            padding: 20px 45px;
            max-width: 800px;
            margin: 0 auto;
        }}
        h1, h2, h3, h4 {{ border-bottom: 1px solid #eaecef; padding-bottom: .3em; }}
        code {{ background-color: rgba(27,31,35,.05); border-radius: 3px; padding: .2em .4em; }}
        ul {{ padding-left: 2em; }}
    </style>
</head>
<body>
    <h1>Solutions Report</h1>
    {html_body}
</body>
</html>
"""
        return HTMLResponse(content=html_content)
    except Exception as e:
        logging.error(f"Error generating markdown report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating markdown report: {str(e)}")
