from dotenv import load_dotenv

load_dotenv() 

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.services import data_loader
from app.utils.file_utils import load_solutions_dict
from app.utils.analytics import get_analytics_service
from app.services.database import get_database_service
from app.models.pydantic_models import SolutionStatus

from app.services.processing_tracker import get_processing_tracker
from app.models.db_models import Solution, Message, Thread
import pandas as pd
import yaml
import json
import os
import logging
from markdown_it import MarkdownIt
from app.utils.file_utils import illustrated_message, illustrated_threads
from app import processing, monitoring

with open("configs/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

SAVE_PATH = config['SAVE_PATH']
os.makedirs(SAVE_PATH, exist_ok=True)
LOG_FILE = os.path.join(SAVE_PATH, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True,  
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

app = FastAPI()
app.include_router(monitoring.router, prefix="/monitoring")

logging.info(f"Config loaded: {config}")
MESSAGES_FILE_PATH = config['MESSAGES_FILE_PATH']
SOLUTIONS_DICT_FILENAME = config['SOLUTIONS_DICT_FILENAME']


@app.post("/full-process")
def full_process_endpoint():
    return processing.full_process(config)

@app.post("/process-first-batch")
def process_first_batch_endpoint():
    return processing.process_first_batch(config)

@app.post("/process-next-batches")
def process_next_batches_endpoint():
    return processing.process_next_batches(config)


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
    for i, thread in enumerate(thread_data):
        actual_date = thread.get('actual_date') or thread.get('Actual_Date', 'no date')
        header = thread.get('header') or thread.get('Header', 'N/A')
        markdown_output = f"""## {i}. {actual_date} {header}

#### Whole Thread Messages:
"""
        whole_thread = thread.get('whole_thread', [])
        messages = []
        if whole_thread:
            whole_thread_ids = [t['message_id'] for t in whole_thread]
            for message_id in whole_thread_ids:

                message_content = illustrated_message(message_id, messages_df)
                topic_id = thread.get('topic_id') or thread.get('Topic_ID', 'N/A')
                answer_id = thread.get('answer_id') or thread.get('Answer_ID', 'N/A')
                if message_id == topic_id:
                    messages.append(f"- ({message_id}) - **Topic started** :{message_content} ")
                elif message_id == answer_id:
                    messages.append(f"- ({message_id}) **Answer** : {message_content}")
                else:
                    messages.append(f"- ({message_id}) {message_content} ")
            markdown_output += "\n".join(messages)
        else:
            markdown_output += "\n  N/A"
        label = thread.get('label') or thread.get('Label', 'N/A')
        solution = thread.get('solution') or thread.get('Solution','')
        if label == SolutionStatus.UNRESOLVED:
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

        report_sections = list(iterate_final_threads(solutions_dict.values(), messages_df))
        full_report_md = "\n\n---\n\n".join(report_sections)

        output_file = os.path.join(SAVE_PATH, "solutions_report.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_report_md)
        logging.info(f"Markdown report generated and saved to {output_file}")

        md = MarkdownIt()
        html_body = md.render(full_report_md)

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