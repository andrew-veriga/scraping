from dotenv import load_dotenv

load_dotenv() 

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.services.processing_hierarchical import process_first_batch_hierarchical
from app.utils.file_utils import load_solutions_dict
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


@app.get("/health")
def health_check():
    """Health check endpoint with database status."""
    try:
        db_service = get_database_service()
        db_health = db_service.health_check()
        
        return {
            "status": "healthy" if db_health["status"] == "healthy" else "unhealthy",
            "database": db_health,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@app.get("/pool-status")
def pool_status():
    """Get detailed connection pool status."""
    try:
        db_service = get_database_service()
        pool_status = db_service.get_pool_status()
        
        return {
            "pool_status": pool_status,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@app.post("/cleanup-connections")
def cleanup_connections():
    """Force cleanup of database connections."""
    try:
        db_service = get_database_service()
        db_service.cleanup_connections()
        
        return {
            "status": "success",
            "message": "Connections cleaned up successfully",
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@app.post("/warmup-pool")
def warmup_pool():
    """Warm up the connection pool."""
    try:
        db_service = get_database_service()
        connections_created = db_service.warmup_pool()
        
        return {
            "status": "success",
            "message": f"Pool warmup completed. Created {connections_created} connections.",
            "connections_created": connections_created,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": pd.Timestamp.now().isoformat()
        }

@app.post("/full-process")
def full_process_endpoint():
    return processing.full_process(config)

@app.post("/process-first-batch")
def process_first_batch_endpoint():
    return processing.process_first_batch(config)

@app.post("/process-next-batches")
def process_next_batches_endpoint():
    return processing.process_next_batches(config)

@app.post("/process-first-batch-hierarchical")
def process_first_batch_hierarchical_endpoint():
    """
    Process the first batch of Discord messages with hierarchical parent-child relationships.
    
    This enhanced endpoint:
    - Loads Discord messages from Excel file
    - Analyzes parent-child relationships from 'Referenced Message ID' 
    - Creates hierarchical message structure in database
    - Returns detailed statistics and validation results
    """
    try:
        return process_first_batch_hierarchical(config)
    except Exception as e:
        logging.error(f"Error in hierarchical first batch processing: {e}", exc_info=True)
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

def generate_html_template(html_body: str) -> str:
    """
    Generates a complete HTML document with embedded CSS styles for the solutions report.
    
    Args:
        html_body: The rendered HTML body content from markdown
        
    Returns:
        Complete HTML document as a string
    """
    return f"""
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


def iterate_final_threads(thread_data, db_service):
    """
    Iterates over the final threads and yields formatted markdown output.
    """
    # Use a single database session for all message lookups to avoid connection pool exhaustion
    with db_service.get_session() as session:
        for i, thread in enumerate(thread_data):
            actual_date = thread.get('actual_date')
            header = thread.get('header') 
            markdown_output = f"""## {i}. {actual_date} {header}

#### Whole Thread Messages:
"""
            whole_thread = thread.get('whole_thread', [])
            messages = []
            if whole_thread:
                whole_thread_ids = [t['message_id'] for t in whole_thread]
                for message_id in whole_thread_ids:

                    message_content = illustrated_message(message_id, db_service, session)
                    topic_id = thread.get('topic_id') or thread.get('Topic_ID', 'N/A')
                    answer_id = thread.get('answer_id')
                    if message_id == topic_id:
                        messages.append(f"- ({message_id}) - **Topic started** :{message_content} ")
                    elif message_id == answer_id:
                        messages.append(f"- ({message_id}) **Answer** : {message_content}")
                    else:
                        messages.append(f"- ({message_id}) {message_content} ")
                markdown_output += "\n".join(messages)
            else:
                markdown_output += "\n  N/A"
            label = thread.get('label') or thread.get('label', 'N/A')
            solution = thread.get('solution') or thread.get('solution','')
            if label == SolutionStatus.UNRESOLVED:
                markdown_output += f"\n\n### solution {label}"
            else:
                markdown_output += f"\n\n### solution {label}: {solution}"
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

        db_service = get_database_service()
        

        report_sections = list(iterate_final_threads(solutions_dict.values(), db_service))
        full_report_md = "\n\n---\n\n".join(report_sections)

        output_file = os.path.join(SAVE_PATH, "solutions_report.md")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_report_md)
        logging.info(f"Markdown report generated and saved to {output_file}")
        return generate_html_report_from_existing_markdown_file(output_file)
    


    except Exception as e:
        logging.error(f"Error generating markdown report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating markdown report: {str(e)}")

@app.get("/exisiting-markdown-report", response_class=HTMLResponse)
def generate_html_report_from_existing_markdown_file(report_md_file: str=".\\results\\solutions_report.md") -> HTMLResponse:
    """
    Generates an HTML report from a Markdown file.
    """
    try:
        # file_path = os.path.join(SAVE_PATH, report_md_file)
        with open(report_md_file, 'r', encoding='utf-8') as f:
            full_report_md = f.read()
        md = MarkdownIt()
        html_body = md.render(full_report_md)
        html_content = generate_html_template(html_body)
        return HTMLResponse(content=html_content)
    except Exception as e:
        logging.error(f"Error generating HTML report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating HTML report: {str(e)}")
    