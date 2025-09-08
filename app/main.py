from dotenv import load_dotenv

load_dotenv() # This loads variables from .env into the environment

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.services import data_loader, thread_processor
from app.utils.file_utils import load_solutions_dict, save_solutions_dict, get_end_date_from_solutions, create_dict_from_list
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
        
        # Load messages into database
        messages_loaded = data_loader.load_messages_to_database(messages_df)
        logging.info(f"Loaded {messages_loaded} messages into database")
        
        start_date = messages_df['DateTime'].min().normalize()
        end_date = start_date + pd.Timedelta(days=INTERVAL_FIRST)
        str_interval = f"{start_date.date()}-{end_date.date()}"
        
        # Create processing batch record
        db_service = get_database_service()
        with db_service.get_session() as session:
            batch_data = {
                'batch_type': 'first_batch',
                'start_date': start_date,
                'end_date': end_date,
                'messages_processed': len(messages_df)
            }
            processing_batch = db_service.create_processing_batch(session, batch_data)
            session.commit()
            batch_id = processing_batch.id
            logging.info(f"Created processing batch record with ID: {batch_id}")
        
        first_some_days_df = messages_df[messages_df['DateTime'] < end_date].copy()
        # TODO first_some_days_df - pd.DataFrame
        # надо заменить на получение этого списка из SQL
        step1_output_filename = thread_processor.first_thread_gathering(first_some_days_df,  f"first_{str_interval}", SAVE_PATH)
        first_technical_filename = thread_processor.filter_technical_topics(step1_output_filename, f"first_{str_interval}", messages_df, SAVE_PATH) # messages_df is needed for illustrated_threads
        first_solutions_filename = thread_processor.generalization_solution(first_technical_filename,  f"first_{str_interval}", SAVE_PATH)
        solutions_list = json.load(open(first_solutions_filename, 'r'))
        first_solutions_dict = create_dict_from_list(solutions_list)
        solutions_dict = thread_processor.check_in_rag_and_save(first_solutions_dict)
        save_solutions_dict(solutions_dict, SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        
        # Update database with solutions
        thread_processor._update_database_with_solutions(solutions_dict)
        
        # Complete processing batch
        with db_service.get_session() as session:
            batch_stats = {
                'threads_created': len(solutions_dict),
                'technical_threads': len([s for s in solutions_dict.values() if s.get('label') != SolutionStatus.UNRESOLVED]),
                'solutions_added': len([s for s in solutions_dict.values() if s.get('solution')])
            }
            db_service.complete_processing_batch(session, batch_id, batch_stats)
            session.commit()
            logging.info(f"Completed processing batch {batch_id} with stats: {batch_stats}")
        
        return {"message": "First batch processed successfully", "database_messages_loaded": messages_loaded, "solutions_created": len(solutions_dict)}
    except Exception as e:
        logging.error("Error processing first batch", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def process_batch(solutions_dict, lookback_date:pd.Timestamp, next_start_date: pd.Timestamp, next_end_date: pd.Timestamp, messages_df):
    str_interval = f"{next_start_date.date()}-{next_end_date.date()}"
    next_batch_df = messages_df[(next_start_date <= messages_df['DateTime']) & (messages_df['DateTime'] < next_end_date)].copy()
    if next_batch_df.empty:
        logging.info(f"No messages found in the interval {str_interval}. Skipping this batch.")
        return False
    
    # Create processing batch record
    db_service = get_database_service()
    with db_service.get_session() as session:
        batch_data = {
            'batch_type': 'incremental_batch',
            'start_date': next_start_date,
            'end_date': next_end_date,
            'lookback_date': lookback_date,
            'messages_processed': len(next_batch_df)
        }
        processing_batch = db_service.create_processing_batch(session, batch_data)
        session.commit()
        batch_id = processing_batch.id
        logging.info(f"Created incremental processing batch record with ID: {batch_id}")

    prev_threads = [t for t in solutions_dict.values() if pd.Timestamp(t.get('actual_date') or t.get('Actual_Date')) > lookback_date]
    next_step1_output_filename = thread_processor.next_thread_gathering(next_batch_df, prev_threads, str_interval, SAVE_PATH, messages_df)
    next_technical_filename = thread_processor.filter_technical_topics(next_step1_output_filename, f"next_{str_interval}", messages_df, SAVE_PATH)
    next_solutions_filename = thread_processor.generalization_solution(next_technical_filename, f"next_{str_interval}", SAVE_PATH)

    adding_solutions_dict = thread_processor.new_solutions_revision_and_add(next_solutions_filename, next_technical_filename, solutions_dict, lookback_date)

    # Count existing solutions before processing
    
    initial_solution_count = len(solutions_dict)
    
    solutions_dict = thread_processor.check_in_rag_and_save(solutions_dict, adding_solutions_dict)
    
    save_solutions_dict(solutions_dict, SOLUTIONS_DICT_FILENAME, save_path=SAVE_PATH)
    
    # Update database with solutions
    thread_processor._update_database_with_solutions(solutions_dict)
    
    # Complete processing batch
    with db_service.get_session() as session:
        new_solution_count = len(solutions_dict) - initial_solution_count
        #TODO: Я думаю, что посчитать количество измененных топиков здесь не получится, так как все изменения видны только внутри thread_processor.new_solutions_revision_and_add
        # можно там вычислять batch_stats и возвращать его сюда как solutions_dict, batch_stats = thread_processor.new_solutions_revision_and_add

        batch_stats = {
            'threads_created': new_solution_count,
            'threads_modified': len(solutions_dict) - new_solution_count,  # Approximate
            'technical_threads': len([s for s in solutions_dict.values() if s.get('label') != SolutionStatus.UNRESOLVED]),
            'solutions_added': len([s for s in solutions_dict.values() if s.get('solution')])
        }
        db_service.complete_processing_batch(session, batch_id, batch_stats)
        session.commit()
        logging.info(f"Completed processing batch {batch_id} with stats: {batch_stats}")
    
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
                # Ensure message_id is a string for lookup
                # Assuming formatted_message function is defined and accessible
                message_content = illustrated_message(message_id, messages_df)
                topic_id = thread.get('topic_id') or thread.get('Topic_ID', 'N/A')
                answer_id = thread.get('answer_id') or thread.get('Answer_ID', 'N/A')
                if message_id == topic_id:
                    messages.append(f"""- ({message_id}) - **Topic started** :{message_content} """)
                elif message_id == answer_id:
                    messages.append(f"- ({message_id}) **Answer** : {message_content}")
                else:
                    messages.append(f"- ({message_id}) {message_content} ")
            markdown_output += "\n".join(messages)
        else:
            markdown_output += "\n  N/A"
        label = thread.get('label') or thread.get('Label', 'N/A') # Check the label field
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


# Admin endpoints for duplicate management
@app.get("/admin/duplicates")
def get_pending_duplicates(limit: int = 50, offset: int = 0):
    """Get duplicates pending admin review."""
    try:
        analytics_service = get_analytics_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            pending_duplicates = analytics_service.get_pending_duplicates_for_review(session, limit)
            return {
                "pending_duplicates": pending_duplicates,
                "total_count": len(pending_duplicates),
                "limit": limit,
                "offset": offset
            }
    except Exception as e:
        logging.error(f"Error fetching pending duplicates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/duplicates/{solution_id}")
def get_solution_duplicates(solution_id: int):
    """Get duplicate chain for a specific solution."""
    try:
        analytics_service = get_analytics_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            duplicate_chain = analytics_service.get_solution_duplicate_chain(session, solution_id)
            return duplicate_chain
    except Exception as e:
        logging.error(f"Error fetching solution duplicates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/admin/duplicates/{duplicate_id}/review")
def review_duplicate(duplicate_id: int, status: str, reviewed_by: str, notes: str = None):
    """Review and update duplicate status."""
    try:
        # Validate status
        valid_statuses = ['confirmed_duplicate', 'false_positive', 'pending_review']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            success = db_service.update_duplicate_status(session, duplicate_id, status, reviewed_by, notes)
            
            if not success:
                raise HTTPException(status_code=404, detail="Duplicate record not found")
            
            session.commit()
            
            return {
                "duplicate_id": duplicate_id,
                "status": status,
                "reviewed_by": reviewed_by,
                "notes": notes,
                "message": "Duplicate status updated successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating duplicate status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/frequency-analysis")
def get_frequency_analysis(limit: int = 20, min_duplicates: int = 2):
    """Get problem frequency analysis."""
    try:
        analytics_service = get_analytics_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            frequent_problems = analytics_service.get_frequent_problems(session, limit, min_duplicates)
            duplicate_stats = analytics_service.get_duplicate_statistics(session)
            trends = analytics_service.get_duplicate_trends(session, days_back=30)
            categories = analytics_service.get_problem_categories_analysis(session)
            
            return {
                "frequent_problems": frequent_problems,
                "statistics": duplicate_stats,
                "trends": trends,
                "categories": categories
            }
    except Exception as e:
        logging.error(f"Error fetching frequency analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/solutions/{topic_id}/duplicates")
def get_topic_duplicates(topic_id: str):
    """Get all duplicates of a specific solution by topic ID."""
    try:
        db_service = get_database_service()
        analytics_service = get_analytics_service()
        
        with db_service.get_session() as session:
            # First get the thread and solution
            thread = db_service.get_thread_by_topic_id(session, topic_id)
            if not thread:
                raise HTTPException(status_code=404, detail=f"Thread with topic_id {topic_id} not found")
            
            solution = session.query(Solution).filter(Solution.thread_id == thread.id).first()
            if not solution:
                raise HTTPException(status_code=404, detail=f"Solution for topic_id {topic_id} not found")
            
            duplicate_chain = analytics_service.get_solution_duplicate_chain(session, solution.id)
            return duplicate_chain
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching topic duplicates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/duplicate-statistics")
def get_duplicate_statistics():
    """Get comprehensive duplicate statistics for admin dashboard."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            stats = db_service.get_duplicate_statistics(session)
            return stats
    except Exception as e:
        logging.error(f"Error fetching duplicate statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/recalculate-duplicate-counts")
def recalculate_duplicate_counts():
    """Recalculate duplicate counts for all solutions."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            updated_count = db_service.bulk_update_duplicate_counts(session)
            session.commit()
            
            return {
                "message": f"Successfully updated duplicate counts for {updated_count} solutions"
            }
    except Exception as e:
        logging.error(f"Error recalculating duplicate counts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# Processing transparency endpoints
@app.get("/processing/messages/{message_id}")
def get_message_processing_history(message_id: str):
    """Get complete processing history for a message by message_id."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # First find the message by message_id
            message = session.query(Message).filter(Message.message_id == message_id).first()
            if not message:
                raise HTTPException(status_code=404, detail=f"Message with ID {message_id} not found")
            
            processing_history = processing_tracker.get_message_processing_history(session, message.id)
            return processing_history
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching message processing history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing/threads/{topic_id}")
def get_thread_processing_history(topic_id: str):
    """Get complete processing history for a thread by topic_id."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # First find the thread by topic_id
            thread = session.query(Thread).filter(Thread.topic_id == topic_id).first()
            if not thread:
                raise HTTPException(status_code=404, detail=f"Thread with topic_id {topic_id} not found")
            
            processing_history = processing_tracker.get_thread_processing_history(session, thread.id)
            return processing_history
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching thread processing history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing/pipeline")
def get_processing_pipeline():
    """Get current processing pipeline configuration."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # Get active pipeline steps
            from app.models.db_models import ProcessingPipeline
            
            pipeline_steps = session.query(ProcessingPipeline).filter(
                ProcessingPipeline.is_active == True
            ).order_by(ProcessingPipeline.step_order).all()
            
            pipeline = {
                'pipeline_name': 'default',
                'steps': [
                    {
                        'order': step.step_order,
                        'name': step.step_name,
                        'config': step.step_config,
                        'created_at': step.created_at.isoformat()
                    }
                    for step in pipeline_steps
                ]
            }
            
            return pipeline
    except Exception as e:
        logging.error(f"Error fetching processing pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/processing/pipeline")
def create_processing_pipeline(pipeline_data: dict):
    """Create or update processing pipeline configuration."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        pipeline_name = pipeline_data.get('pipeline_name', 'default')
        steps = pipeline_data.get('steps', [])
        
        if not steps:
            raise HTTPException(status_code=400, detail="Steps are required")
        
        with db_service.get_session() as session:
            success = processing_tracker.create_processing_pipeline(session, pipeline_name, steps)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to create processing pipeline")
            
            session.commit()
            
            return {
                "message": f"Processing pipeline '{pipeline_name}' created successfully",
                "pipeline_name": pipeline_name,
                "steps_count": len(steps)
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating processing pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing/messages/{message_id}/annotations")
def get_message_annotations(message_id: str):
    """Get all annotations for a message."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # First find the message by message_id
            message = session.query(Message).filter(Message.message_id == message_id).first()
            if not message:
                raise HTTPException(status_code=404, detail=f"Message with ID {message_id} not found")
            
            from app.models.db_models import MessageAnnotation
            annotations = session.query(MessageAnnotation).filter(
                MessageAnnotation.message_id == message.id
            ).order_by(MessageAnnotation.annotated_at).all()
            
            return {
                'message_id': message_id,
                'annotations': [
                    {
                        'id': ann.id,
                        'type': ann.annotation_type,
                        'value': ann.annotation_value,
                        'confidence_score': float(ann.confidence_score) if ann.confidence_score else None,
                        'annotated_by': ann.annotated_by,
                        'annotated_at': ann.annotated_at.isoformat()
                    }
                    for ann in annotations
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching message annotations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/processing/messages/{message_id}/annotate")
def annotate_message(message_id: str, annotation_data: dict):
    """Add manual annotation to a message."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        annotation_type = annotation_data.get('type')
        annotation_value = annotation_data.get('value')
        confidence_score = annotation_data.get('confidence_score')
        annotated_by = annotation_data.get('annotated_by', 'manual')
        
        if not annotation_type:
            raise HTTPException(status_code=400, detail="Annotation type is required")
        
        with db_service.get_session() as session:
            # First find the message by message_id
            message = session.query(Message).filter(Message.message_id == message_id).first()
            if not message:
                raise HTTPException(status_code=404, detail=f"Message with ID {message_id} not found")
            
            annotation = processing_tracker.annotate_message(
                session=session,
                message_id=message.id,
                annotation_type=annotation_type,
                annotation_value=annotation_value,
                confidence_score=confidence_score,
                annotated_by=annotated_by
            )
            
            if not annotation:
                raise HTTPException(status_code=500, detail="Failed to create annotation")
            
            session.commit()
            
            return {
                "message": "Annotation created successfully",
                "annotation_id": annotation.id,
                "message_id": message_id,
                "type": annotation_type
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating message annotation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing/annotations/search")
def search_annotations(annotation_type: str = None, annotated_by: str = None, limit: int = 100):
    """Search messages by annotation type or annotated_by."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            from app.models.db_models import MessageAnnotation
            
            query = session.query(MessageAnnotation, Message).join(
                Message, MessageAnnotation.message_id == Message.id
            )
            
            if annotation_type:
                query = query.filter(MessageAnnotation.annotation_type == annotation_type)
            
            if annotated_by:
                query = query.filter(MessageAnnotation.annotated_by == annotated_by)
            
            results = query.order_by(MessageAnnotation.annotated_at.desc()).limit(limit).all()
            
            return {
                'results': [
                    {
                        'annotation_id': ann.id,
                        'message_id': msg.message_id,
                        'annotation_type': ann.annotation_type,
                        'annotation_value': ann.annotation_value,
                        'confidence_score': float(ann.confidence_score) if ann.confidence_score else None,
                        'annotated_by': ann.annotated_by,
                        'annotated_at': ann.annotated_at.isoformat(),
                        'message_content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,
                        'message_datetime': msg.datetime.isoformat()
                    }
                    for ann, msg in results
                ]
            }
    except Exception as e:
        logging.error(f"Error searching annotations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/processing/statistics")
def get_processing_statistics():
    """Get comprehensive processing statistics."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            stats = processing_tracker.get_processing_statistics(session)
            return stats
    except Exception as e:
        logging.error(f"Error fetching processing statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/processing/{date}")
def get_processing_audit_log(date: str):
    """Get processing audit log for a specific date (YYYY-MM-DD)."""
    try:
        from datetime import datetime, timedelta
        
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            from app.models.db_models import MessageProcessing
            
            # Get processing steps for the specified date
            start_datetime = datetime.combine(target_date, datetime.min.time())
            end_datetime = start_datetime + timedelta(days=1)
            
            processing_steps = session.query(MessageProcessing, Message).join(
                Message, MessageProcessing.message_id == Message.id
            ).filter(
                MessageProcessing.processed_at >= start_datetime,
                MessageProcessing.processed_at < end_datetime
            ).order_by(MessageProcessing.processed_at).all()
            
            return {
                'date': date,
                'total_processing_steps': len(processing_steps),
                'processing_steps': [
                    {
                        'message_id': msg.message_id,
                        'processing_step': ps.processing_step,
                        'step_order': ps.step_order,
                        'result': ps.result,
                        'confidence_score': float(ps.confidence_score) if ps.confidence_score else None,
                        'processed_at': ps.processed_at.isoformat(),
                        'processing_version': ps.processing_version
                    }
                    for ps, msg in processing_steps
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching processing audit log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audit/confidence/low")
def get_low_confidence_classifications(threshold: float = 0.7, limit: int = 50):
    """Get classifications with low confidence scores for review."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            from app.models.db_models import MessageProcessing, MessageAnnotation
            
            # Get low confidence processing steps
            low_confidence_processing = session.query(MessageProcessing, Message).join(
                Message, MessageProcessing.message_id == Message.id
            ).filter(
                MessageProcessing.confidence_score.isnot(None),
                MessageProcessing.confidence_score.cast(db_service.engine.dialect.numeric_type) < threshold
            ).order_by(MessageProcessing.confidence_score).limit(limit // 2).all()
            
            # Get low confidence annotations
            low_confidence_annotations = session.query(MessageAnnotation, Message).join(
                Message, MessageAnnotation.message_id == Message.id
            ).filter(
                MessageAnnotation.confidence_score.isnot(None),
                MessageAnnotation.confidence_score.cast(db_service.engine.dialect.numeric_type) < threshold
            ).order_by(MessageAnnotation.confidence_score).limit(limit // 2).all()
            
            return {
                'threshold': threshold,
                'low_confidence_processing': [
                    {
                        'message_id': msg.message_id,
                        'processing_step': ps.processing_step,
                        'confidence_score': float(ps.confidence_score),
                        'result': ps.result,
                        'processed_at': ps.processed_at.isoformat(),
                        'message_content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content
                    }
                    for ps, msg in low_confidence_processing
                ],
                'low_confidence_annotations': [
                    {
                        'message_id': msg.message_id,
                        'annotation_type': ann.annotation_type,
                        'confidence_score': float(ann.confidence_score),
                        'annotation_value': ann.annotation_value,
                        'annotated_at': ann.annotated_at.isoformat(),
                        'message_content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content
                    }
                    for ann, msg in low_confidence_annotations
                ]
            }
    except Exception as e:
        logging.error(f"Error fetching low confidence classifications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
