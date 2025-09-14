
import logging
import pandas as pd
from fastapi import HTTPException
from app.services import data_loader_hierarchical
from app.services import data_loader, thread_service, solution_service, processing_hierarchical
import json
from app.utils.file_utils import load_solutions_dict, save_solutions_dict, create_dict_from_list
from app.services.database import get_database_service
from app.models.pydantic_models import SolutionStatus

def full_process(config):
    """
    Runs the full data processing pipeline:
    1. Processes the initial batch of messages to create a baseline solutions file.
    2. Processes all subsequent batches of messages until up-to-date.
    """
    try:
        logging.info("Starting full process...")
        logging.info("Starting process_first_batch...")
        process_first_batch(config)
        logging.info("process_first_batch complete.")
        logging.info("Starting process_next_batches...")
        process_next_batches(config)
        logging.info("Full process finished successfully.")
        return {"message": "Full process completed successfully."}
    except Exception as e:
        logging.error("An error occurred during the full process.", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred during the full process: {getattr(e, 'detail', str(e))}")

def process_first_batch(config):
    try:
        MESSAGES_FILE_PATH = config['MESSAGES_FILE_PATH']
        INTERVAL_FIRST = config['INTERVAL_FIRST']
        SOLUTIONS_DICT_FILENAME = config['SOLUTIONS_DICT_FILENAME']
        SAVE_PATH = config['SAVE_PATH']

        logging.info(f"Processing first batch from {MESSAGES_FILE_PATH}")
        messages_df = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)
        
        messages_loaded = data_loader.load_messages_to_database(messages_df)
        logging.info(f"Loaded {messages_loaded} messages into database")
        
        start_date = messages_df['DateTime'].min().normalize()
        end_date = start_date + pd.Timedelta(days=INTERVAL_FIRST)
        str_interval = f"{start_date.date()}-{end_date.date()}"
        
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

        step1_output_filename = thread_service.first_thread_gathering(first_some_days_df,  f"first_{str_interval}", SAVE_PATH)
        first_technical_filename = thread_service.filter_technical_threads(step1_output_filename, f"first_{str_interval}", SAVE_PATH)
        first_solutions_filename = thread_service.generalization_solution(first_technical_filename,  f"first_{str_interval}", SAVE_PATH)
        solutions_list = json.load(open(first_solutions_filename, 'r'))
        first_solutions_dict = create_dict_from_list(solutions_list)
        solutions_dict = solution_service.check_in_rag_and_save({},first_solutions_dict)
        save_solutions_dict(solutions_dict, SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        
        solution_service.update_database_with_solutions(solutions_dict)
        
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

def process_batch(solutions_dict, lookback_date:pd.Timestamp, next_start_date: pd.Timestamp, next_end_date: pd.Timestamp, messages_df, config):
    str_interval = f"{next_start_date.date()}-{next_end_date.date()}"
    next_batch_df = messages_df[(next_start_date <= messages_df['DateTime']) & (messages_df['DateTime'] < next_end_date)].copy()
    if next_batch_df.empty:
        logging.info(f"No messages found in the interval {str_interval}. Skipping this batch.")
        return False
    result_stats = {}    
    db_stats = data_loader_hierarchical.load_messages_to_database_hierarchical(next_batch_df)
        
    # result = {
    #     'status': 'success',
    #     **convert_numpy_types(db_stats)
    # }
        
    logging.info(f"✅ Database loading complete - {db_stats['new_messages_created']} new messages created")
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

    next_step1_output_filename = thread_service.next_thread_gathering(next_batch_df, lookback_date, str_interval, config['SAVE_PATH'], messages_df)
    next_technical_filename = thread_service.filter_technical_threads(next_step1_output_filename, f"next_{str_interval}", config['SAVE_PATH'])
    next_solutions_filename = thread_service.generalization_solution(next_technical_filename, f"next_{str_interval}", config['SAVE_PATH'])

    adding_solutions_dict = solution_service.new_solutions_revision_and_add(next_solutions_filename, next_technical_filename, solutions_dict, lookback_date)

    initial_solution_count = len(solutions_dict)
    
    solutions_dict = solution_service.check_in_rag_and_save(solutions_dict, adding_solutions_dict)
    
    save_solutions_dict(solutions_dict, config['SOLUTIONS_DICT_FILENAME'], save_path=config['SAVE_PATH'])
    
    solution_service.update_database_with_solutions(adding_solutions_dict)
    
    with db_service.get_session() as session:
        new_solution_count = len(solutions_dict) - initial_solution_count

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

def process_next_batches(config):
    try:
        MESSAGES_FILE_PATH = config['MESSAGES_FILE_PATH']
        INTERVAL_FIRST = config['INTERVAL_FIRST']
        INTERVAL_NEXT = config['INTERVAL_NEXT']
        INTERVAL_BACK = config['INTERVAL_BACK']
        SOLUTIONS_DICT_FILENAME = config['SOLUTIONS_DICT_FILENAME']
        SAVE_PATH = config['SAVE_PATH']

        logging.info(f"Processing next batch from {MESSAGES_FILE_PATH}")
        messages_df = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)
        solutions_dict = load_solutions_dict(SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        
        # Get latest solution date from database instead of file
        db_service = get_database_service()
        with db_service.get_session() as session:
            latest_solution_date = db_service.get_latest_solution_date(session)
        if latest_solution_date:
            next_end_date = latest_solution_date
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
            if not process_batch(solutions_dict, lookback_date, next_start_date, next_end_date, messages_df, config):
                logging.info(f"No messages in the interval: {next_start_date} to {next_end_date}")
        return {"message": "Next batch processed successfully"}
    except Exception as e:
        logging.error(f"Error processing next batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
