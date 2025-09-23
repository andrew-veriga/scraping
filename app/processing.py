
import logging
import pandas as pd
from fastapi import HTTPException
from app.services import data_loader, thread_service, solution_service
import json
from app.utils.file_utils import (
    load_solutions_dict,
    save_solutions_dict,
    create_dict_from_list,
    get_latest_normalized_date,
    set_latest_solution_date
    )
from app.services.database import get_database_service
from app.models.pydantic_models import SolutionStatus
from datetime import datetime, timezone
from typing import Dict, Any
import numpy as np

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def initial_full_loading_raw_data(messages_file_path: str) -> Dict[str, Any]:
    """
    Load and preprocess Discord data from Excel file and load authors to database.
    Returns comprehensive statistics about the loading process.
    Handles all errors internally and returns error information in the result.
    """
    result_stats = {
        'data_loading': {'status': 'failed', 'error': None},
        'database_operations': {'status': 'failed', 'error': None}
    }
    
    try:
        logging.info(f"📂 Loading Discord data from {messages_file_path}")
        messages_df, authors_df = data_loader.load_and_preprocess_data(messages_file_path)
        
        # Update data loading stats on success
        result_stats['data_loading'] = {
            'status': 'success',
            'total_messages_loaded': int(len(messages_df)),
            'total_authors_loaded': int(len(authors_df)),
            'date_range': {
                'earliest': messages_df['DateTime'].min().isoformat(),
                'latest': messages_df['DateTime'].max().isoformat()
            }
        }
        
        logging.info(f"✅ Data loading complete - {len(messages_df)} messages, {len(authors_df)} authors loaded")
        
    except Exception as e:
        error_msg = f"Failed to load Discord data from {messages_file_path}: {e}"
        logging.error(f"❌ {error_msg}")
        result_stats['data_loading'] = {'status': 'failed', 'error': str(e)}
        return {
            'messages_df': None,
            'stats': result_stats,
            'db_stats': None,
            'error': error_msg
        }
    
    try:
        # Load authors to database
        logging.info("👥 Loading authors to database...")
        db_stats = data_loader.load_authors_to_database(authors_df)
        
        # Update database operations stats on success
        result_stats['database_operations'] = {
            'status': 'success',
            'authors_loaded': db_stats.get('new_authors_created', 0),
            'total_authors': db_stats.get('total_authors', len(authors_df)),
            'existing_skipped': db_stats.get('existing_authors_skipped', 0)
        }
        
        logging.info(f"✅ Authors database loading complete - {db_stats.get('new_authors_created', 0)} new authors created")
        
    except Exception as e:
        error_msg = f"Failed to load authors to database: {e}"
        logging.error(f"❌ {error_msg}")
        result_stats['database_operations'] = {'status': 'failed', 'error': str(e)}
        return {
            'messages_df': messages_df,
            'stats': result_stats,
            'db_stats': None,
            'error': error_msg
        }
    
    # Return successful result
    return {
        'messages_df': messages_df,
        'stats': result_stats,
        'db_stats': db_stats,
        'error': None
    }

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

def process_first_batch(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the first batch of Discord messages with enhanced error handling and statistics.
    
    This enhanced version:
    1. Loads Discord messages from Excel
    2. Creates hierarchical message structure in database
    3. Returns detailed statistics and validation results
    """
    # Extract configuration
    INTERVAL_FIRST = config.get('INTERVAL_FIRST', 2)  # days for first batch
    SOLUTIONS_DICT_FILENAME = config['SOLUTIONS_DICT_FILENAME']
    SAVE_PATH = config['SAVE_PATH']
    MESSAGES_FILE_PATH = config['MESSAGES_FILE_PATH']

    try:
        # Initialize result statistics
        result_stats = {
            'status': 'processing',
            'start_time': datetime.now(timezone.utc),
            'messages_file': MESSAGES_FILE_PATH,
            'data_loading': {},
            'database_operations': {},
            'processing_summary': {}
        }
        
        # Step 1: Load and preprocess Discord data
        logging.info(f"📂 Step 1: Loading Discord data from {MESSAGES_FILE_PATH}")
        loading_result = initial_full_loading_raw_data(MESSAGES_FILE_PATH)
        
        # Check for errors in loading result
        if loading_result['error']:
            result_stats['data_loading'] = loading_result['stats']['data_loading']
            result_stats['database_operations'] = loading_result['stats']['database_operations']
            raise HTTPException(status_code=500, detail=loading_result['error'])
        
        messages_df = loading_result['messages_df']
        
        result_stats['data_loading'] = loading_result['stats']['data_loading']
        result_stats['database_operations'] = loading_result['stats']['database_operations']
        db_stats = loading_result['db_stats']
        logging.info(f"✅ Data loading complete - {len(messages_df)} messages loaded")


        # Step 2: Filter messages for first batch (based on time interval)
        start_date = messages_df['DateTime'].min().normalize()
        end_date = start_date + pd.Timedelta(days=INTERVAL_FIRST)
        first_batch_df = messages_df[messages_df['DateTime'] < end_date].copy()
        set_latest_solution_date(config, new_date=start_date- pd.Timedelta(days=1))
        
        logging.info(f"📅 Step 2: Processing first {INTERVAL_FIRST} days: {start_date.date()} to {end_date.date()}")
        logging.info(f"   First batch contains {len(first_batch_df)} messages from {len(messages_df)} total")
        
        # Step 3: Load messages into database
        try:
            logging.info("💾 Step 3: Loading messages into database...")
            db_stats = data_loader.load_messages_to_database(first_batch_df)
            result_stats['database_operations'] = {
                'status': 'success',
                'messages_loaded': db_stats.get('new_messages_created', 0),
                'total_messages': db_stats.get('total_messages', len(first_batch_df)),
                'existing_skipped': db_stats.get('existing_messages_skipped', 0)
            }
            logging.info(f"✅ Database loading complete - {db_stats.get('new_messages_created', 0)} messages loaded")
            
        except Exception as e:
            logging.error(f"❌ Failed to load messages to database: {e}")
            result_stats['database_operations'] = {'status': 'failed', 'error': str(e)}
            raise HTTPException(status_code=500, detail=f"Failed to load messages to database: {e}")
        
        # Step 4: Create processing batch record
        try:
            logging.info("📝 Step 4: Creating processing batch record...")
            db_service = get_database_service()
            with db_service.get_session() as session:
                batch_data = {
                    'batch_type': 'first_batch',
                    'start_date': start_date,
                    'end_date': end_date
                }
                processing_batch = db_service.create_processing_batch(session, batch_data)
                session.commit()
                batch_id = processing_batch.id
                logging.info(f"✅ Created processing batch record with ID: {batch_id}")
                
        except Exception as e:
            logging.error(f"❌ Failed to create processing batch record: {e}")
            batch_id = None
        
        # Step 5: LLM Thread Gathering
        try:
            logging.info("🤖 Step 5: LLM Thread Gathering - Grouping messages into conversation threads...")
            str_interval = f"{start_date.date()}-{end_date.date()}"
            step1_output_filename = thread_service.first_thread_gathering(first_batch_df, f"first_{str_interval}", SAVE_PATH)
            logging.info(f"✅ LLM thread gathering completed")
            
        except Exception as e:
            logging.error(f"❌ Failed LLM thread gathering: {e}")
            raise HTTPException(status_code=500, detail=f"Failed LLM thread gathering: {e}")

        # Step 6: Technical Topic Filtering
        try:
            logging.info("🔬 Step 6: Filtering technical topics...")
            first_technical_filename = thread_service.filter_technical_threads(step1_output_filename, f"first_{str_interval}", SAVE_PATH)
            logging.info(f"✅ Technical filtering completed")
            
        except Exception as e:
            logging.error(f"❌ Failed technical filtering: {e}")
            raise HTTPException(status_code=500, detail=f"Failed technical filtering: {e}")

        # Step 7: Solution Generalization
        try:
            logging.info("💡 Step 7: Extracting and generalizing solutions...")
            first_solutions_filename = thread_service.generalization_solution(first_technical_filename, f"first_{str_interval}", SAVE_PATH)
            
            # Load solutions from file and convert to dict format
            solutions_list = json.load(open(first_solutions_filename, 'r'))
            first_solutions_dict = create_dict_from_list(solutions_list)
            logging.info(f"✅ Solution generalization completed - {len(first_solutions_dict)} solutions extracted")
            
        except Exception as e:
            logging.error(f"❌ Failed solution generalization: {e}")
            raise HTTPException(status_code=500, detail=f"Failed solution generalization: {e}")

        # Step 8: RAG Duplicate Checking and Database Update
        try:
            logging.info("🔍 Step 8: RAG duplicate checking and database storage...")
            solutions_dict = solution_service.check_in_rag_and_save({}, first_solutions_dict)
            save_solutions_dict(solutions_dict, config)
            solution_service.update_database_with_solutions(solutions_dict)
            logging.info(f"✅ RAG processing completed - {len(solutions_dict)} unique solutions saved to database")
            
        except Exception as e:
            logging.error(f"❌ Failed RAG processing: {e}")
            raise HTTPException(status_code=500, detail=f"Failed RAG processing: {e}")
        
        # Step 9: Complete processing batch record
        try:
            with db_service.get_session() as session:
                batch_stats = {
                    'status': 'completed'
                }
                db_service.complete_processing_batch(session, batch_id, batch_stats)
                session.commit()
                logging.info(f"✅ Completed processing batch {batch_id} with stats: {batch_stats}")
                
        except Exception as e:
            logging.error(f"❌ Failed to complete processing batch record: {e}")
        
        # Step 10: Compile final statistics and results
        result_stats['processing_summary'] = {
            'status': 'completed',
            'end_time': datetime.now(timezone.utc),
            'batch_id': batch_id,
            'total_processing_time': (datetime.now(timezone.utc) - result_stats['start_time']).total_seconds(),
            'messages_in_first_batch': int(len(first_batch_df)),
            'solutions_extracted': len(solutions_dict),
            'ready_for_incremental_processing': True
        }
        
        # Log final summary
        logging.info("🎉 First batch processing finished successfully!")
        logging.info(f"   📊 Processed {result_stats['processing_summary']['messages_in_first_batch']} messages")
        logging.info(f"   💡 Solutions extracted: {result_stats['processing_summary']['solutions_extracted']}")
        logging.info(f"   ⏱️  Total processing time: {result_stats['processing_summary']['total_processing_time']:.2f} seconds")
        
        # Return enhanced response
        response = {
            "status": "success",
            "message": "First batch processed successfully with enhanced monitoring",
            "statistics": result_stats,
            "database_messages_loaded": db_stats.get('new_messages_created', 0),
            "solutions_created": len(solutions_dict),
            "next_steps": [
                "Process next incremental batches using process_next_batches endpoint",
                "Query solutions from database for RAG similarity searches", 
                "Generate reports from database-stored solutions"
            ]
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        return convert_numpy_types(response)
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
    db_stats = data_loader.load_messages_to_database(next_batch_df)
        
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
            'lookback_date': lookback_date
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
    
    save_solutions_dict(solutions_dict, config)
    
    solution_service.update_database_with_solutions(adding_solutions_dict)
    
    with db_service.get_session() as session:
        new_solution_count = len(solutions_dict) - initial_solution_count

        batch_stats = {
            'status': 'completed'
        }
        db_service.complete_processing_batch(session, batch_id, batch_stats)
        session.commit()
        logging.info(f"Completed processing batch {batch_id} with stats: {batch_stats}")
    
    return True


def process_next_batches(config):
    try:
        MESSAGES_FILE_PATH = config['MESSAGES_FILE_PATH']
        NEXT_BATCH_FILE_PATH = config['NEXT_BATCH_FILE_PATH']
        INTERVAL_FIRST = config['INTERVAL_FIRST']
        INTERVAL_NEXT = config['INTERVAL_NEXT']
        INTERVAL_BACK = config['INTERVAL_BACK']
        SOLUTIONS_DICT_FILENAME = config['SOLUTIONS_DICT_FILENAME']
        SAVE_PATH = config['SAVE_PATH']

        logging.info(f"Processing next batch from {NEXT_BATCH_FILE_PATH}")
        # load full list of messages and authors from file
        if NEXT_BATCH_FILE_PATH != MESSAGES_FILE_PATH:
            loading_result = initial_full_loading_raw_data(NEXT_BATCH_FILE_PATH)
            # Check for errors in loading result
            if loading_result['error']:
                raise HTTPException(status_code=500, detail=loading_result['error'])
            messages_df = loading_result['messages_df']
        else:
            messages_df, _ = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)

        solutions_dict = load_solutions_dict(SOLUTIONS_DICT_FILENAME, SAVE_PATH)
        
        # Get latest solution date from database instead of file

        latest_normalized_date = get_latest_normalized_date(config)
        if latest_normalized_date:
            next_end_date = latest_normalized_date
        else:
            next_end_date = messages_df['DateTime'].min().normalize() + pd.Timedelta(days=INTERVAL_FIRST)

        while True:
            next_start_date = next_end_date
            lookback_date = next_start_date - pd.Timedelta(days=INTERVAL_BACK)
            next_end_date = next_start_date + pd.Timedelta(days=INTERVAL_NEXT)

            if next_start_date > messages_df['DateTime'].max():
                break
            logging.info("##" * 60)
            logging.info(f"Processing batch. Lookback: {lookback_date}, Start: {next_start_date}, End: {next_end_date}")
            if not process_batch(solutions_dict, lookback_date, next_start_date, next_end_date, messages_df, config):
                logging.info(f"No messages in the interval: {next_start_date} to {next_end_date}")
        return {"message": "Next batch processed successfully"}
    except Exception as e:
        logging.error(f"Error processing next batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
