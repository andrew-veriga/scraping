"""
Enhanced processing service for hierarchical Discord message structure
Handles first batch processing with parent-child message relationships
"""

import logging
import pandas as pd
import json
import os
from datetime import datetime, timezone
from fastapi import HTTPException
from typing import Dict, Any, List
from sqlalchemy import func

from app.services import data_loader_hierarchical, thread_service, solution_service
from app.services.database import get_database_service
from app.models.pydantic_models import SolutionStatus
from app.models.db_models import ProcessingBatch
from app.utils.file_utils import create_dict_from_list
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

def _load_messages_batch_to_database(messages_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Load a batch of hierarchical messages into the database.
    
    Args:
        messages_df: DataFrame containing messages to load
        
    Returns:
        Dict containing database operation results and statistics
        
    Raises:
        HTTPException: If database loading fails
    """
    logging.info("💾 Loading hierarchical messages into database...")
    
    try:
        db_stats = data_loader_hierarchical.load_messages_to_database_hierarchical(messages_df)
        
        result = {
            'status': 'success',
            **convert_numpy_types(db_stats)
        }
        
        logging.info(f"✅ Database loading complete - {db_stats['new_messages_created']} new messages created")
        return result
        
    except Exception as e:
        logging.error(f"❌ Failed to load messages to database: {e}")
        error_result = {
            'status': 'failed', 
            'error': str(e)
        }
        raise HTTPException(status_code=500, detail=f"Failed to load to database: {e}")


def process_first_batch_hierarchical(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process the first batch of Discord messages with hierarchical structure.
    
    This enhanced version:
    1. Loads Discord messages from Excel
    2. Analyzes parent-child relationships from Referenced Message ID
    3. Creates hierarchical message structure in database
    4. Returns detailed statistics and validation results
    """
    
    try:
        # Extract configuration
        messages_file_path = config['MESSAGES_FILE_PATH']
        save_path = config['SAVE_PATH']
        interval_first = config.get('INTERVAL_FIRST', 2)  # days for first batch
        
        
        # Initialize result statistics
        result_stats = {
            'status': 'processing',
            'start_time': datetime.now(timezone.utc),
            'messages_file': messages_file_path,
            'data_loading': {},
            'hierarchy_analysis': {},
            'database_operations': {},
            'validation_results': {},
            'processing_summary': {}
        }
        
        # Step 1: Load and preprocess Discord data
        try:
            messages_df = data_loader_hierarchical.load_and_preprocess_data(messages_file_path)
            result_stats['data_loading'] = {
                'status': 'success',
                'total_messages_loaded': int(len(messages_df)),
                'date_range': {
                    'earliest': messages_df['DateTime'].min().isoformat(),
                    'latest': messages_df['DateTime'].max().isoformat()
                }
            }
            
        except Exception as e:
            logging.error(f"❌ Failed to load Discord data: {e}")
            result_stats['data_loading'] = {'status': 'failed', 'error': str(e)}
            raise HTTPException(status_code=500, detail=f"Failed to load Discord data: {e}")
        
        # Step 2: Analyze message hierarchy
        try:
            hierarchical_df, hierarchy_stats = data_loader_hierarchical.analyze_message_hierarchy(messages_df)
            result_stats['hierarchy_analysis'] = {
                'status': 'success',
                **convert_numpy_types(hierarchy_stats)
            }
            logging.info(f"✅ Hierarchy analysis complete - {hierarchy_stats['threads_identified']} threads identified")
            
        except Exception as e:
            logging.error(f"❌ Failed to analyze message hierarchy: {e}")
            result_stats['hierarchy_analysis'] = {'status': 'failed', 'error': str(e)}
            raise HTTPException(status_code=500, detail=f"Failed to analyze hierarchy: {e}")
        
        # Step 3: Validate hierarchy integrity
        logging.info("🛡️ Step 3: Validating message hierarchy integrity...")
        try:
            validation_results = data_loader_hierarchical.validate_message_hierarchy(hierarchical_df)
            result_stats['validation_results'] = validation_results
            
            if not validation_results['valid']:
                logging.warning(f"⚠️ Hierarchy validation found {len(validation_results['issues'])} issues")
                for issue in validation_results['issues']:
                    logging.warning(f"   - {issue}")
            else:
                logging.info("✅ Hierarchy validation passed")
                
        except Exception as e:
            logging.error(f"❌ Failed to validate hierarchy: {e}")
            result_stats['validation_results'] = {'valid': False, 'error': str(e)}
        
        # Step 4: Filter messages for first batch (based on time interval)
        start_date = hierarchical_df['DateTime'].min().normalize()
        end_date = start_date + pd.Timedelta(days=interval_first)
        first_batch_df = hierarchical_df[hierarchical_df['DateTime'] < end_date].copy()
        
        logging.info(f"📅 Processing first {interval_first} days: {start_date.date()} to {end_date.date()}")
        logging.info(f"   First batch contains {len(first_batch_df)} messages from {len(hierarchical_df)} total")
        
        # Step 5: Load hierarchical messages into database
        result_stats['database_operations'] = _load_messages_batch_to_database(first_batch_df)
        
        # Step 6: LLM Thread Gathering
        logging.info("🤖 Step 6: LLM Thread Gathering - Grouping messages into conversation threads...")
        try:
            str_interval = f"{start_date.date()}-{end_date.date()}"
            step1_output_filename = thread_service.first_thread_gathering(first_batch_df, f"first_{str_interval}", save_path)
            result_stats['llm_thread_gathering'] = {
                'status': 'success',
                'output_file': step1_output_filename,
                'messages_processed': len(first_batch_df)
            }
            logging.info(f"✅ LLM thread gathering completed")
            
        except Exception as e:
            logging.error(f"❌ Failed LLM thread gathering: {e}")
            result_stats['llm_thread_gathering'] = {'status': 'failed', 'error': str(e)}
            raise HTTPException(status_code=500, detail=f"Failed LLM thread gathering: {e}")

        # Step 7: Technical Topic Filtering
        logging.info("🔬 Step 7: Filtering technical topics...")
        try:
            # Get messages dataframe for full context (not just first batch)
            full_messages_df = data_loader_hierarchical.load_and_preprocess_data(messages_file_path)
            technical_filename = thread_service.filter_technical_threads(step1_output_filename, f"first_{str_interval}", full_messages_df, save_path)
            result_stats['technical_filtering'] = {
                'status': 'success',
                'output_file': technical_filename
            }
            logging.info(f"✅ Technical filtering completed")
            
        except Exception as e:
            logging.error(f"❌ Failed technical filtering: {e}")
            result_stats['technical_filtering'] = {'status': 'failed', 'error': str(e)}
            raise HTTPException(status_code=500, detail=f"Failed technical filtering: {e}")

        # Step 8: solution Generalization
        logging.info("💡 Step 8: Extracting and generalizing solutions...")
        try:
            solutions_filename = thread_service.generalization_solution(technical_filename, f"first_{str_interval}", save_path)
            
            # Load solutions from file and convert to dict format
            import json
            with open(solutions_filename, 'r') as f:
                solutions_list = json.load(f)
            solutions_dict = create_dict_from_list(solutions_list)
            
            result_stats['solution_generalization'] = {
                'status': 'success',
                'output_file': solutions_filename,
                'solutions_extracted': len(solutions_dict)
            }
            logging.info(f"✅ solution generalization completed - {len(solutions_dict)} solutions extracted")
            
        except Exception as e:
            logging.error(f"❌ Failed solution generalization: {e}")
            result_stats['solution_generalization'] = {'status': 'failed', 'error': str(e)}
            raise HTTPException(status_code=500, detail=f"Failed solution generalization: {e}")

        # Step 9: RAG Duplicate Checking and Database Update
        logging.info("🔍 Step 9: RAG duplicate checking and database storage...")
        try:
            # Check for duplicates using RAG
            final_solutions_dict = solution_service.check_in_rag_and_save({}, solutions_dict)
            
            # Update database with solutions (replaces JSON file storage)
            solution_service.update_database_with_solutions(final_solutions_dict)
            
            result_stats['rag_processing'] = {
                'status': 'success',
                'initial_solutions': len(solutions_dict),
                'final_solutions': len(final_solutions_dict),
                'duplicates_found': len(solutions_dict) - len(final_solutions_dict)
            }
            logging.info(f"✅ RAG processing completed - {len(final_solutions_dict)} unique solutions saved to database")
            
        except Exception as e:
            logging.error(f"❌ Failed RAG processing: {e}")
            result_stats['rag_processing'] = {'status': 'failed', 'error': str(e)}
            raise HTTPException(status_code=500, detail=f"Failed RAG processing: {e}")

        # Step 10: Create processing batch record
        logging.info("📝 Step 10: Creating processing batch record...")
        try:
            db_service = get_database_service()
            with db_service.get_session() as session:
                batch_data = {
                    'batch_type': 'first_hierarchical',  # Shorter name to fit column limit
                    'start_date': start_date,
                    'end_date': end_date,
                    'messages_processed': len(first_batch_df),
                    'threads_created': len(final_solutions_dict),
                    'technical_threads': len([s for s in final_solutions_dict.values() if s.get('label') != SolutionStatus.UNRESOLVED]),
                    'solutions_added': len([s for s in final_solutions_dict.values() if s.get('solution')]),
                    'lookback_date': None
                }
                processing_batch = db_service.create_processing_batch(session, batch_data)
                session.commit()
                batch_id = processing_batch.id
                
                logging.info(f"✅ Created processing batch record with ID: {batch_id}")
                
        except Exception as e:
            logging.error(f"❌ Failed to create processing batch record: {e}")
            batch_id = None
        
        # Step 11: Compile final statistics and results
        result_stats['processing_summary'] = {
            'status': 'completed',
            'end_time': datetime.now(timezone.utc),
            'batch_id': batch_id,
            'total_processing_time': (datetime.now(timezone.utc) - result_stats['start_time']).total_seconds(),
            'messages_in_first_batch': int(len(first_batch_df)),
            'threads_in_first_batch': int(first_batch_df['thread_id'].nunique()),
            'hierarchy_max_depth': int(first_batch_df['depth_level'].max()),
            'llm_threads_created': len(final_solutions_dict) if 'final_solutions_dict' in locals() else 0,
            'technical_threads': len([s for s in final_solutions_dict.values() if s.get('label') != SolutionStatus.UNRESOLVED]) if 'final_solutions_dict' in locals() else 0,
            'solutions_extracted': len([s for s in final_solutions_dict.values() if s.get('solution')]) if 'final_solutions_dict' in locals() else 0,
            'ready_for_incremental_processing': True
        }
        
        # Log final summary
        logging.info("🎉 Complete first batch hierarchical processing finished successfully!")
        logging.info(f"   📊 Processed {result_stats['processing_summary']['messages_in_first_batch']} messages")
        logging.info(f"   🌳 Maximum conversation depth: {result_stats['processing_summary']['hierarchy_max_depth']}")
        logging.info(f"   🤖 LLM threads created: {result_stats['processing_summary']['llm_threads_created']}")
        logging.info(f"   🔬 Technical threads: {result_stats['processing_summary']['technical_threads']}")
        logging.info(f"   💡 Solutions extracted: {result_stats['processing_summary']['solutions_extracted']}")
        logging.info(f"   ⏱️  Total processing time: {result_stats['processing_summary']['total_processing_time']:.2f} seconds")
        
        # Return success response
        response = {
            "status": "success",
            "message": "Complete first batch processing finished successfully with hierarchical structure and LLM analysis",
            "statistics": result_stats,
            "next_steps": [
                "Process next incremental batches using process_next_batches endpoint",
                "Query solutions from database for RAG similarity searches", 
                "Generate reports from database-stored solutions",
                "Monitor processing status via hierarchical-status endpoint"
            ]
        }
        
        # Convert all numpy types to native Python types for JSON serialization
        return convert_numpy_types(response)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
        
    except Exception as e:
        # Log and wrap unexpected errors
        logging.error(f"❌ Unexpected error in first batch processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail=f"Unexpected error in first batch processing: {str(e)}"
        )


def get_hierarchical_processing_status() -> Dict[str, Any]:
    """
    Get the current status of hierarchical processing.
    Returns information about messages, threads, and hierarchy in the database.
    """
    
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # Get message statistics
            from app.models.db_models import Message, Thread, Solution
            
            total_messages = session.query(Message).count()
            root_messages = session.query(Message).filter(Message.is_root_message == True).count()
            reply_messages = total_messages - root_messages
            
            # Get thread statistics  
            total_threads = session.query(Thread).count()
            
            # Get depth statistics
            if total_messages > 0:
                max_depth = session.query(func.max(Message.depth_level)).scalar() or 0
                avg_depth = session.query(func.avg(Message.depth_level)).scalar() or 0
            else:
                max_depth = avg_depth = 0
            
            # Get solution statistics
            total_solutions = session.query(Solution).count()
            
            # Get recent processing batches
            recent_batches = session.query(ProcessingBatch).order_by(
                ProcessingBatch.started_at.desc()
            ).limit(5).all()
            
            return {
                "database_status": "connected",
                "message_statistics": {
                    "total_messages": total_messages,
                    "root_messages": root_messages,
                    "reply_messages": reply_messages,
                    "hierarchy_depth": {
                        "maximum": max_depth,
                        "average": round(avg_depth, 2)
                    }
                },
                "thread_statistics": {
                    "total_threads": total_threads,
                    "messages_per_thread_avg": round(total_messages / total_threads, 2) if total_threads > 0 else 0
                },
                "solution_statistics": {
                    "total_solutions": total_solutions
                },
                "recent_processing": [
                    {
                        "batch_id": batch.id,
                        "type": batch.batch_type,
                        "started_at": batch.started_at.isoformat(),
                        "status": batch.status,
                        "messages_processed": batch.messages_processed
                    }
                    for batch in recent_batches
                ]
            }
            
    except Exception as e:
        logging.error(f"Failed to get processing status: {e}")
        return {
            "database_status": "error",
            "error": str(e)
        }