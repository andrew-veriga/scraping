import json
import re
# import datetime
from datetime import timezone
import os
from typing import Optional
import pandas as pd
import logging
from sqlalchemy import func
from app.services import gemini_service, prompts
from app.services.processing_tracker import get_processing_tracker
from app.services.database import get_database_service
from app.models.pydantic_models import ThreadStatus
from app.models.db_models import Message
from app.utils.file_utils import *
from google.genai.types import Content
import yaml 
from app.utils.yaml_file_utils import load_yaml_genai_files, save_yaml_genai_files
def _create_thread_object(session, thread_id, thread_messages):
    """
    Create a new Thread object in the database if it doesn't already exist.
    
    Args:
        session: Database session
        thread_id: The topic_id for the thread
        thread_messages: List of message IDs in the thread
        
    Returns:
        int: Number of threads created (0 or 1)
    """
    from app.models.db_models import Thread
    
    # Check if thread already exists
    existing_thread = session.query(Thread).filter(Thread.topic_id == thread_id).first()
    
    if not existing_thread:
        # Find the actual date from messages in this thread
        actual_date = None
        
        # Get the latest message date in this thread
        latest_msg = session.query(Message).filter(
            Message.message_id.in_(thread_messages)
        ).order_by(Message.datetime.desc()).first()
        
        if latest_msg:
            actual_date = latest_msg.datetime
        else:
            actual_date = session.query(Message).filter(
                Message.message_id==thread_messages[0]
            ).first().datetime
        # Create thread object
               
        new_thread = Thread(
            topic_id=thread_id,
            header='',  # Will be updated later by LLM analysis
            actual_date=actual_date,
            status='new',
            is_technical=False,  # Will be determined later by technical filtering
            is_processed=False
        )
        session.add(new_thread)
        logging.info(f"Created Thread object for {thread_id}")
        return 1
    
    return 0

from app.services.pictured_messages import parts_from_db_messages, parts_from_dataframe, make_text_part,make_json_part

from google.genai import types
from datetime import datetime

def first_thread_gathering(logs_df: pd.DataFrame, prefix: str, global_config: dict)->tuple[str, object]: 
    """
    Gather  messages into threads using LLM from DataFrame provided flat message list
    Arguments:
        logs_df: DataFrame of messages
        prefix: string prefix for debug saving
        global_config: global configuration dictionary
    Returns:
        full_path: path to the saved threads in json format
    """
    prompts.reload_prompts()
    # list of valid message ids for validation of LLM response
    valid_ids_set = set(logs_df['Message ID'].astype(str))
    db_service = get_database_service()
    # get new messages parts
    parts, image_genai_cache = parts_from_dataframe(logs_df, global_config=global_config)
    
    # Save with types.File object tags
    
    parts.extend([make_text_part(prompts.system_prompt),make_text_part(prompts.prompt_gathering_logic)])
        
    # Save with safe_dump to avoid Python object tags
     # processing_tracker = get_processing_tracker()
    db_service = get_database_service()

    if image_genai_cache:
        gemini_service.config_step1.cached_content = image_genai_cache
    response = gemini_service.generate_content(
        contents=[
        Content(parts=parts, role="user"),
        ],
        config=gemini_service.config_step1,
        valid_ids_set=valid_ids_set,
        log_prefix="First thread gathering:"
    )
    output_filename = f'{prefix}_gathering.json'

    threads_list_dict = [thread.model_dump() for thread in response.threads]
    
    try:
        with db_service.get_session() as session:###############################
            message_to_thread = {}
            message_to_parent = {}  # Track LLM-determined parent relationships
            parent_id_updates = 0
            
            # Build mappings from LLM response and create Thread objects
            cnt_db_created = 0
            for thread in response.threads:
                thread_id = thread.topic_id
                
                # Create Thread object in database
                try:
                    thread_messages = [str(msg.message_id) for msg in thread.whole_thread]
                    cnt_db_created += _create_thread_object(
                        session, 
                        thread_id, 
                        thread_messages
                        )
                        
                except Exception as e:
                    logging.warning(f"Could not create Thread object for {thread_id}: {e}")
                
                # Build message mappings
                for msg in thread.whole_thread:
                    message_to_thread[str(msg.message_id)] = thread_id
                    # Store LLM-determined parent_id (might be different from Referenced Message ID)
                    if msg.parent_id:
                        message_to_parent[str(msg.message_id)] = str(msg.parent_id)
            
            # Update database with LLM analysis results
            for idx, row in logs_df.iterrows():
                
                db_message = session.query(Message).filter(
                    Message.message_id == idx
                ).first()
                
                if db_message:
                    # Update thread assignment
                    assigned_thread = message_to_thread.get(idx)
                    if assigned_thread and db_message.thread_id != assigned_thread:
                        db_message.thread_id = assigned_thread
                    
                    # Update parent_id based on LLM analysis (this is key!)
                    llm_parent_id = message_to_parent.get(idx)
                    old_parent = db_message.parent_id
                    parent_id_updated = False
                    
                    if llm_parent_id != old_parent:
                        db_message.parent_id = llm_parent_id
                        parent_id_updates += 1
                        parent_id_updated = True
                        logging.info(f"Updated parent_id for {idx}: {old_parent} -> {llm_parent_id}")

                    # Record processing step
                    result = {
                        'assigned_to_thread': assigned_thread,
                        'thread_grouping_method': 'llm_analysis',
                        'total_threads_created': len(response.threads),
                        'parent_id_updated': parent_id_updated,
                        'previous_parent_id': old_parent,
                        'new_parent_id': llm_parent_id
                    }
                    
                    # processing_tracker.record_processing_step(
                    #     session=session,
                    #     message_id=db_message.message_id,
                    #     processing_step=processing_tracker.STEPS['THREAD_GROUPING'],
                    #     result=result,
                    #     metadata={'batch_prefix': prefix, 'processing_type': 'first_batch'}
                    # )
            
            # # Recalculate hierarchy metadata after LLM updates
            # if parent_id_updates > 0:
            #     logging.info("🔄 Recalculating hierarchy metadata after LLM parent_id updates...")
            #     depth_updates = _recalculate_hierarchy_metadata(session)
            #     logging.info(f"  📏 Updated depth levels for {depth_updates} messages")
            
            session.commit()
            logging.info(f"Thread gathering completed:")
            logging.info(f"  📝 Recorded processing steps for {len(logs_df)} messages")
            logging.info(f"  🆕 Created Thread objects: {cnt_db_created}")
            logging.info(f"  🔗 Updated parent_id relationships: {parent_id_updates}")
            if parent_id_updates > 0:
                logging.info(f"  📏 Recalculated hierarchy metadata for improved accuracy")
            
    except Exception as e:
        logging.error(f"Failed to record thread grouping processing steps: {e}")
    
    save_path = global_config[f'SAVE_PATH']
    full_path = os.path.join(save_path, output_filename)
    with open(full_path, 'w') as f:
        json.dump(threads_list_dict, f, indent=4, default=convert_Timestamp_to_str)
    logging.info(f"Start step 1. Processed {len(logs_df)} messages")
    logging.info(f"Successfully saved {len(response.threads)} grouped threads to {full_path}")
    return full_path, image_genai_cache

def filter_technical_threads(filename: str, prefix: str, save_path: str):
    """
    Extract technical topics from JSON file using LLM with DataFrame logs provided full texts of messages
    return only technical threads
    """
    prompts.reload_prompts()
    with open(filename, 'r') as f:
        threads_json_data = json.load(f,object_hook=convert_str_to_Timestamp)
    
    db_service = get_database_service()

    processed_threads = illustrated_threads(threads_json_data, db_service)
    # processing_tracker = get_processing_tracker()

    response = gemini_service.generate_content(
        contents=[
            json.dumps(processed_threads, indent=4),
            prompts.system_prompt,
            prompts.prompt_step_2
            ],
        config=gemini_service.config_step2,
    )

    technical_threads = [thread for thread in processed_threads if thread.get('topic_id') in response.technical_topics]
    logging.info(f"{prefix} step 2. Selected {len(response.technical_topics)} technical topics from {len(processed_threads)} threads")

    try:
        with db_service.get_session() as session:
            cnt_db_updated = 0
            
            for thread in processed_threads:
                thread_id = thread.get('topic_id') or thread.get('Topic_ID')
                is_technical = thread_id in response.technical_topics
                thread['is_technical'] = is_technical
                # if is_technical:
                    # processing_tracker.annotate_message(
                    #     session=session,
                    #     message_id=thread_id,
                    #     annotation_type=processing_tracker.ANNOTATION_TYPES['TECHNICAL'],
                    #     annotation_value={'thread_id': thread_id, 'indicators': thread.get('technical_indicators', [])},
                    #     annotated_by='gemini_ai'
                    # )
                # Update Thread object with technical classification
                from app.models.db_models import Thread
                db_thread = session.query(Thread).filter(Thread.topic_id == thread_id).first()
                if db_thread:
                    if db_thread.is_technical != is_technical:
                        db_thread.is_technical = is_technical
                        db_thread.updated_at = func.now()
                        cnt_db_updated += 1
                        logging.info(f"Updated Thread {thread_id}: is_technical = {is_technical}")
                
                # whole_thread = thread.get('whole_thread', [])
                # for msg in whole_thread:
                #     msg_id = str(msg['message_id'])
                #     db_message = session.query(Message).filter(
                #         Message.message_id == msg_id
                #     ).first()
                    
                #     if db_message:
                #         result = {
                #             'is_technical': is_technical,
                #             'thread_id': thread_id,
                #             'technical_keywords_found': thread.get('technical_indicators', []),
                #             'filtering_method': 'llm_analysis'
                #         }
                        
                #         processing_tracker.record_processing_step(
                #             session=session,
                #             message_id=db_message.message_id,
                #             processing_step=processing_tracker.STEPS['TECHNICAL_FILTERING'],
                #             result=result,
                #             metadata={'batch_prefix': prefix}
                #         )
                        
            
            session.commit()
            logging.info(f"Technical filtering completed:")
            logging.info(f"  🔬 Updated Thread technical flags: {cnt_db_updated}")
            logging.info(f"  📝 Recorded processing steps for all messages")
            
    except Exception as e:
        logging.error(f"Failed to record technical filtering processing steps: {e}")

    output_filename = f'{prefix}_technical.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        f.write(json.dumps(processed_threads,  indent=2,default=convert_Timestamp_to_str))

    logging.info(f"Successfully saved {len(processed_threads)} labelled as is technical =Yes/No threads to {full_path}")
    return full_path

def generalization_solution(filename: str,prefix: str, save_path: str):
    """
    Generalize technical topics from JSON file using LLM
    Create "header" and "solution" fields
    """
    prompts.reload_prompts()
    with open(filename, 'r') as f:
        technical_threads = json.load(f,object_hook=convert_str_to_Timestamp)
    technical_threads = {t.get('topic_id'): t for t in technical_threads}


    valid_ids_set = set()
    text_threads = {}
    for key,t in technical_threads.items(): 
        whole_thread = t.get('whole_thread', [])
        valid_ids_set = valid_ids_set.union(set([msg['message_id'] for msg in whole_thread]))
        text_threads[key] = t['whole_thread']

    # processing_tracker = get_processing_tracker()
    db_service = get_database_service()
        
    response_solutions = gemini_service.generate_content( 
        contents=[
            json.dumps(text_threads, indent=2,default=convert_Timestamp_to_str),
            prompts.system_prompt,
            prompts.prompt_step_3
            ],
        config=gemini_service.solution_config,
        valid_ids_set=valid_ids_set,
        log_prefix=f"{prefix} generalization solution:"
    )

    logging.info(f"{prefix} step 3. Generalization of {len(response_solutions.threads)} technical threads")

    solutions_list = []
    try:
        with db_service.get_session() as session:
            cnt_db_updated = 0
            
            for solution_thread in response_solutions.threads:
                solution=solution_thread.model_dump()
                thread_id = solution_thread.topic_id
                whole_thread = technical_threads[thread_id]['whole_thread']
                solution['whole_thread'] = whole_thread
                solution['is_technical']=technical_threads[thread_id]['is_technical']
                # solution['actual_date']=pd.Timestamp(solution.actual_date)
                solutions_list.append(solution)
                
                # Update Thread object with solution details
                from app.models.db_models import Thread
                db_thread = session.query(Thread).filter(Thread.topic_id == thread_id).first()
                if db_thread:
                    # Update thread with LLM-generated solution details
                    if db_thread.header != solution_thread.header:
                        db_thread.header = solution_thread.header
                    
                    db_thread.solution = solution_thread.solution
                    db_thread.label = solution_thread.label
                    db_thread.answer_id = solution_thread.answer_id
                    db_thread.is_processed = True
                    db_thread.updated_at = func.now()
                    cnt_db_updated += 1
                    
                    logging.info(f"Updated Thread {thread_id} with solution: {solution_thread.header[:50]}...")
                # for tracking
                # for msg in whole_thread:
                #     msg_id = str(msg['message_id'])
                #     db_message = session.query(Message).filter(
                #         Message.message_id == msg_id
                #     ).first()
                    
                #     if db_message:
                #         result = {
                #             'solution_generated': True,
                #             'thread_id': thread_id,
                #             'solution_header': solution_thread.header,
                #             'solution_label': solution_thread.label,
                #             'extraction_method': 'llm_analysis'
                #         }
                        
                #         processing_tracker.record_processing_step(
                #             session=session,
                #             message_id=db_message.message_id,
                #             processing_step=processing_tracker.STEPS['SOLUTION_EXTRACTION'],
                #             result=result,
                #             metadata={'batch_prefix': prefix, 'solution_id': thread_id}
                #         )
                        
                #         if msg_id == solution_thread.answer_id:
                #             processing_tracker.annotate_message(
                #                 session=session,
                #                 message_id=db_message.message_id,
                #                 annotation_type=processing_tracker.ANNOTATION_TYPES['SOLUTION'],
                #                 annotation_value={'is_answer': True, 'thread_id': thread_id},
                #                 annotated_by='gemini_ai'
                #             )
            
            session.commit()
            logging.info(f"solution extraction completed:")
            logging.info(f"  💡 Updated Thread objects with solutions: {cnt_db_updated}")
            logging.info(f"  📝 Recorded processing steps for all messages")
            
    except Exception as e:
        logging.error(f"Failed to record solution extraction processing steps: {e}")

    output_filename = f'{prefix}_solutions.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        json.dump(solutions_list, f, indent=2, default=convert_Timestamp_to_str)
    logging.info(f"Successfully saved {len(solutions_list)} threads to {full_path}")
    return full_path

def next_thread_gathering(next_batch_df: pd.DataFrame, lookback_date: pd.Timestamp, str_interval: str, global_config: dict):
    """
    Gather next batch of messages into raw threads using LLM from DataFrame provided flat message list.
    if the new messages are related to the previous threads, update the previous threads.
    
    Arguments:
        next_batch_df: DataFrame of next batch of messages
        lookback_date: starting date of the previous threads
        str_interval: string prefix for debug saving
        global_config: global configuration dictionary
    Returns:
        full_path: path to the saved threads in json format
    """
    
    prompts.reload_prompts()
    # set of valid message ids for validation of LLM response
    valid_ids_set = set(next_batch_df['Message ID'].astype(str))
    db_service = get_database_service()
    # get new messages parts
    new_raw_messages_parts, image_genai_cache = parts_from_dataframe(next_batch_df, global_config=global_config)
    
    # start creating parts: add prompt for LLM to parts
    parts = [
        make_text_part(prompts.system_prompt), 
        new_raw_messages_parts,
        make_text_part(prompts.prompt_gathering_logic),
        make_text_part("Also you already have gathered threads from previous days:")
        ]
    # get previous threads
    start_date = next_batch_df['DateTime'].min().normalize()
    # Make timezone-aware to match database datetime format
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=datetime.timezone.utc)
    with db_service.get_session() as session:
        lookback_threads = db_service.get_lookback_threads(session, lookback_date)
        for thread in lookback_threads:
            thread_dict = {
                'Topic id': thread.topic_id,
                'Header': thread.header,
                'Actual Date': thread.actual_date.isoformat() if thread.actual_date else None,
                'Answer Id': thread.answer_id,
                'Label': thread.label,
                'Solution': thread.solution,
                'Status': thread.status,
                'is_technical': thread.is_technical,
                # 'is_processed': thread.is_processed,
                # 'whole_thread': [{'message_id': msg.message_id, 'parent_id': msg.parent_id} for msg in thread.messages]
            }
            # add previous threads headers to parts
            parts.append(make_json_part(thread_dict))
            # add previous threads messages to parts: get messages from database
            db_parts, image_genai_cache = parts_from_db_messages(thread.topic_id,global_config=global_config, session=session)
            parts.extend(db_parts)
            # add previous threads message ids to valid message ids set
            valid_ids_set.update([msg.message_id for msg in thread.messages])
        # add additional prompt for LLM to parts: describe task to recompile previous threads with new messages from next_batch_df
        additional_prompt = make_text_part(prompts.prompt_addition_step1.format(start_date=start_date.isoformat()))
        parts.append(additional_prompt)
    
    # generate response from LLM

    logging.info(f"Next step 1. Processing next {len(next_batch_df)} raw messages...")
    if image_genai_cache:
        gemini_service.config_addition_step1.cached_content = image_genai_cache

    response = gemini_service.generate_content(
        contents= parts,
        config=gemini_service.config_addition_step1,
        valid_ids_set=valid_ids_set,
        log_prefix="Next thread gathering:"
    )
    # get only new and modified threads from response
    added_threads_pydantic = []
    cnt_modified = 0
    cnt_new = 0
    with db_service.get_session() as session:
        for thread in response.threads:
            messages = db_service.get_messages_by_list([m.message_id for m in thread.whole_thread], session=session)
            min_datetime = min([me.datetime for me in messages])
            max_datetime = max([me.datetime for me in messages])
        
            # thread.whole_thread = messages
            if min_datetime < start_date and max_datetime > start_date:
                thread.status = ThreadStatus.MODIFIED
                added_threads_pydantic.append(thread)
                cnt_modified += 1
            elif min_datetime > start_date:
                thread.status = ThreadStatus.NEW
                added_threads_pydantic.append(thread)
                cnt_new += 1
            else:
                thread.status = ThreadStatus.PERSISTED
        
    logging.info(f"Added {cnt_new} new threads. Modified {cnt_modified} threads created before {start_date}.")

    # Create/Update Thread objects in database
    # processing_tracker = get_processing_tracker()
    cnt_db_created = 0
    cnt_db_updated = 0
    
    try:
        with db_service.get_session() as session:###############################
            message_to_thread = {}
            message_to_parent = {}
            
            for thread in added_threads_pydantic:
                thread_id = thread.topic_id
                
                # Check if thread already exists
                from app.models.db_models import Thread
                existing_thread = session.query(Thread).filter(Thread.topic_id == thread_id).first()
                
                if thread.status == ThreadStatus.NEW and not existing_thread:
                    # Create new Thread object in database
                    try:
                        thread_messages = [str(msg.message_id) for msg in thread.whole_thread]
                        cnt_db_created += _create_thread_object(
                            session, 
                            thread_id, 
                            thread_messages, 
                        )
                        
                    except Exception as e:
                        logging.warning(f"Could not create Thread object for {thread_id}: {e}")
                        
                elif thread.status == ThreadStatus.MODIFIED and existing_thread:
                    # Update existing thread
                    try:
                        existing_thread.status = ThreadStatus.MODIFIED
                        existing_thread.updated_at = func.now()
                        cnt_db_updated += 1
                        logging.info(f"Updated Thread object for {thread_id}")
                        
                    except Exception as e:
                        logging.warning(f"Could not update Thread object for {thread_id}: {e}")
                
                # Update parent_id relationships from LLM analysis (same as first_thread_gathering)
                for msg in thread.whole_thread:
                    message_to_thread[str(msg.message_id)] = thread_id
                    if msg.parent_id:
                        message_to_parent[str(msg.message_id)] = str(msg.parent_id)
                    else:
                        message_to_parent[str(msg.message_id)] = ''
            # Update message parent_id and thread_id assignments
            parent_id_updates = 0
            thread_assignment_updates = 0
            for message_id_str, thread_id in message_to_thread.items():
                db_message = session.query(Message).filter(Message.message_id == message_id_str).first()
                
                if db_message:
                    # Update thread assignment
                    if db_message.thread_id != thread_id:
                        db_message.thread_id = thread_id
                        thread_assignment_updates += 1
                    
                    # Update parent_id from LLM analysis
                    llm_parent_id = message_to_parent.get(message_id_str)
                    if llm_parent_id and llm_parent_id != db_message.parent_id:
                        old_parent = db_message.parent_id
                        db_message.parent_id = llm_parent_id
                        parent_id_updates += 1
                        logging.info(f"Updated parent_id for {message_id_str}: {old_parent} -> {llm_parent_id}")
                        
                    # Record processing step for tracking
                    # processing_tracker.record_processing_step(
                    #     session=session,
                    #     message_id=db_message.message_id,
                    #     processing_step=processing_tracker.STEPS['THREAD_GROUPING'],
                    #     result={
                    #         'assigned_to_thread': thread_id,
                    #         'thread_grouping_method': 'llm_incremental_analysis',
                    #         'parent_id_updated': llm_parent_id != old_parent if 'old_parent' in locals() else False,
                    #         'new_parent_id': llm_parent_id
                    #     },
                    #     metadata={'batch_prefix': str_interval, 'processing_type': 'incremental_batch'}
                    # )
            
            ### changes and additions to session are not saved here
            session.commit()
            logging.info(f"Next thread gathering database updates:")
            logging.info(f"  🆕 Created Thread objects: {cnt_db_created}")
            logging.info(f"  🔄 Updated Thread objects: {cnt_db_updated}")
            logging.info(f"  🧵 Updated thread assignments: {thread_assignment_updates}")
            logging.info(f"  🔗 Updated parent_id relationships: {parent_id_updates}")
            
    except Exception as e:
        logging.exception(f"Failed to update database during next thread gathering: {e}")

    output_filename = f'next_{str_interval}_group.json'
    save_path = global_config[f'SAVE_PATH']
    full_path = os.path.join(save_path, output_filename)
    added_threads = [t.model_dump() for t in added_threads_pydantic]
    with open(full_path, 'w') as f:
        json.dump(added_threads, f, indent=2, default=convert_Timestamp_to_str)
    logging.info(f"Successfully saved {len(added_threads)} new/modified threads to {full_path}")
    return full_path

if __name__=="__main__":
    test_filename = generalization_solution("C:\\VSCode\\scraping\\results\\first_2025-07-19-2025-07-21_technical.json","test_solution_2025-07-19-2025-07-21",'./results')
    print(test_filename)