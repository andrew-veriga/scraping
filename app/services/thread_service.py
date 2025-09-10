
import json
import re
from datetime import datetime, timezone
import os
from typing import Optional
import pandas as pd
import logging

from app.services import gemini_service, prompts
from app.services.processing_tracker import get_processing_tracker
from app.services.database import get_database_service
from app.models.pydantic_models import ThreadStatus
from app.models.db_models import Message
from app.utils.file_utils import *

def first_thread_gathering(logs_df, prefix, save_path): 
    """
    Group messages into threads using LLM from DataFrame provided flat message list
    """
    prompts.reload_prompts()
    logs_csv = logs_df.to_csv(index=False)
    valid_ids_set = set(logs_df['Message ID'].astype(str))

    processing_tracker = get_processing_tracker()
    db_service = get_database_service()
    
    response = gemini_service.generate_content(
        contents=[
            logs_csv,
            prompts.system_prompt,
            prompts.prompt_start_step_1.format(prompt_group_logic=prompts.prompt_group_logic)
            ],
        config=gemini_service.config_step1,
        valid_ids_set=valid_ids_set,
        log_prefix="First thread gathering:"
    )
    output_filename = f'{prefix}_gathering.json'

    threads_list_dict = [thread.model_dump() for thread in response.threads]

    try:
        with db_service.get_session() as session:
            message_to_thread = {}
            for thread in response.threads:
                thread_id = thread.topic_id
                for msg in thread.whole_thread:
                    message_to_thread[str(msg.message_id)] = thread_id
            
            for idx, row in logs_df.iterrows():
                message_id_str = str(row['Message ID'])
                
                db_message = session.query(Message).filter(
                    Message.message_id == message_id_str
                ).first()
                
                if db_message:
                    assigned_thread = message_to_thread.get(message_id_str)
                    result = {
                        'assigned_to_thread': assigned_thread,
                        'thread_grouping_method': 'llm_analysis',
                        'total_threads_created': len(response.threads)
                    }
                    
                    processing_tracker.record_processing_step(
                        session=session,
                        message_id=db_message.id,
                        processing_step=processing_tracker.STEPS['THREAD_GROUPING'],
                        result=result,
                        metadata={'batch_prefix': prefix, 'processing_type': 'first_batch'}
                    )
            
            session.commit()
            logging.info(f"Recorded thread grouping processing steps for {len(logs_df)} messages")
            
    except Exception as e:
        logging.error(f"Failed to record thread grouping processing steps: {e}")

    full_path = os.path.join(save_path, output_filename)
    with open(full_path, 'w') as f:
        json.dump(threads_list_dict, f, indent=4, default=convert_datetime_to_str)
    logging.info(f"Start step 1. Processed {len(logs_df)} messages")
    logging.info(f"Successfully saved {len(response.threads)} grouped threads to {output_filename}")
    return full_path

def filter_technical_topics(filename, prefix: str, messages_df, save_path):
    """
    Extract technical topics from JSON file using LLM with DataFrame logs provided full texts of messages
    return only technical threads
    """
    prompts.reload_prompts()
    with open(filename, 'r') as f:
        threads_json_data = json.load(f)
    
    from app.utils.file_utils import convert_legacy_format
    threads_json_data = convert_legacy_format(threads_json_data)

    processed_threads = illustrated_threads(threads_json_data, messages_df)
    processing_tracker = get_processing_tracker()
    db_service = get_database_service()

    response = gemini_service.generate_content(
        contents=[
            json.dumps(processed_threads, indent=4),
            prompts.system_prompt,
            prompts.prompt_step_2
            ],
        config=gemini_service.config_step2,
    )

    technical_threads = [thread for thread in processed_threads if (thread.get('topic_id') or thread.get('Topic_ID')) in response.technical_topics]
    logging.info(f"{prefix} step 2. Selected {len(response.technical_topics)} technical topics from {len(processed_threads)} threads")

    try:
        with db_service.get_session() as session:
            for thread in processed_threads:
                thread_id = thread.get('topic_id') or thread.get('Topic_ID')
                is_technical = thread_id in response.technical_topics
                
                whole_thread = (thread.get('whole_thread') or thread.get('Whole_thread', []))
                for msg in whole_thread:
                    msg_id = str(msg['message_id'])
                    db_message = session.query(Message).filter(
                        Message.message_id == str(msg_id)
                    ).first()
                    
                    if db_message:
                        result = {
                            'is_technical': is_technical,
                            'thread_id': thread_id,
                            'technical_keywords_found': thread.get('technical_indicators', []),
                            'filtering_method': 'llm_analysis'
                        }
                        
                        processing_tracker.record_processing_step(
                            session=session,
                            message_id=db_message.id,
                            processing_step=processing_tracker.STEPS['TECHNICAL_FILTERING'],
                            result=result,
                            metadata={'batch_prefix': prefix}
                        )
                        
                        if is_technical:
                            processing_tracker.annotate_message(
                                session=session,
                                message_id=db_message.id,
                                annotation_type=processing_tracker.ANNOTATION_TYPES['TECHNICAL'],
                                annotation_value={'thread_id': thread_id, 'indicators': thread.get('technical_indicators', [])},
                                annotated_by='gemini_ai'
                            )
            
            session.commit()
            logging.info(f"Recorded technical filtering processing steps")
            
    except Exception as e:
        logging.error(f"Failed to record technical filtering processing steps: {e}")

    output_filename = f'{prefix}_technical.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        f.write(json.dumps(technical_threads,  indent=2))

    logging.info(f"Successfully saved {len(technical_threads)} filtered threads to {output_filename}")
    return full_path

def generalization_solution(filename,prefix: str, save_path):
    """
    Generalize technical topics from JSON file using LLM
    Create "Header" and "Solution" fields
    """
    prompts.reload_prompts()
    with open(filename, 'r') as f:
        technical_threads = json.load(f)
    technical_threads = {t.get('topic_id'): t for t in technical_threads}

    from app.utils.file_utils import convert_legacy_format
    technical_threads = convert_legacy_format(technical_threads)
    valid_ids_set = set()
    text_threads = {}
    for key,t in technical_threads.items(): 
        whole_thread = t.get('whole_thread', [])
        valid_ids_set = valid_ids_set.union(set([msg['message_id'] for msg in whole_thread]))
        text_threads[key] = t['whole_thread_formatted']

    processing_tracker = get_processing_tracker()
    db_service = get_database_service()
        
    response_solutions = gemini_service.generate_content( 
        contents=[
            json.dumps(text_threads, indent=2),
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
            for solution_thread in response_solutions.threads:
                solution=solution_thread.model_dump()
                thread_id = solution_thread.topic_id
                whole_thread = technical_threads[thread_id]['whole_thread']
                solution['whole_thread'] = whole_thread
                solutions_list.append(solution)

                for msg in whole_thread:
                    msg_id = str(msg['message_id'])
                    db_message = session.query(Message).filter(
                        Message.message_id == msg_id
                    ).first()
                    
                    if db_message:
                        result = {
                            'solution_generated': True,
                            'thread_id': thread_id,
                            'solution_header': solution_thread.header,
                            'solution_label': solution_thread.label,
                            'extraction_method': 'llm_analysis'
                        }
                        
                        processing_tracker.record_processing_step(
                            session=session,
                            message_id=db_message.id,
                            processing_step=processing_tracker.STEPS['SOLUTION_EXTRACTION'],
                            result=result,
                            metadata={'batch_prefix': prefix, 'solution_id': thread_id}
                        )
                        
                        if msg_id == solution_thread.answer_id:
                            processing_tracker.annotate_message(
                                session=session,
                                message_id=db_message.id,
                                annotation_type=processing_tracker.ANNOTATION_TYPES['SOLUTION'],
                                annotation_value={'is_answer': True, 'thread_id': thread_id},
                                annotated_by='gemini_ai'
                            )
            
            session.commit()
            logging.info(f"Recorded solution extraction processing steps")
            
    except Exception as e:
        logging.error(f"Failed to record solution extraction processing steps: {e}")

    output_filename = f'{prefix}_solutions.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        json.dump(solutions_list, f, indent=2, default=convert_datetime_to_str)
    logging.info(f"Successfully saved {len(solutions_list)} threads to {output_filename}")
    return full_path

def next_thread_gathering(next_batch_df, lookback_threads, str_interval, save_path, messages_df):
    """
    Gather next batch of messages into raw threads for processing
    """
    prompts.reload_prompts()
    
    next_batch_csv = next_batch_df.to_csv(index=False)

    previous_threads_text = []
    valid_ids_set = set(next_batch_df['Message ID'].astype(str))

    for thread in lookback_threads:
        whole_thread = thread.get('Whole_thread', [])
        whole_thread_ids = [msg['message_id'] for msg in whole_thread]
        messages = []
        if whole_thread_ids:
            valid_ids_set = valid_ids_set.union(set(whole_thread_ids))
            for message_id in whole_thread_ids:
                message_content = illustrated_message(message_id, messages_df)
                if message_id == (thread.get('topic_id') or thread.get('Topic_ID', 'N/A')):
                    messages.append(f"- ({message_id}) - **Topic started** :{message_content} ")
                else:
                    messages.append(f"- ({message_id}) {message_content} ")

        topic_id = thread.get('topic_id') or thread.get('Topic_ID', 'N/A')
        actual_date = thread.get('actual_date') or thread.get('Actual_Date', 'N/A')
        previous_threads_text.append(f"Topic: {topic_id} - {actual_date} \n" + "\n".join(messages))

    prmpt = prompts.prompt_addition_step1.format(
        prompt_group_logic=prompts.prompt_group_logic,
        JSON_prev="\n".join(previous_threads_text)
    )

    logging.info(f"Next step 1. Processing next {len(next_batch_df)} raw messages...")
    response = gemini_service.generate_content(
        contents=[
            prompts.system_prompt,
            next_batch_csv,
            prmpt
            ],
        config=gemini_service.config_addition_step1,
        valid_ids_set=valid_ids_set,
        log_prefix="Next thread gathering:"
    )

    added_threads_pydantic = [t for t in response.threads if t.status !=ThreadStatus.PERSISTED]
    added_threads = [t.model_dump() for t in added_threads_pydantic]
    cnt_modified = len([t for t in added_threads if t['status']==ThreadStatus.MODIFIED])
    cnt_new = len([t for t in added_threads if t['status']==ThreadStatus.NEW])

    logging.info(f"Added {cnt_new} new threads. Modified {cnt_modified} threads created before {str_interval}.")

    output_filename = f'next_{str_interval}_group.json'
    full_path = os.path.join(save_path, output_filename)
    with open(full_path, 'w') as f:
        json.dump(added_threads, f, indent=2, default=convert_datetime_to_str)
    logging.info(f"Successfully saved {len(added_threads)} new/modified threads to {output_filename}")
    return full_path
