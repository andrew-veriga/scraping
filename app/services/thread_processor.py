import json
import re
from datetime import datetime, timezone
import os
import pandas as pd
import logging

from app.services import gemini_service, prompts
from app.services.rag_service import get_rag_service
from app.services.database import get_database_service
from app.services.processing_tracker import get_processing_tracker
from app.models import pydantic_models
from app.models.db_models import Message, Thread, Solution
from app.utils.file_utils import * # illustrated_message, illustrated_threads, save_solutions_dict, load_solutions_dict, create_dict_from_list, add_new_solutions_to_dict, add_or_update_solution, convert_datetime_to_str


def first_thread_gathering(logs_df, prefix, save_path):
    """
    Group messages into threads using LLM from DataFrame provided flat message list
    """
    logs_csv = logs_df.to_csv(index=False)
    valid_ids_set = set(logs_df['Message ID'].unique())
    
    # Record the start of thread grouping processing
    processing_tracker = get_processing_tracker()
    db_service = get_database_service()
    
    response = gemini_service.generate_content(
        contents=[
            logs_csv,
            prompts.system_prompt,
            prompts.prompt_start_step_1
            ],
        config=gemini_service.config_step1,
        valid_ids_set=valid_ids_set,
        log_prefix="First thread gathering:"
    )
    output_filename = f'{prefix}_gathering.json'

    # The response is now the parsed and validated Pydantic object
    threads_list_dict = [thread.model_dump() for thread in response.threads]

    # Record processing steps for each message
    try:
        with db_service.get_session() as session:
            # Create message ID to thread ID mapping for recording
            message_to_thread = {}
            for thread in response.threads:
                thread_id = thread.topic_id
                for msg_id in thread.whole_thread:
                    message_to_thread[str(msg_id)] = thread_id
            
            # Record thread grouping steps
            for idx, row in logs_df.iterrows():
                message_id_str = str(row['Message ID'])
                
                # Find the database message
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
    with open(filename, 'r') as f:
        threads_json_data = json.load(f)
    
    # Convert to consistent format if needed
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

    # Record technical filtering processing steps
    try:
        with db_service.get_session() as session:
            for thread in processed_threads:
                thread_id = thread.get('topic_id') or thread.get('Topic_ID')
                is_technical = thread_id in response.technical_topics
                
                # Record processing for all messages in thread
                for msg_id in (thread.get('whole_thread') or thread.get('Whole_thread', [])):
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
                        
                        # Add annotation for technical messages
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
    with open(filename, 'r') as f:
        technical_threads = json.load(f)
    
    # Convert to consistent format if needed
    from app.utils.file_utils import convert_legacy_format
    technical_threads = convert_legacy_format(technical_threads)
    valid_ids_set = set()
    for t in technical_threads: 
        # Handle both old and new formats during transition
        whole_thread = t.get('whole_thread') or t.get('Whole_thread', [])
        valid_ids_set = valid_ids_set.union(set(whole_thread))
        
    processing_tracker = get_processing_tracker()
    db_service = get_database_service()
        
    response_solutions = gemini_service.generate_content( 
        contents=[
            json.dumps(technical_threads, indent=2),
            prompts.system_prompt,
            prompts.prompt_step_3
            ],
        config=gemini_service.solution_config,
        valid_ids_set=valid_ids_set,
        log_prefix=f"{prefix} generalization solution:"
    )

    logging.info(f"{prefix} step 3. Generalization of {len(response_solutions.threads)} technical threads")
    solutions_list = [thread.model_dump() for thread in response_solutions.threads]

    # Record solution extraction processing steps
    try:
        with db_service.get_session() as session:
            for solution_thread in response_solutions.threads:
                thread_id = solution_thread.topic_id
                
                # Record processing for all messages in thread
                for msg_id in solution_thread.whole_thread:
                    db_message = session.query(Message).filter(
                        Message.message_id == str(msg_id)
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
                        
                        # Add annotation for solution messages
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
# solutions_dict item:
#   "1404142135313174578": {
#     "Header": "User's USDC deposit in a Suidollar.io dapp vault is not appearing, despite the transaction being correct on Suivision.xyz.",
#     "topic_id": "1404142135313174578",
#     "Actual_Date": "2025-08-10T16:45:05+00:00",
#     "Answer_ID": "1404142439865909278",
#     "Whole_thread": [
#       "1404142135313174578",
#       "1404142439865909278",
#       "1404143532108808243"
#     ],
#     "Label": "outside",
#     "Solution": "Please reach out to the Suidollar.io support team directly for issues with deposits not appearing in their dapp.",
#     "Topic": "2025-08-10 16:39:32 User 752613342467588167: Good afternoon, is there any problem with the suidollar.io dapp? Yesterday I made an approximate deposit of 2700 USDC in the \"Delta Neutral USDC Vault\", and the deposit does not appear. I have reviewed the transaction in Suivision.xyz and everything seems correct. Could you help me? Thank you!"
#   }
def next_thread_gathering(next_batch_df, lookback_threads, str_interval, save_path, messages_df):
    """
    Gather next batch of messages into raw threads for processing
    """
    df_indexed = messages_df.copy()
    df_indexed.set_index('Message ID', inplace=True)
    
    next_batch_csv = next_batch_df.to_csv(index=False)

    
    # previous_threads_json = illustrated_threads(technical_threads_json, messages_df)
    previous_threads_text = []
    valid_ids_set = set(next_batch_df['Message ID'].unique())
    # valid_ids_set = set(df_indexed.index)
    for thread in lookback_threads:
        whole_thread_ids = thread.get('Whole_thread', [])
        messages = []  # Initialize messages before the if statement
        if whole_thread_ids:
            valid_ids_set = valid_ids_set.union(set(whole_thread_ids))
            for message_id in whole_thread_ids:
                # Ensure message_id is a string for lookup
                # Assuming formatted_message function is defined and accessible
                message_content =illustrated_message(message_id, df_indexed)
                if message_id == (thread.get('topic_id') or thread.get('Topic_ID', 'N/A')):
                    messages.append(f"""- ({message_id}) - **Topic started** :{message_content} """)
                else:
                    messages.append(f"- ({message_id}) {message_content} ")

        topic_id = thread.get('topic_id') or thread.get('Topic_ID', 'N/A')
        actual_date = thread.get('actual_date') or thread.get('Actual_Date', 'N/A')
        previous_threads_text.append(f"Topic: {topic_id} - {actual_date} \\n" + "\n".join(messages))

    prmpt = prompts.prompt_addition_step1.format(JSON_prev="\n".join(previous_threads_text))

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

    # The response is now the parsed and validated Pydantic object
    added_threads_pydantic = [t for t in response.threads if t.status !='persisted']
    added_threads = [t.model_dump() for t in added_threads_pydantic]
    cnt_modified = len([t for t in added_threads if t['status']=='modified'])
    cnt_new = len([t for t in added_threads if t['status']=='new'])

    logging.info(f"Added {cnt_new} new threads. Modified {cnt_modified} threads created before {str_interval}.")
    # str_interval = f"{next_start_date.date()}-{next_end_date.date()}"
    output_filename = f'next_{str_interval}_group.json'
    full_path = os.path.join(save_path, output_filename)
    with open(full_path, 'w') as f:
        json.dump(added_threads, f, indent=2, default=convert_datetime_to_str)
    logging.info(f"Successfully saved {len(added_threads)} new/modified threads to {output_filename}")
    return full_path

def new_solutions_revision_and_add(next_solutions_filename,next_technical_filename, solutions_dict:dict, lookback_date)->dict:
    """
    Check improvement of solutions for topics file using LLM
    args:
    next_solutions_filename - file name of saved new solutions
    next_technical_filename - file name of adding raw threads with 'new' and 'modified' status
    """
    with open(next_solutions_filename, 'r') as f:
        next_solutions_list = json.load(f)

    # dictionaries for a all previous solutions from lookback_date and new batch of solutions
    prev_solution_dict = {}
    for topic_id, solution in solutions_dict.items():
        actual_date = solution.get('actual_date') or solution.get('Actual_Date')
        if pd.Timestamp(actual_date).tz_convert(lookback_date.tzinfo) >= lookback_date:
            prev_solution_dict[topic_id] = solution
    new_solution_dict = create_dict_from_list(next_solutions_list)
    # list of modified thread IDs
    with open(next_technical_filename, 'r') as f:
        adding_threads = json.load(f)

    modified_threads = [(t.get('topic_id') or t.get('Topic_ID')) for t in adding_threads if t['status']=='modified']

    new_threads = [(t.get('topic_id') or t.get('Topic_ID')) for t in adding_threads if t['status']=='new']
    if len(modified_threads) > 0:
        # pairs of old and modified sloutions
        logging.info(f"{len(modified_threads)} comparising")
        modified_pairs = {}
        for m in modified_threads:
            if m not in prev_solution_dict:
                logging.warning(f"Topic {m} marked as modified but not found in previous solutions")
                continue
            if m not in new_solution_dict:
                logging.warning(f"Topic {m} marked as modified but not found in new solutions")
                continue
            modified_pairs[m] = {'prev': prev_solution_dict[m],'new':new_solution_dict[m]}
        pairs_in_text = []
        for key, p in modified_pairs.items():
            pairs_in_text.append(f"""
topic_id: {key}
Previous version:
    statement: {p['prev']['Header']}
    solution: {p['prev']['Solution']}
    status: {p['prev']['Label']}
New version:
    statement: {p['new']['Header']}
    solution: {p['new']['Solution']}
    status: {p['new']['Label']}
"""
        )
        # logging.debug("pairs: %s", pairs_in_text)

        response_solutions = gemini_service.generate_content(
            contents=[
                prompts.system_prompt,
                prompts.revision_prompt.format(pairs='\n'.join(pairs_in_text))
                ],
            config=gemini_service.revision_config,
        )

        revised_solutions=response_solutions.comparisions

        for s in revised_solutions:
            if  s.Label=='improved': #thread has significant improved solution, should be replaced in the main dictionary
                add_or_update_solution(solutions_dict, new_solution_dict[s.topic_id])
                logging.info(f'Topic {s.topic_id} improved')
            elif s.Label=='persisted': #thread persists the header and solution text, but should change the message list
                logging.info(f"Topic {s.topic_id} get {len(new_solution_dict[s.topic_id]['whole_thread']) - len(solutions_dict[s.topic_id]['whole_thread'])} new messages")
                solutions_dict[s.topic_id]['whole_thread'] = new_solution_dict[s.topic_id]['whole_thread']
                solutions_dict[s.topic_id]['actual_date'] = new_solution_dict[s.topic_id]['actual_date']
                solutions_dict[s.topic_id]['answer_id'] = new_solution_dict[s.topic_id]['answer_id']
                solutions_dict[s.topic_id]['label'] = new_solution_dict[s.topic_id]['label']
            else:
                # s.Label=='changed': thread has changes in header and solution text. Should be checked in RAG
                changed_solution = new_solution_dict[s.topic_id]
                rag_check_result = _check_solution_with_rag(changed_solution, exclude_solution_id=s.topic_id)
                
                if rag_check_result['recommendation'] == 'add':
                    new_threads.append(s.topic_id)
                    logging.info(f'Topic {s.topic_id} marked as changed but unique enough to add: {rag_check_result["reason"]}')
                elif rag_check_result['recommendation'] == 'merge':
                    # Merge with existing similar solution
                    _merge_similar_solutions(solutions_dict, s.topic_id, changed_solution, rag_check_result['similar_solutions'])
                    logging.info(f'Topic {s.topic_id} merged with similar existing solution')
                elif rag_check_result['recommendation'] == 'mark_duplicate':
                    # Mark as duplicate and add to solutions for tracking
                    add_or_update_solution(solutions_dict, changed_solution)
                    _create_duplicate_records(changed_solution, rag_check_result['similar_solutions'])
                    logging.info(f'Topic {s.topic_id} marked as duplicate: {rag_check_result["reason"]}')
                else:
                    # Skip or review - log the decision
                    logging.info(f'Topic {s.topic_id} skipped due to RAG check: {rag_check_result["reason"]}')
                    pass


    new_solutions_for_add = {key: s for key, s in new_solution_dict.items() if key in new_threads}
    
    # Check each new solution against RAG before adding
    filtered_solutions_for_add = []
    for topic_id, solution_data in new_solutions_for_add.items():
        rag_check_result = _check_solution_with_rag(solution_data)
        
        if rag_check_result['recommendation'] == 'add':
            filtered_solutions_for_add.append(solution_data)
            logging.info(f'New solution {topic_id} approved for addition: {rag_check_result["reason"]}')
        elif rag_check_result['recommendation'] == 'merge':
            # Try to merge with existing similar solution
            _merge_similar_solutions(solutions_dict, topic_id, solution_data, rag_check_result['similar_solutions'])
            logging.info(f'New solution {topic_id} merged with existing solution')
        elif rag_check_result['recommendation'] == 'mark_duplicate':
            # Mark as duplicate and add to solutions for tracking
            filtered_solutions_for_add.append(solution_data)
            _create_duplicate_records(solution_data, rag_check_result['similar_solutions'])
            logging.info(f'New solution {topic_id} marked as duplicate: {rag_check_result["reason"]}')
        else:
            # Skip duplicate or review needed
            logging.info(f'New solution {topic_id} skipped: {rag_check_result["reason"]}')
            if rag_check_result['similar_solutions']:
                similar_topic = rag_check_result['similar_solutions'][0][0].thread.topic_id
                logging.info(f'  Similar existing solution: {similar_topic}')
    
    add_new_solutions_to_dict(solutions_dict, filtered_solutions_for_add)
    logging.info(f"Added {len(filtered_solutions_for_add)} new solutions after RAG filtering")
    
    return solutions_dict


# RAG Helper Functions for the implemented TODOs

def _check_solution_with_rag(solution_data: dict, exclude_solution_id: str = None) -> dict:
    """
    Check solution against existing solutions using RAG similarity search.
    
    Args:
        solution_data: Dictionary containing Header, Solution, and Label
        exclude_solution_id: Topic ID to exclude from similarity search
        
    Returns:
        Dictionary with recommendation, reason, and similar_solutions
    """
    try:
        rag_service = get_rag_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # Get the database solution ID if excluding
            exclude_db_solution_id = None
            if exclude_solution_id:
                thread = db_service.get_thread_by_topic_id(session, exclude_solution_id)
                if thread and thread.solutions:
                    exclude_db_solution_id = thread.solutions[0].id
            
            # Check solution uniqueness using RAG
            result = rag_service.check_solution_uniqueness(
                session=session,
                header=solution_data.get('Header', ''),
                solution_text=solution_data.get('Solution', ''),
                label=solution_data.get('Label', ''),
                exclude_solution_id=exclude_db_solution_id
            )
            
            return result
            
    except Exception as e:
        logging.error(f"RAG check failed: {e}")
        # Fallback to allowing the solution
        return {
            'is_unique': True,
            'similar_solutions': [],
            'recommendation': 'add',
            'reason': f'RAG check failed, allowing by default: {e}'
        }


def _merge_similar_solutions(solutions_dict: dict, topic_id: str, new_solution_data: dict, 
                           similar_solutions: list):
    """
    Merge new solution with most similar existing solution.
    
    Args:
        solutions_dict: Current solutions dictionary
        topic_id: Topic ID of new solution
        new_solution_data: New solution data
        similar_solutions: List of (Solution, similarity_score) tuples
    """
    if not similar_solutions:
        return
        
    try:
        # Get the most similar solution
        most_similar_solution, similarity_score = similar_solutions[0]
        existing_topic_id = most_similar_solution.thread.topic_id
        
        if existing_topic_id in solutions_dict:
            existing_solution = solutions_dict[existing_topic_id]
            
            # Merge strategies based on similarity score
            if similarity_score >= 0.95:
                # Very similar - just update message list
                new_messages = set(new_solution_data.get('Whole_thread', []))
                existing_messages = set(existing_solution.get('Whole_thread', []))
                combined_messages = list(existing_messages.union(new_messages))
                existing_solution['Whole_thread'] = combined_messages
                
                logging.info(f"Merged {topic_id} into {existing_topic_id}: updated message list")
                
            elif similarity_score >= 0.85:
                # Similar - combine solutions
                new_solution_text = new_solution_data.get('Solution', '')
                existing_solution_text = existing_solution.get('Solution', '')
                
                # Combine solutions if they add different information
                if len(new_solution_text) > len(existing_solution_text):
                    existing_solution['Solution'] = new_solution_text
                    existing_solution['Header'] = new_solution_data.get('Header', existing_solution['Header'])
                    logging.info(f"Merged {topic_id} into {existing_topic_id}: updated solution text")
                
                # Always merge message lists
                new_messages = set(new_solution_data.get('Whole_thread', []))
                existing_messages = set(existing_solution.get('Whole_thread', []))
                combined_messages = list(existing_messages.union(new_messages))
                existing_solution['Whole_thread'] = combined_messages
                
    except Exception as e:
        logging.error(f"Failed to merge solutions: {e}")


def _create_duplicate_records(solution_data: dict, similar_solutions: list):
    """
    Create duplicate records for a solution that has been marked as a duplicate.
    
    Args:
        solution_data: Dictionary containing solution information (topic_id, header, solution, etc.)
        similar_solutions: List of (Solution, similarity_score) tuples from RAG search
    """
    try:
        if not similar_solutions:
            logging.warning(f"No similar solutions provided for duplicate marking of {solution_data.get('topic_id') or solution_data.get('Topic_ID')}")
            return
        
        rag_service = get_rag_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # Get the database thread for this solution
            topic_id = solution_data.get('topic_id') or solution_data.get('Topic_ID')
            thread = db_service.get_thread_by_topic_id(session, topic_id)
            
            if not thread:
                logging.error(f"Thread not found in database for topic_id: {topic_id}")
                return
            
            # Get or create the solution in the database
            solution = session.query(Solution).filter(Solution.thread_id == thread.id).first()
            if not solution:
                # Create the solution
                solution_db_data = {
                    'thread_id': thread.id,
                    'header': solution_data.get('Header', ''),
                    'solution': solution_data.get('Solution', ''),
                    'label': solution_data.get('Label', 'unresolved'),
                    'is_duplicate': True
                }
                solution = db_service.create_solution(session, solution_db_data)
                session.flush()
            else:
                # Mark existing solution as duplicate
                solution.is_duplicate = True
                session.flush()
            
            # Create duplicate record with the most similar solution
            most_similar_solution, highest_similarity = similar_solutions[0]
            
            duplicate_record = rag_service.create_duplicate_record(
                session=session,
                solution_id=solution.id,
                original_solution_id=most_similar_solution.id,
                similarity_score=highest_similarity
            )
            
            if duplicate_record:
                logging.info(f"Created duplicate record for {topic_id} -> {most_similar_solution.thread.topic_id} (similarity: {highest_similarity:.3f})")
            
            session.commit()
            
    except Exception as e:
        logging.error(f"Failed to create duplicate records: {e}")


def _update_database_with_solutions(solutions_dict: dict):
    """
    Update database with solutions from the solutions dictionary.
    This helps maintain consistency between file-based and database storage.
    """
    try:
        db_service = get_database_service()
        rag_service = get_rag_service()
        
        with db_service.get_session() as session:
            for topic_id, solution_data in solutions_dict.items():
                # Get or create thread
                thread = db_service.get_thread_by_topic_id(session, topic_id)
                if not thread:
                    # Create thread if it doesn't exist
                    actual_date = solution_data.get('actual_date') or solution_data.get('Actual_Date')
                    if isinstance(actual_date, str):
                        actual_date = pd.to_datetime(actual_date)
                    
                    thread_data = {
                        'topic_id': topic_id,
                        'header': solution_data.get('Header'),
                        'actual_date': actual_date,
                        'answer_id': solution_data.get('Answer_ID'),
                        'label': solution_data.get('Label'),
                        'solution': solution_data.get('Solution'),
                        'status': 'new',
                        'is_technical': True,
                        'is_processed': True
                    }
                    thread = db_service.create_thread(session, thread_data)
                    session.flush()
                
                # Update or create solution
                from app.models.db_models import Solution
                existing_solution = session.query(Solution).filter(
                    Solution.thread_id == thread.id
                ).first()
                
                if existing_solution:
                    # Update existing solution
                    db_service.update_solution(session, thread.id, {
                        'header': solution_data.get('Header', ''),
                        'solution': solution_data.get('Solution', ''),
                        'label': solution_data.get('Label', 'unresolved')
                    })
                else:
                    # Create new solution
                    solution_db_data = {
                        'thread_id': thread.id,
                        'header': solution_data.get('Header', ''),
                        'solution': solution_data.get('Solution', ''),
                        'label': solution_data.get('Label', 'unresolved')
                    }
                    solution = db_service.create_solution(session, solution_db_data)
                    
                    # Generate embedding for new solution
                    rag_service.add_solution_embedding(session, solution)
            
            session.commit()
            logging.info("Database synchronized with solutions dictionary")
            
    except Exception as e:
        logging.error(f"Failed to update database with solutions: {e}")