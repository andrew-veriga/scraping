import json
import re
from datetime import datetime, timezone
import os
import pandas as pd
import logging

from app.services import gemini_service
from app.services.rag_service import get_rag_service
from app.services.database import get_database_service
from app.models import pydantic_models
from app.utils.file_utils import * # illustrated_message, illustrated_threads, save_solutions_dict, load_solutions_dict, create_dict_from_list, add_new_solutions_to_dict, add_or_update_solution, convert_datetime_to_str

def first_thread_gathering(logs_df, save_path):
    """
    Group messages into threads using LLM from DataFrame provided flat message list
    """
    logs_csv = logs_df.to_csv(index=False)
    response = gemini_service.generate_content(
        contents=[
            logs_csv,
            gemini_service.system_prompt,
            gemini_service.prompt_start_step_1
            ],
        config=gemini_service.config_step1,
    )
    output_filename = f'first_group_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'

    threads_list_dict = [thread.model_dump() for thread in response.threads]

    full_path = os.path.join(save_path, output_filename)
    with open(full_path, 'w') as f:
        json.dump(threads_list_dict, f, indent=4, default=convert_datetime_to_str)
    logging.info(f"Start step 1. Processed {len(logs_df)} messages")
    logging.info(f"Successfully saved {len(response.threads)} grouped threads to {output_filename}")
    return full_path



def filter_technical_topics(filename, startnext: str, messages_df, save_path):
    """
    Extract technical topics from JSON file using LLM with DataFrame logs provided full texts of messages
    return only technical threads
    """
    with open(filename, 'r') as f:
        threads_json_data = json.load(f)

    processed_threads = illustrated_threads(threads_json_data, messages_df)

    response = gemini_service.generate_content(
        contents=[
            json.dumps(processed_threads, indent=4),
            gemini_service.system_prompt,
            gemini_service.prompt_step_2
            ],
        config=gemini_service.config_step2,
    )

    technical_threads = [thread for thread in processed_threads if thread['Topic_ID'] in response.technical_topics]
    logging.info(f"{startnext} step 2. Selected {len(response.technical_topics)} technical topics from {len(processed_threads)} threads")

    output_filename = f'{startnext}_technical_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        f.write(json.dumps(technical_threads,  indent=2))

    logging.info(f"Successfully saved {len(technical_threads)} filtered threads to {output_filename}")
    return full_path

def generalization_solution(filename,startnext: str, save_path):
    """
    Generalize technical topics from JSON file using LLM
    Create "Header" and "Solution" fields
    """
    with open(filename, 'r') as f:
        technical_threads = json.load(f)

    response_solutions = gemini_service.generate_content(
        contents=[
            json.dumps(technical_threads, indent=2),
            gemini_service.system_prompt,
            gemini_service.prompt_step_3
            ],
        config=gemini_service.solution_config,
    )
    logging.info(f"{startnext} step 3. Generalization of {len(response_solutions.threads)} technical threads")
    solutions_list = [thread.model_dump() for thread in response_solutions.threads]

    output_filename = f'{startnext}_solutions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    full_path = os.path.join(save_path,output_filename)
    with open(full_path, 'w') as f:
        json.dump(solutions_list, f, indent=2, default=convert_datetime_to_str)
    logging.info(f"Successfully saved {len(solutions_list)} threads to {output_filename}")
    return full_path
# solutions_dict item:
#   "1404142135313174578": {
#     "Header": "User's USDC deposit in a Suidollar.io dapp vault is not appearing, despite the transaction being correct on Suivision.xyz.",
#     "Topic_ID": "1404142135313174578",
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
def next_thread_gathering(next_batch_df, solutions_dict, lookback_date, next_start_date, save_path, messages_df):
    """
    Gather next batch of threads for processing
    """
    df_indexed = messages_df.copy()
    author_ids = messages_df['Author ID'].unique().tolist()
    df_indexed.set_index('Message ID', inplace=True)
    
    next_batch_csv = next_batch_df.to_csv(index=False)

    technical_threads_json = [t for t in solutions_dict.values() if pd.Timestamp(t['Actual_Date']) > lookback_date]
    # previous_threads_json = illustrated_threads(technical_threads_json, messages_df)
    previous_threads_text = []
    for thread in technical_threads_json:
        whole_thread_ids = thread.get('Whole_thread', [])
        if whole_thread_ids:
            messages=[]
            for message_id in whole_thread_ids:
                # Ensure message_id is a string for lookup
                # Assuming formatted_message function is defined and accessible
                message = df_indexed.loc[message_id]
                message_content =illustrated_message(message_id, df_indexed)
                if message_id == thread.get('Topic_ID', 'N/A'):
                    messages.append(f"""- ({message_id}) - **Topic started** :{message_content} """)
                else:
                    messages.append(f"- ({message_id}) {message_content} ")

        previous_threads_text.append(f"Topic: {thread.get('Topic_ID', 'N/A')} - {thread.get('Actual_Date', 'N/A')} \\n" + "\n".join(messages))

    prmpt = gemini_service.prompt_addition_step1.format(JSON_prev="\n".join(previous_threads_text))

    logging.info(f"Next step 1. Processing next {len(next_batch_df)} raw messages...")
    response = gemini_service.generate_content(
        contents=[
            gemini_service.system_prompt,
            next_batch_csv,
            prmpt
            ],
        config=gemini_service.config_addition_step1,
    )
    added_threads_pydantic = [t for t in response.threads if t.status !='persisted']
    added_threads = [t.model_dump() for t in added_threads_pydantic]
    cnt_modified = len([t for t in added_threads if t['status']=='modified'])
    cnt_new = len([t for t in added_threads if t['status']=='new'])

    logging.info(f"Added {cnt_new} new threads. Modified {cnt_modified} threads created before {next_start_date.strftime('%Y-%m-%d_%H-%M-%S')}.")

    output_filename = f'next_group_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
    full_path = os.path.join(save_path, output_filename)
    with open(full_path, 'w') as f:
        json.dump(added_threads, f, indent=4, default=convert_datetime_to_str)
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

    # dictionaries for a all previous solutions  and new batch of solutions
    prev_solution_dict = {
        topic_id: solution
        for topic_id, solution in solutions_dict.items()
        if pd.Timestamp(solution["Actual_Date"]).tz_convert(lookback_date.tzinfo) >= lookback_date
    }
    new_solution_dict = create_dict_from_list(next_solutions_list)
    # list of modified thread IDs
    with open(next_technical_filename, 'r') as f:
        adding_threads = json.load(f)

    modified_threads = [t['Topic_ID'] for t in adding_threads if t['status']=='modified']

    new_threads = [t['Topic_ID'] for t in adding_threads if t['status']=='new']
    if len(modified_threads) > 0:
        # pairs of old and modified sloutions
        logging.info(f"{len(modified_threads)} comparising")
        modified_pairs = {m: {'prev': prev_solution_dict[m],'new':new_solution_dict[m]} for m in modified_threads}
        pairs_in_text = []
        for key, p in modified_pairs.items():
            pairs_in_text.append(f"""
Topic_ID: {key}
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
                gemini_service.system_prompt,
                gemini_service.revision_prompt.format(pairs='\n'.join(pairs_in_text))
                ],
            config=gemini_service.revision_config,
        )

        revised_solutions=response_solutions.comparisions

        for s in revised_solutions:
            if  s.Label=='improved': #thread has significant improved solution, should be replaced in the main dictionary
                add_or_update_solution(solutions_dict, new_solution_dict[s.Topic_ID])
                logging.info(f'Topic {s.Topic_ID} improved')
            elif s.Label=='persisted': #thread persists the header and solution text, but should change the message list
                solutions_dict[s.Topic_ID]['Whole_thread'] = new_solution_dict[s.Topic_ID]['Whole_thread']
                logging.info(f"Topic {s.Topic_ID} get {len(new_solution_dict[s.Topic_ID]['Whole_thread']) - len(solutions_dict[s.Topic_ID]['Whole_thread'])} new messages")
            else:
                # s.Label=='changed': thread has changes in header and solution text. Should be checked in RAG
                changed_solution = new_solution_dict[s.Topic_ID]
                rag_check_result = _check_solution_with_rag(changed_solution, exclude_solution_id=s.Topic_ID)
                
                if rag_check_result['recommendation'] == 'add':
                    new_threads.append(s.Topic_ID)
                    logging.info(f'Topic {s.Topic_ID} marked as changed but unique enough to add: {rag_check_result["reason"]}')
                elif rag_check_result['recommendation'] == 'merge':
                    # Merge with existing similar solution
                    _merge_similar_solutions(solutions_dict, s.Topic_ID, changed_solution, rag_check_result['similar_solutions'])
                    logging.info(f'Topic {s.Topic_ID} merged with similar existing solution')
                else:
                    # Skip or review - log the decision
                    logging.info(f'Topic {s.Topic_ID} skipped due to RAG check: {rag_check_result["reason"]}')
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
                    actual_date = solution_data.get('Actual_Date')
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