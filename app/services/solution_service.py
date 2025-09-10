import json
import logging
import pandas as pd

from app.services import prompts
from app.services.rag_service import get_rag_service
from app.services.database import get_database_service
from app.models.pydantic_models import RevisedStatus, ThreadStatus, SolutionStatus
from app.utils.file_utils import add_or_update_solution, create_dict_from_list, add_new_solutions_to_dict
from app.services import gemini_service
from app.models.db_models import Solution

def new_solutions_revision_and_add(next_solutions_filename,next_technical_filename, solutions_dict:dict, lookback_date)->dict:
    """
    Check improvement of solutions for topics file using LLM
    args:
    next_solutions_filename - file name of saved new solutions
    next_technical_filename - file name of adding raw threads with ThreadStatus.NEW and ThreadStatus.MODIFIED status
    """
    prompts.reload_prompts()
    with open(next_solutions_filename, 'r') as f:
        next_solutions_list = json.load(f)

    prev_solution_dict = {}
    for topic_id, solution in solutions_dict.items():
        actual_date = solution.get('actual_date') or solution.get('Actual_Date')
        if pd.Timestamp(actual_date).tz_convert(lookback_date.tzinfo) >= lookback_date:
            prev_solution_dict[topic_id] = solution
    new_solution_dict = create_dict_from_list(next_solutions_list)

    with open(next_technical_filename, 'r') as f:
        adding_threads = json.load(f)

    modified_threads = [t.get('topic_id') for t in adding_threads if t['status']==ThreadStatus.MODIFIED]

    new_threads = [t.get('topic_id') for t in adding_threads if t['status']==ThreadStatus.NEW]
    if len(modified_threads) > 0:

        logging.info(f"{len(modified_threads)} comparising")
        modified_pairs = {}
        for m in modified_threads:
            if m not in prev_solution_dict:
                logging.warning(f"Topic {m} marked as modified but not found in previous solutions")
                continue
            if m not in new_solution_dict:
                logging.warning(f"Topic {m} marked as modified but not found in new solutions")
                continue
            modified_pairs[m] = {'prev': prev_solution_dict[m],ThreadStatus.NEW:new_solution_dict[m]}
        pairs_in_text = []
        for key, p in modified_pairs.items():
            pairs_in_text.append(f"""

topic_id: {key}
Previous version:
    statement: {p['prev']['Header']}
    solution: {p['prev']['Solution']}
    status: {p['prev']['Label']}
New version:
    statement: {p[ThreadStatus.NEW]['Header']}
    solution: {p[ThreadStatus.NEW]['Solution']}
    status: {p[ThreadStatus.NEW]['Label']}
"""
        )

        response_solutions = gemini_service.generate_content(
            contents=[
                prompts.system_prompt,
                prompts.revision_prompt.format(pairs='\n'.join(pairs_in_text))
                ],
            config=gemini_service.revision_config,
        )

        revised_solutions=response_solutions.comparisions

        for s in revised_solutions:
            if  s.Label==RevisedStatus.IMPROVED:
                add_or_update_solution(solutions_dict, new_solution_dict[s.topic_id])
                logging.info(f'Topic {s.topic_id} improved')
            elif s.Label==RevisedStatus.MINORCHANGES:
                logging.info(f"Topic {s.topic_id} get {len(new_solution_dict[s.topic_id]['whole_thread']) - len(solutions_dict[s.topic_id]['whole_thread'])} new messages")
                solutions_dict[s.topic_id]['whole_thread'] = new_solution_dict[s.topic_id]['whole_thread']
                solutions_dict[s.topic_id]['actual_date'] = new_solution_dict[s.topic_id]['actual_date']
                solutions_dict[s.topic_id]['answer_id'] = new_solution_dict[s.topic_id]['answer_id']
                solutions_dict[s.topic_id]['label'] = new_solution_dict[s.topic_id]['label']
            else:

                changed_solution = new_solution_dict[s.topic_id]
                rag_check_result = _check_solution_with_rag(changed_solution, exclude_solution_id=s.topic_id)
                
                if rag_check_result['recommendation'] == 'add':
                    new_threads.append(s.topic_id)
                    logging.info(f'Topic {s.topic_id} marked as changed but unique enough to add: {rag_check_result["reason"]}')
                elif rag_check_result['recommendation'] == 'merge':

                    _merge_similar_solutions(solutions_dict, s.topic_id, changed_solution, rag_check_result['similar_solutions'])
                    logging.info(f'Topic {s.topic_id} merged with similar existing solution')
                elif rag_check_result['recommendation'] == 'mark_duplicate':

                    add_or_update_solution(solutions_dict, changed_solution)
                    _create_duplicate_records(changed_solution, rag_check_result['similar_solutions'])
                    logging.info(f'Topic {s.topic_id} marked as duplicate: {rag_check_result["reason"]}')
                else:

                    logging.info(f'Topic {s.topic_id} skipped due to RAG check: {rag_check_result["reason"]}')
                    pass


    new_solutions_for_add = {key: s for key, s in new_solution_dict.items() if key in new_threads}
    
    
    return new_solutions_for_add

def check_in_rag_and_save(solutions_dict, new_solutions_for_add):
    """
    Check new solutions against RAG and add to solutions_dict if approved
    """
    if not new_solutions_for_add:
        logging.info("No new solutions to check in RAG")
        return
    
    logging.info(f"Checking {len(new_solutions_for_add)} new solutions in RAG...")

    filtered_solutions_for_add = []
    for topic_id, solution_data in new_solutions_for_add.items():
        rag_check_result = _check_solution_with_rag(solution_data)
        
        if rag_check_result['recommendation'] == 'add':
            filtered_solutions_for_add.append(solution_data)
            logging.info(f'New solution {topic_id} approved for addition: {rag_check_result["reason"]}')
        elif rag_check_result['recommendation'] == 'merge':

            _merge_similar_solutions(solutions_dict, topic_id, solution_data, rag_check_result['similar_solutions'])
            logging.info(f'New solution {topic_id} merged with existing solution')
        elif rag_check_result['recommendation'] == 'mark_duplicate':

            filtered_solutions_for_add.append(solution_data)
            _create_duplicate_records(solution_data, rag_check_result['similar_solutions'])
            logging.info(f'New solution {topic_id} marked as duplicate: {rag_check_result["reason"]}')
        else:

            logging.info(f'New solution {topic_id} skipped: {rag_check_result["reason"]}')
            if rag_check_result['similar_solutions']:
                similar_topic = rag_check_result['similar_solutions'][0][0].thread.topic_id
                logging.info(f'  Similar existing solution: {similar_topic}')
    
    add_new_solutions_to_dict(solutions_dict, filtered_solutions_for_add)
    logging.info(f"Added {len(filtered_solutions_for_add)} new solutions after RAG filtering")
    return solutions_dict

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

            exclude_db_solution_id = None
            if exclude_solution_id:
                thread = db_service.get_thread_by_topic_id(session, exclude_solution_id)
                if thread and thread.solutions:
                    exclude_db_solution_id = thread.solutions[0].id
            
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

        most_similar_solution, similarity_score = similar_solutions[0]
        existing_topic_id = most_similar_solution.thread.topic_id
        
        if existing_topic_id in solutions_dict:
            existing_solution = solutions_dict[existing_topic_id]
            
            if similarity_score >= 0.95:

                new_messages = set(new_solution_data.get('Whole_thread', []))
                existing_messages = set(existing_solution.get('Whole_thread', []))
                combined_messages = list(existing_messages.union(new_messages))
                existing_solution['Whole_thread'] = combined_messages
                
                logging.info(f"Merged {topic_id} into {existing_topic_id}: updated message list")
                
            elif similarity_score >= 0.85:

                new_solution_text = new_solution_data.get('Solution', '')
                existing_solution_text = existing_solution.get('Solution', '')
                
                if len(new_solution_text) > len(existing_solution_text):
                    existing_solution['Solution'] = new_solution_text
                    existing_solution['Header'] = new_solution_data.get('Header', existing_solution['Header'])
                    logging.info(f"Merged {topic_id} into {existing_topic_id}: updated solution text")
                
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

            topic_id = solution_data.get('topic_id') or solution_data.get('Topic_ID')
            thread = db_service.get_thread_by_topic_id(session, topic_id)
            
            if not thread:
                logging.error(f"Thread not found in database for topic_id: {topic_id}")
                return
            
            solution = session.query(Solution).filter(Solution.thread_id == thread.id).first()
            if not solution:

                solution_db_data = {
                    'thread_id': thread.id,
                    'header': solution_data.get('Header', ''),
                    'solution': solution_data.get('Solution', ''),
                    'label': solution_data.get('Label', SolutionStatus.UNRESOLVED),
                    'is_duplicate': True
                }
                solution = db_service.create_solution(session, solution_db_data)
                session.flush()
            else:

                solution.is_duplicate = True
                session.flush()
            
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

def update_database_with_solutions(solutions_dict: dict):
    """
    Update database with solutions from the solutions dictionary.
    This helps maintain consistency between file-based and database storage.
    """
    try:
        rag_service = get_rag_service()
        db_service = get_database_service()
        
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
                        'status': ThreadStatus.NEW,
                        'is_technical': True,
                        'is_processed': True
                    }
                    thread = db_service.create_thread(session, thread_data)
                    session.flush()
                
                # Update or create solution
                existing_solution = session.query(Solution).filter(
                    Solution.thread_id == thread.id
                ).first()
                
                if existing_solution:
                    # Update existing solution
                    db_service.update_solution(session, thread.id, {
                        'header': solution_data.get('Header', ''),
                        'solution': solution_data.get('Solution', ''),
                        'label': solution_data.get('Label', SolutionStatus.UNRESOLVED)
                    })
                else:
                    # Create new solution
                    solution_db_data = {
                        'thread_id': thread.id,
                        'header': solution_data.get('Header', ''),
                        'solution': solution_data.get('Solution', ''),
                        'label': solution_data.get('Label', SolutionStatus.UNRESOLVED)
                    }
                    solution = db_service.create_solution(session, solution_db_data)
                    
                    # Generate embedding for new solution
                    rag_service.add_solution_embedding(session, solution)
            
            session.commit()
            logging.info("Database synchronized with solutions dictionary")
            
    except Exception as e:
        logging.error(f"Failed to update database with solutions: {e}")