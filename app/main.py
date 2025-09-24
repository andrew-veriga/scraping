from dotenv import load_dotenv

load_dotenv() 

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from app.processing import process_first_batch
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

def build_message_hierarchy(thread_data):
    """Build hierarchical structure from whole_thread messages."""
    whole_thread = thread_data.get('whole_thread', [])
    logging.info(f"build_message_hierarchy called with {len(whole_thread) if whole_thread else 0} messages")
    
    if not whole_thread or len(whole_thread) == 0:
        logging.info("No messages in whole_thread, returning empty hierarchy")
        return []
    
    # Create a map of message_id to message data
    message_map = {msg['message_id']: msg for msg in whole_thread}
    logging.info(f"Created message_map with {len(message_map)} messages")
    
    import collections
        
    try:
        # Find root messages (those with parent_id = null)
        root_message = [msg for msg in whole_thread if msg.get('parent_id') == 'None' or msg.get('parent_id') is None]
        if not root_message or len(root_message) == 0: # If no explicit root, take the first message as root??? TODO: it's not correct, whole_thread[0] may not be the root
            root_message = {'message_id': whole_thread[0]['message_id'], 'parent_id': None}
        else:
            root_message = root_message[0]

        queues = [collections.deque([root_message])]    
        for msg in whole_thread:
            if msg.get('parent_id') is not None:
                if msg['parent_id'] not in [q[-1]['message_id'] for q in queues]:
                    q = collections.deque()
                    queues.append(q)
                    q.append(msg)
                else: # find the queue with this parent_id
                    for q in queues:
                        if q[-1]['message_id'] == msg['parent_id']:
                            q.append(msg)
                            break
        for q in queues:
            logging.info(f"Queue with root {q[0]['message_id']} has {len(q)} messages")


            # parents = [msg['parent_id'] for msg in whole_thread ]
            # parents = list(set(parents)) # Unique parent IDs

            # nodes = {parent_id: [leaf for leaf in whole_thread if leaf['parent_id'] == parent_id] for parent_id in parents}
            # nodes['root'] = root_message
            # for parent_id, children in nodes.items():
            #     if len(children) == 1: #then expand it and reassign its parent to up level
            #         children[0]['expanded'] = True  # Expand single-child nodes by default
            #         single_child = children[0]
            #         # reassign parent_id to up level
            #         message_map[single_child['message_id']]['parent_id'] = parent_id 

            # for k, v in nodes.items():
            #     logging.info(f"Node {k} has {len(v)} children")
    except Exception as e:
        logging.error(f"Error finding root message: {e}", exc_info=True)
        return []
    
    def build_children(parent_id):
        children = []
        for msg in whole_thread:
            if msg.get('parent_id') == parent_id:
                child_node = {
                    'id': msg['message_id'],
                    'type': 'message',
                    'data': {
                        'message_id': msg['message_id'],
                        'parent_id': msg.get('parent_id'),
                        'content': msg.get('content', f"Message {msg['message_id']}"),
                    },
                    'children': build_children(msg['message_id']),
                    'expanded': False
                }
                children.append(child_node)
        return children
    
    # Build hierarchy starting from root messages
    hierarchy = []
    root_node = {
        'id': root_message['message_id'],
        'type': 'message',
        'data': {
            'message_id': root_message['message_id'],
            'parent_id': root_message.get('parent_id'),
            'content': root_message.get('content', f"Message {root_message['message_id']}")
        },
        'children': build_children(root_message['message_id']),
        'expanded': False
    }
    hierarchy.append(root_node)
    
    return hierarchy

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
        
        # Add warning if pool is in critical state
        if pool_status.get('status') == 'critical':
            logging.error(f"CRITICAL: Connection pool status: {pool_status}")
        
        return {
            "pool_status": pool_status,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Failed to get pool status: {e}")
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

# API endpoints for visual interface
@app.get("/threads")
def get_threads(limit: int = 10, offset: int = 0, status: str = None, technical: bool = None, source: str = "file"):
    """Get threads with optional filtering. source can be 'file' or 'database'."""
    try:
        if source == "file":
            # Load from solutions_dict.json
            solutions_dict = load_solutions_dict(SOLUTIONS_DICT_FILENAME, SAVE_PATH)
            thread_data = []
            
            for topic_id, thread_info in solutions_dict.items():
                # Build hierarchical structure from whole_thread
                whole_thread = thread_info.get('whole_thread', [])
                logging.info(f"Processing thread {topic_id}, whole_thread length: {len(whole_thread)}")
                messages_hierarchy = build_message_hierarchy(thread_info)
                logging.info(f"Built hierarchy for thread {topic_id}, hierarchy length: {len(messages_hierarchy)}")
                
                thread_dict = {
                    'topic_id': topic_id,
                    'header': thread_info.get('header', ''),
                    'actual_date': thread_info.get('actual_date', ''),
                    'answer_id': thread_info.get('answer_id', ''),
                    'label': thread_info.get('label', ''),
                    'solution': thread_info.get('solution', ''),
                    'whole_thread': whole_thread,
                    'messages_hierarchy': messages_hierarchy,  # New hierarchical structure
                    'status': thread_info.get('label', ''),
                    'is_technical': False,  # Default value
                    'is_processed': True,   # Default value
                    'created_at': thread_info.get('actual_date', ''),
                    'updated_at': thread_info.get('actual_date', ''),
                }
                thread_data.append(thread_dict)
            
            # Apply filters
            if status:
                thread_data = [t for t in thread_data if t['label'] == status]
            
            # Apply pagination
            thread_data = thread_data[offset:offset + limit]
            
            return thread_data
        else:
            # Load from database (original logic)
            db_service = get_database_service()
            with db_service.get_session() as session:
                query = session.query(Thread)
                
                if status:
                    query = query.filter(Thread.status == status)
                if technical is not None:
                    query = query.filter(Thread.is_technical == technical)
                
                threads = query.offset(offset).limit(limit).all()
                
                # Convert to dict format
                thread_data = []
                for thread in threads:
                    thread_dict = {
                        'topic_id': thread.topic_id,
                        'header': thread.header,
                        'actual_date': thread.actual_date.isoformat() if thread.actual_date else None,
                        'answer_id': thread.answer_id,
                        'label': thread.label,
                        'solution': thread.solution,
                        'status': thread.status,
                        'is_technical': thread.is_technical,
                        'is_processed': thread.is_processed,
                        'created_at': thread.created_at.isoformat() if thread.created_at else None,
                        'updated_at': thread.updated_at.isoformat() if thread.updated_at else None,
                    }
                    thread_data.append(thread_dict)
                
                return thread_data
    except Exception as e:
        logging.error(f"Error fetching threads: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    """Get a specific thread by ID."""
    try:
        db_service = get_database_service()
        with db_service.get_session() as session:
            thread = db_service.get_thread_by_topic_id(session,thread_id)
            if not thread:
                raise HTTPException(status_code=404, detail="Thread not found")
            
            # Get messages for this thread
            messages = db_service.get_thread_messages(session, thread_id)
            
            thread_dict = {
                'topic_id': thread.topic_id,
                'header': thread.header,
                'actual_date': thread.actual_date.isoformat() if thread.actual_date else None,
                'answer_id': thread.answer_id,
                'label': thread.label,
                'solution': thread.solution,
                'status': thread.status,
                'is_technical': thread.is_technical,
                'is_processed': thread.is_processed,
                'created_at': thread.created_at.isoformat() if thread.created_at else None,
                'updated_at': thread.updated_at.isoformat() if thread.updated_at else None,
                'messages': [
                    {
                        'message_id': msg.message_id,
                        'parent_id': msg.parent_id,
                        'author_id': msg.author_id,
                        'content': msg.content,
                        'datetime': msg.datetime.isoformat() if msg.datetime else None,
                        'thread_id': msg.thread_id,
                        'referenced_message_id': msg.referenced_message_id,
                    }
                    for msg in messages
                ]
            }
            
            return thread_dict
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching thread {thread_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/threads/{thread_id}")
def update_thread(thread_id: str, updates: dict):
    """Update a thread."""
    try:
        db_service = get_database_service()
        with db_service.get_session() as session:
            thread = db_service.update_thread(session, thread_id, updates)
            if not thread:
                raise HTTPException(status_code=404, detail="Thread not found")
            
            return {
                'topic_id': thread.topic_id,
                'header': thread.header,
                'actual_date': thread.actual_date.isoformat() if thread.actual_date else None,
                'answer_id': thread.answer_id,
                'label': thread.label,
                'solution': thread.solution,
                'status': thread.status,
                'is_technical': thread.is_technical,
                'is_processed': thread.is_processed,
                'updated_at': thread.updated_at.isoformat() if thread.updated_at else None,
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/threads/{thread_id}")
def delete_thread(thread_id: str):
    """Delete a thread."""
    try:
        db_service = get_database_service()
        with db_service.get_session() as session:
            thread = db_service.get_thread_by_topic_id(session, thread_id)
            if not thread:
                raise HTTPException(status_code=404, detail="Thread not found")
            
            session.delete(thread)
            session.commit()
            
            return {"message": "Thread deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting thread {thread_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/messages/{message_id}")
def get_message(message_id: str):
    """Get a specific message by ID."""
    try:
        db_service = get_database_service()
        with db_service.get_session() as session:
            message = db_service.get_message_by_message_id(message_id, session)
            if not message:
                raise HTTPException(status_code=404, detail="Message not found")
            
            return {
                'message_id': message.message_id,
                'parent_id': message.parent_id,
                'author_id': message.author_id,
                'content': message.content,
                'datetime': message.datetime.isoformat() if message.datetime else None,
                'thread_id': message.thread_id,
                'referenced_message_id': message.referenced_message_id,
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching message {message_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/threads/hierarchy")
def perform_hierarchy_operation(operation: dict):
    """Perform hierarchy operations like move, merge, split."""
    try:
        operation_type = operation.get('operation')
        source_id = operation.get('source_id')
        target_id = operation.get('target_id')
        data = operation.get('data', {})

        if operation == 'update_message_parent':
            message_id = data.get('messageId') or source_id
            new_parent_id = data.get('newParentId') or target_id
            
            # Update the message's parent_id in your database
            # This is the core functionality you need to implement
            update_message_parent_id(message_id, new_parent_id)
            
            return {
                "success": True,
                "message": f"Updated parent_id for message {message_id} to {new_parent_id}",
                "data": {
                    "message_id": message_id,
                    "new_parent_id": new_parent_id
                }
            }

        db_service = get_database_service()
        
        # This is a placeholder for hierarchy operations
        # You would implement the actual logic based on the operation type
        with db_service.get_session() as session:
            if operation_type == 'move_conversation':
                # Implement conversation move logic
                pass
            elif operation_type == 'move_message':
                # Implement message move logic
                pass
            elif operation_type == 'merge_threads':
                # Implement thread merge logic
                pass
            elif operation_type == 'split_conversation':
                # Implement conversation split logic
                pass
            else:
                raise HTTPException(status_code=400, detail=f"Unknown operation: {operation_type}")
            
            return {"message": f"Operation {operation_type} completed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error performing hierarchy operation: {e}")
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
            db_messages = db_service.get_thread_messages(session, thread.get('topic_id'))
 
            messages = []
            if db_messages:
                for db_message in db_messages:
                    message_id = db_message.message_id

                    message_content = db_message.illustrated_message or db_message.content or "N/A"
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
    