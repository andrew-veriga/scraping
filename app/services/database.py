import os
import yaml
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text, and_, or_, desc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from contextlib import contextmanager
import hashlib
import json

from app.models.db_models import (
    Base, Message, Thread, ThreadMessage, Solution, 
    SolutionEmbedding, SolutionSimilarity, ProcessingBatch, LLMCache
)

class DatabaseService:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.engine = None
        self.SessionLocal = None
        self._initialize_database()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load database configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            # Try to get database URL from environment first, then fall back to config
            db_url = os.environ.get('DATABASE_URL')
            
            if not db_url:
                # Fall back to config (with environment variable expansion)
                db_config = self.config.get('database', {})
                db_url = db_config.get('url')
                
                # Handle environment variable expansion in config
                if db_url and db_url.startswith('${') and db_url.endswith('}'):
                    env_var_name = db_url[2:-1]  # Remove ${ and }
                    db_url = os.environ.get(env_var_name)
            
            if not db_url:
                raise ValueError("Database URL not found in environment (DATABASE_URL) or configuration")
            
            # Create engine with connection pooling
            self.engine = create_engine(
                db_url,
                pool_size=db_config.get('pool_size', 5),
                max_overflow=db_config.get('max_overflow', 10),
                pool_timeout=db_config.get('pool_timeout', 30),
                pool_recycle=db_config.get('pool_recycle', 3600),
                echo=False  # Set to True for SQL debugging
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(bind=self.engine)
            
            # Try to enable pgvector extension first (before creating tables)
            self.pgvector_available = self._enable_pgvector()
            
            # Create tables (will use appropriate column types based on pgvector availability)
            self._create_tables()
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create all database tables."""
        try:
            Base.metadata.create_all(bind=self.engine)
            self.logger.info("Database tables created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create tables: {e}")
            raise
    
    def _enable_pgvector(self) -> bool:
        """Enable pgvector extension in PostgreSQL. Returns True if successful."""
        try:
            # Try to create extension using raw connection for better permissions
            with self.engine.connect() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                connection.commit()
                self.logger.info("pgvector extension enabled successfully")
                return True
        except Exception as e:
            self.logger.warning(f"Could not enable pgvector extension: {e}")
            self.logger.info("Falling back to JSON storage for embeddings")
            return False
    
    @contextmanager
    def get_session(self) -> Session:
        """Get database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    # Message operations
    def create_message(self, session: Session, message_data: Dict[str, Any]) -> Message:
        """Create a new message record."""
        message = Message(
            message_id=message_data['message_id'],
            author_id=message_data['author_id'],
            content=message_data['content'],
            datetime=message_data['datetime'],
            dated_message=message_data['dated_message'],
            referenced_message_id=message_data.get('referenced_message_id', '')
        )
        session.add(message)
        return message
    
    def get_message_by_message_id(self, session: Session, message_id: str) -> Optional[Message]:
        """Get message by Discord message ID."""
        return session.query(Message).filter(Message.message_id == message_id).first()
    
    def bulk_create_messages(self, messages_data: List[Dict[str, Any]]) -> int:
        """Bulk create messages from DataFrame or list."""
        created_count = 0
        with self.get_session() as session:
            for message_data in messages_data:
                try:
                    # Check if message already exists
                    existing = self.get_message_by_message_id(session, message_data['message_id'])
                    if not existing:
                        self.create_message(session, message_data)
                        created_count += 1
                except IntegrityError:
                    session.rollback()
                    continue
            
            try:
                session.commit()
                self.logger.info(f"Created {created_count} new messages")
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to bulk create messages: {e}")
                raise
        
        return created_count
    
    # Thread operations
    def create_thread(self, session: Session, thread_data: Dict[str, Any]) -> Thread:
        """Create a new thread record."""
        thread = Thread(
            topic_id=thread_data['topic_id'],
            header=thread_data.get('header'),
            actual_date=thread_data['actual_date'],
            answer_id=thread_data.get('answer_id'),
            label=thread_data.get('label'),
            solution=thread_data.get('solution'),
            status=thread_data.get('status', 'new'),
            is_technical=thread_data.get('is_technical', False),
            is_processed=thread_data.get('is_processed', False)
        )
        session.add(thread)
        return thread
    
    def get_thread_by_topic_id(self, session: Session, topic_id: str) -> Optional[Thread]:
        """Get thread by topic ID."""
        return session.query(Thread).filter(Thread.topic_id == topic_id).first()
    
    def update_thread(self, session: Session, topic_id: str, updates: Dict[str, Any]) -> Optional[Thread]:
        """Update thread with new data."""
        thread = self.get_thread_by_topic_id(session, topic_id)
        if thread:
            for key, value in updates.items():
                if hasattr(thread, key):
                    setattr(thread, key, value)
            thread.updated_at = func.now()
        return thread
    
    def get_unprocessed_threads(self, session: Session, limit: Optional[int] = None) -> List[Thread]:
        """Get threads that haven't been processed yet."""
        query = session.query(Thread).filter(Thread.is_processed == False)
        if limit:
            query = query.limit(limit)
        return query.all()
    
    def get_technical_threads(self, session: Session, from_date: Optional[datetime] = None) -> List[Thread]:
        """Get threads marked as technical."""
        query = session.query(Thread).filter(Thread.is_technical == True)
        if from_date:
            query = query.filter(Thread.actual_date >= from_date)
        return query.order_by(Thread.actual_date).all()
    
    # Thread-Message relationship operations
    def add_messages_to_thread(self, session: Session, thread_id: int, message_ids: List[str]):
        """Add messages to a thread."""
        for order, message_id in enumerate(message_ids):
            message = self.get_message_by_message_id(session, message_id)
            if message:
                try:
                    thread_message = ThreadMessage(
                        thread_id=thread_id,
                        message_id=message.id,
                        order_in_thread=order
                    )
                    session.add(thread_message)
                except IntegrityError:
                    # Message already in thread
                    continue
    
    def get_thread_messages(self, session: Session, thread_id: int) -> List[Message]:
        """Get all messages for a thread in order."""
        return session.query(Message).join(ThreadMessage).filter(
            ThreadMessage.thread_id == thread_id
        ).order_by(ThreadMessage.order_in_thread).all()
    
    # Solution operations
    def create_solution(self, session: Session, solution_data: Dict[str, Any]) -> Solution:
        """Create a new solution record."""
        solution = Solution(
            thread_id=solution_data['thread_id'],
            header=solution_data['header'],
            solution=solution_data['solution'],
            label=solution_data['label'],
            confidence_score=solution_data.get('confidence_score')
        )
        session.add(solution)
        return solution
    
    def update_solution(self, session: Session, thread_id: int, updates: Dict[str, Any]) -> Optional[Solution]:
        """Update solution for a thread."""
        solution = session.query(Solution).filter(Solution.thread_id == thread_id).first()
        if solution:
            for key, value in updates.items():
                if hasattr(solution, key):
                    setattr(solution, key, value)
            solution.updated_at = func.now()
            solution.version += 1
        return solution
    
    def get_solutions_for_similarity_check(self, session: Session, label: Optional[str] = None) -> List[Solution]:
        """Get solutions for similarity comparison."""
        query = session.query(Solution)
        if label:
            query = query.filter(Solution.label == label)
        return query.all()
    
    # Processing batch operations
    def create_processing_batch(self, session: Session, batch_data: Dict[str, Any]) -> ProcessingBatch:
        """Create a new processing batch record."""
        batch = ProcessingBatch(
            batch_type=batch_data['batch_type'],
            start_date=batch_data['start_date'],
            end_date=batch_data['end_date'],
            lookback_date=batch_data.get('lookback_date'),
            messages_processed=batch_data.get('messages_processed', 0),
            threads_created=batch_data.get('threads_created', 0),
            threads_modified=batch_data.get('threads_modified', 0),
            technical_threads=batch_data.get('technical_threads', 0),
            solutions_added=batch_data.get('solutions_added', 0)
        )
        session.add(batch)
        return batch
    
    def complete_processing_batch(self, session: Session, batch_id: int, stats: Dict[str, Any]):
        """Mark processing batch as completed."""
        batch = session.query(ProcessingBatch).filter(ProcessingBatch.id == batch_id).first()
        if batch:
            batch.completed_at = func.now()
            batch.status = 'completed'
            for key, value in stats.items():
                if hasattr(batch, key):
                    setattr(batch, key, value)
    
    def get_latest_processing_date(self, session: Session) -> Optional[datetime]:
        """Get the latest processing date from completed batches."""
        batch = session.query(ProcessingBatch).filter(
            ProcessingBatch.status == 'completed'
        ).order_by(desc(ProcessingBatch.end_date)).first()
        return batch.end_date if batch else None
    
    # LLM Cache operations
    def _generate_cache_key(self, request_data: Dict[str, Any]) -> str:
        """Generate cache key from request data."""
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def get_cached_llm_response(self, session: Session, request_data: Dict[str, Any], 
                               request_type: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM response if exists and not expired."""
        cache_key = self._generate_cache_key(request_data)
        cache_entry = session.query(LLMCache).filter(
            and_(
                LLMCache.request_hash == cache_key,
                LLMCache.request_type == request_type,
                or_(LLMCache.ttl_expires_at.is_(None), 
                    LLMCache.ttl_expires_at > func.now())
            )
        ).first()
        
        if cache_entry:
            # Update access stats
            cache_entry.last_accessed = func.now()
            cache_entry.access_count += 1
            return cache_entry.response_data
        
        return None
    
    def cache_llm_response(self, session: Session, request_data: Dict[str, Any], 
                          request_type: str, response_data: Dict[str, Any], 
                          ttl_hours: Optional[int] = None):
        """Cache LLM response."""
        cache_key = self._generate_cache_key(request_data)
        
        # Calculate expiration if TTL is provided
        expires_at = None
        if ttl_hours:
            expires_at = datetime.now(timezone.utc) + pd.Timedelta(hours=ttl_hours)
        
        cache_entry = LLMCache(
            request_hash=cache_key,
            request_type=request_type,
            response_data=response_data,
            ttl_expires_at=expires_at
        )
        
        try:
            session.add(cache_entry)
            session.commit()
        except IntegrityError:
            # Entry already exists, update it
            session.rollback()
            existing = session.query(LLMCache).filter(
                LLMCache.request_hash == cache_key
            ).first()
            if existing:
                existing.response_data = response_data
                existing.last_accessed = func.now()
                existing.access_count += 1
                existing.ttl_expires_at = expires_at
    
    # Utility methods
    def get_database_stats(self) -> Dict[str, int]:
        """Get basic database statistics."""
        with self.get_session() as session:
            stats = {
                'messages': session.query(Message).count(),
                'threads': session.query(Thread).count(),
                'solutions': session.query(Solution).count(),
                'technical_threads': session.query(Thread).filter(Thread.is_technical == True).count(),
                'processed_threads': session.query(Thread).filter(Thread.is_processed == True).count(),
                'cached_responses': session.query(LLMCache).count()
            }
            return stats
    
    def cleanup_expired_cache(self, session: Session) -> int:
        """Remove expired cache entries."""
        expired_count = session.query(LLMCache).filter(
            and_(
                LLMCache.ttl_expires_at.is_not(None),
                LLMCache.ttl_expires_at < func.now()
            )
        ).delete()
        return expired_count


# Global database service instance
db_service = None

def get_database_service() -> DatabaseService:
    """Get global database service instance."""
    global db_service
    if db_service is None:
        db_service = DatabaseService()
    return db_service