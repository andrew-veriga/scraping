import os
import yaml
import logging
import time
import random
from typing import List, Optional, Dict, Any, Generator
from datetime import datetime, timezone
import pandas as pd
from sqlalchemy import create_engine, text, and_, or_, desc, func
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError, OperationalError, DisconnectionError
from contextlib import contextmanager
from app.models.pydantic_models import ThreadStatus
import hashlib
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from app.models.db_models import (
    Base, Message, Thread, Solution, SolutionDuplicate,
    SolutionEmbedding, SolutionSimilarity, ProcessingBatch, LLMCache,
    MessageProcessing, MessageAnnotation, ProcessingPipeline
)

class DatabaseService:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.engine = None
        self.SessionLocal = None
        # Load retry configuration from config
        db_config = self.config.get('database', {})
        self.max_retries = db_config.get('max_retries', 3)
        self.base_delay = db_config.get('base_delay', 1.0)
        self.max_delay = db_config.get('max_delay', 60.0)
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
            db_url = os.environ.get('PEERA_DB_URL')
            
            db_config = self.config.get('database', {})
            
            if not db_url:
                # Fall back to config (with environment variable expansion)
                db_url = db_config.get('url')
                
                # Handle environment variable expansion in config
                if db_url and db_url.startswith('${') and db_url.endswith('}'):
                    env_var_name = db_url[2:-1]  # Remove ${ and }
                    db_url = os.environ.get(env_var_name)
            
            if not db_url:
                raise ValueError("Database URL not found in environment (PEERA_DB_URL) or configuration")
            
            # Create engine with connection pooling and SSL configuration
            pool_size = db_config.get('pool_size', 10)
            max_overflow = db_config.get('max_overflow', 20)
            pool_timeout = db_config.get('pool_timeout', 60)
            pool_recycle = db_config.get('pool_recycle', 1800)
            pool_pre_ping = db_config.get('pool_pre_ping', True)
            
            self.logger.info(f"Creating database engine with pool_size={pool_size}, max_overflow={max_overflow}")
            
            self.engine = create_engine(
                db_url,
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=pool_pre_ping,
                echo=False,  # Set to True for SQL debugging
                connect_args={
                    'sslmode': db_config.get('ssl_mode', 'prefer'),
                    'connect_timeout': db_config.get('connect_timeout', 30),
                    'application_name': 'discord-sui-analyzer'
                }
            )
            
            # Log pool configuration after creation
            pool = self.engine.pool
            self.logger.info(f"Pool created: type={type(pool).__name__}, size={getattr(pool, '_pool_size', 'unknown')}")
            
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
    
    def _is_connection_error(self, error: Exception) -> bool:
        """Check if the error is a connection-related error that should trigger a retry."""
        error_str = str(error).lower()
        connection_errors = [
            'ssl syscall error',
            'eof detected',
            'connection reset',
            'connection refused',
            'connection timeout',
            'server closed the connection',
            'connection lost',
            'database is not accepting connections',
            'could not connect to server',
            'connection terminated unexpectedly'
        ]
        return any(err in error_str for err in connection_errors)
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        jitter = random.uniform(0.1, 0.3) * delay
        return delay + jitter
    
    def _test_connection(self) -> bool:
        """Test database connection health."""
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
                return True
        except Exception as e:
            self.logger.warning(f"Connection health check failed: {e}")
            return False
    
    def _reconnect(self):
        """Recreate database connection."""
        try:
            self.logger.info("Attempting to reconnect to database...")
            self.engine.dispose()  # Close all existing connections
            self._initialize_database()
            self.logger.info("Database reconnection successful")
        except Exception as e:
            self.logger.error(f"Failed to reconnect to database: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get database session with automatic cleanup and retry logic."""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            session = None
            try:
                # Test connection health before creating session
                if attempt > 0:
                    if not self._test_connection():
                        self._reconnect()
                
                session = self.SessionLocal()
                yield session
                return  # Success, exit retry loop
                
            except (OperationalError, DisconnectionError) as e:
                last_error = e
                if session:
                    try:
                        session.rollback()
                        session.close()
                    except:
                        pass
                
                if self._is_connection_error(e) and attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(f"Database connection error (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    
                    # Try to reconnect on connection errors
                    try:
                        self._reconnect()
                    except Exception as reconnect_error:
                        self.logger.error(f"Reconnection failed: {reconnect_error}")
                        if attempt == self.max_retries:
                            raise reconnect_error
                else:
                    # Not a connection error or max retries reached
                    break
                    
            except Exception as e:
                last_error = e
                if session:
                    try:
                        session.rollback()
                        session.close()
                    except:
                        pass
                self.logger.error(f"Database session error: {e}")
                break
        
        # If we get here, all retries failed
        if last_error:
            self.logger.error(f"Database operation failed after {self.max_retries + 1} attempts")
            raise last_error
        else:
            raise Exception("Database operation failed for unknown reason")
    
    # Message operations
    def create_message(self, session: Session, message_data: Dict[str, Any]) -> Message:
        """Create a new message record."""
        message = Message(
            message_id=message_data['message_id'],
            parent_id=message_data.get('parent_id'),
            author_id=message_data['author_id'],
            content=message_data['content'],
            datetime=message_data['datetime'],
            dated_message=message_data['dated_message'],
            referenced_message_id=message_data.get('referenced_message_id', '')
        )
        session.add(message)
        return message
    
    def get_message_by_message_id(self, message_id: str, session: Optional[Session] = None) -> Optional[Message]:
        """Get message by Discord message ID."""
        if session is not None:
            # Use provided session (for use within existing transactions)
            return session.query(Message).filter(Message.message_id == message_id).first()
        else:
            # Create new session with retry logic
            with self.get_session() as new_session:
                return new_session.query(Message).filter(Message.message_id == message_id).first()
    
    def bulk_create_messages(self, messages_data: List[Dict[str, Any]]) -> int:
        """Bulk create messages from DataFrame or list."""
        created_count = 0
        with self.get_session() as session:
            for message_data in messages_data:
                try:
                    # Check if message already exists
                    existing = self.get_message_by_message_id(message_data['message_id'], session)
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
    
    def bulk_create_messages_hierarchical(self, messages_data: List[Dict[str, Any]]) -> int:
        """Bulk create messages with hierarchical structure (no FK constraints on parent_id)."""
        created_count = 0
        
        with self.get_session() as session:
            try:
                for message_data in messages_data:
                    try:
                        # Check if message already exists
                        existing = self.get_message_by_message_id(message_data['message_id'], session)
                        if not existing:
                            # Create message with hierarchical fields
                            message = Message(
                                message_id=message_data['message_id'],
                                parent_id=message_data.get('parent_id'),
                                author_id=message_data['author_id'],
                                content=message_data['content'],
                                datetime=message_data['datetime'],
                                dated_message=message_data['dated_message'],
                                referenced_message_id=message_data.get('referenced_message_id'),
                                thread_id=message_data.get('thread_id')
                            )
                            session.add(message)
                            created_count += 1
                    except IntegrityError as e:
                        session.rollback()
                        self.logger.warning(f"Could not create message {message_data['message_id']}: {e}")
                        continue
                
                session.commit()
                self.logger.info(f"Created {created_count} new hierarchical messages")
                
            except Exception as e:
                session.rollback()
                self.logger.error(f"Failed to bulk create hierarchical messages: {e}")
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
            status=thread_data.get('status', ThreadStatus.NEW),
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
    
    # Thread-Message relationship operations (now direct with hierarchical structure)
    def get_thread_messages(self, session: Session, thread_id: str) -> List[Message]:
        """Get all messages for a thread in order (using direct relationship)."""
        return session.query(Message).filter(
            Message.thread_id == thread_id
        ).order_by(Message.datetime).all()
    
    def get_thread_root_message(self, session: Session, thread_id: str) -> Optional[Message]:
        """Get the root message for a thread."""
        return session.query(Message).filter(
            Message.thread_id == thread_id,
            Message.parent_id == None
        ).first()
    
    # solution operations
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
            lookback_date=batch_data.get('lookback_date')
        )
        session.add(batch)
        return batch
    
    def complete_processing_batch(self, session: Session, batch_id: int, stats: Dict[str, Any]):
        """Mark processing batch as completed."""
        batch = session.query(ProcessingBatch).filter(ProcessingBatch.id == batch_id).first()
        if batch:
            batch.completed_at = func.now()
            batch.status = 'completed'
    
    def get_latest_processing_date(self, session: Session) -> Optional[datetime]:
        """Get the latest processing date from completed batches."""
        batch = session.query(ProcessingBatch).filter(
            ProcessingBatch.status == 'completed'
        ).order_by(desc(ProcessingBatch.end_date)).first()
        return batch.end_date if batch else None
    
    def get_latest_solution_date(self, session: Session) -> Optional[datetime]:
        """Get the latest datetime from threads, normalized and incremented by one day."""
        thread = session.query(Thread).filter(
            Thread.actual_date.is_not(None)
        ).order_by(desc(Thread.actual_date)).first()
        if thread:
            # Convert to pandas timestamp, normalize, and add one day
            
            latest_date = pd.Timestamp(thread.actual_date).normalize() + pd.Timedelta(days=1)
            return latest_date.to_pydatetime()
        return None
    
    def get_latest_threads_from_actual_date(self, session: Session, lookback_date: datetime) -> List[Thread]:
        """Get all threads with actual_date >= lookback_date."""
        return session.query(Thread).filter(
            Thread.actual_date >= lookback_date
        ).order_by(Thread.actual_date).all()
    
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
    
    # Duplicate management methods
    def create_duplicate_record(self, session: Session, duplicate_data: Dict[str, Any]) -> Optional[SolutionDuplicate]:
        """Create a new duplicate record."""
        try:
            duplicate_record = SolutionDuplicate(**duplicate_data)
            session.add(duplicate_record)
            session.flush()
            return duplicate_record
        except IntegrityError as e:
            session.rollback()
            self.logger.error(f"Failed to create duplicate record: {e}")
            return None
    
    def get_solution_duplicates(self, session: Session, solution_id: int) -> List[SolutionDuplicate]:
        """Get all duplicates of a solution."""
        return session.query(SolutionDuplicate).filter(
            SolutionDuplicate.original_solution_id == solution_id
        ).order_by(SolutionDuplicate.created_at).all()
    
    def get_duplicate_by_id(self, session: Session, duplicate_id: int) -> Optional[SolutionDuplicate]:
        """Get a specific duplicate record by ID."""
        return session.query(SolutionDuplicate).filter(SolutionDuplicate.id == duplicate_id).first()
    
    def update_duplicate_status(self, session: Session, duplicate_id: int, 
                               status: str, reviewed_by: str = None, notes: str = None) -> bool:
        """Update the status of a duplicate record."""
        try:
            duplicate_record = session.query(SolutionDuplicate).filter(
                SolutionDuplicate.id == duplicate_id
            ).first()
            
            if not duplicate_record:
                return False
            
            duplicate_record.status = status
            if reviewed_by:
                duplicate_record.reviewed_by = reviewed_by
                duplicate_record.reviewed_at = func.now()
            if notes:
                duplicate_record.notes = notes
            
            session.flush()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update duplicate status: {e}")
            return False
    
    def get_pending_duplicates(self, session: Session, limit: int = 50, offset: int = 0) -> List[SolutionDuplicate]:
        """Get duplicates pending review."""
        return session.query(SolutionDuplicate).filter(
            SolutionDuplicate.status == 'pending_review'
        ).order_by(
            SolutionDuplicate.similarity_score.desc(), 
            SolutionDuplicate.created_at.desc()
        ).offset(offset).limit(limit).all()
    
    def get_duplicate_statistics(self, session: Session) -> Dict[str, Any]:
        """Get comprehensive duplicate statistics."""
        try:
            # Basic counts
            total_duplicates = session.query(SolutionDuplicate).count()
            pending_duplicates = session.query(SolutionDuplicate).filter(
                SolutionDuplicate.status == 'pending_review'
            ).count()
            confirmed_duplicates = session.query(SolutionDuplicate).filter(
                SolutionDuplicate.status == 'confirmed_duplicate'
            ).count()
            false_positives = session.query(SolutionDuplicate).filter(
                SolutionDuplicate.status == 'false_positive'
            ).count()
            
            # Solutions with duplicates
            solutions_with_duplicates = session.query(Solution).filter(
                Solution.duplicate_count > 0
            ).count()
            
            # Most duplicated solution
            most_duplicated = session.query(Solution).filter(
                Solution.duplicate_count > 0
            ).order_by(Solution.duplicate_count.desc()).first()
            
            return {
                'total_duplicates': total_duplicates,
                'pending_review': pending_duplicates,
                'confirmed_duplicates': confirmed_duplicates,
                'false_positives': false_positives,
                'solutions_with_duplicates': solutions_with_duplicates,
                'most_duplicated_count': most_duplicated.duplicate_count if most_duplicated else 0,
                'most_duplicated_header': most_duplicated.header[:100] if most_duplicated else None
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get duplicate statistics: {e}")
            return {}
    
    def bulk_update_duplicate_counts(self, session: Session) -> int:
        """Recalculate and update duplicate counts for all solutions."""
        try:
            # Get all solutions that have duplicates
            duplicate_counts = session.query(
                SolutionDuplicate.original_solution_id,
                func.count(SolutionDuplicate.id).label('count')
            ).group_by(SolutionDuplicate.original_solution_id).all()
            
            updated_count = 0
            for original_id, count in duplicate_counts:
                solution = session.query(Solution).filter(Solution.id == original_id).first()
                if solution and solution.duplicate_count != count:
                    solution.duplicate_count = count
                    updated_count += 1
            
            # Reset count to 0 for solutions with no duplicates
            solutions_with_zero_duplicates = session.query(Solution).filter(
                and_(
                    Solution.duplicate_count > 0,
                    ~Solution.id.in_([dc.original_solution_id for dc in duplicate_counts])
                )
            ).all()
            
            for solution in solutions_with_zero_duplicates:
                solution.duplicate_count = 0
                updated_count += 1
            
            session.flush()
            return updated_count
            
        except Exception as e:
            self.logger.error(f"Failed to bulk update duplicate counts: {e}")
            return 0
    
    def get_thread_by_topic_id(self, session: Session, topic_id: str) -> Optional[Thread]:
        """Get thread by its topic_id."""
        return session.query(Thread).filter(Thread.topic_id == topic_id).first()
    
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
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive database health check."""
        health_status = {
            'status': 'healthy',
            'connection_test': False,
            'pool_status': {},
            'error': None
        }
        
        try:
            # Test basic connection
            health_status['connection_test'] = self._test_connection()
            
            # Get connection pool status
            pool = self.engine.pool
            pool_status = {}
            
            # Get available pool metrics safely
            try:
                pool_status['size'] = pool.size()
            except:
                pool_status['size'] = 'unknown'
                
            try:
                pool_status['checked_in'] = pool.checkedin()
            except:
                pool_status['checked_in'] = 'unknown'
                
            try:
                pool_status['checked_out'] = pool.checkedout()
            except:
                pool_status['checked_out'] = 'unknown'
                
            try:
                pool_status['overflow'] = pool.overflow()
            except:
                pool_status['overflow'] = 'unknown'
            
            # Try to get additional pool info if available
            try:
                if hasattr(pool, 'invalidated'):
                    pool_status['invalidated'] = pool.invalidated
                elif hasattr(pool, 'invalid'):
                    pool_status['invalidated'] = pool.invalid()
                else:
                    pool_status['invalidated'] = 'not_available'
            except:
                pool_status['invalidated'] = 'error'
            
            health_status['pool_status'] = pool_status
            
            if not health_status['connection_test']:
                health_status['status'] = 'unhealthy'
                health_status['error'] = 'Connection test failed'
                
        except Exception as e:
            health_status['status'] = 'unhealthy'
            health_status['error'] = str(e)
            self.logger.error(f"Health check failed: {e}")
        
        return health_status
    
    def cleanup_connections(self):
        """Force cleanup of all connections in the pool."""
        try:
            self.logger.info("Cleaning up database connections...")
            self.engine.dispose()
            self.logger.info("Database connections cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Failed to cleanup connections: {e}")
    
    def warmup_pool(self):
        """Warm up the connection pool by creating the configured number of connections."""
        try:
            self.logger.info("Warming up connection pool...")
            pool = self.engine.pool
            pool_size = getattr(pool, '_pool_size', 5)
            
            # Create connections to fill the pool
            connections = []
            for i in range(pool_size):
                try:
                    conn = pool.connect()
                    connections.append(conn)
                    self.logger.debug(f"Created connection {i+1}/{pool_size}")
                except Exception as e:
                    self.logger.warning(f"Failed to create connection {i+1}: {e}")
                    break
            
            # Return connections to the pool
            for conn in connections:
                try:
                    conn.close()
                except:
                    pass
            
            self.logger.info(f"Pool warmup completed. Created {len(connections)} connections.")
            return len(connections)
            
        except Exception as e:
            self.logger.error(f"Failed to warmup pool: {e}")
            return 0
    
    def get_pool_status(self) -> Dict[str, Any]:
        """Get detailed connection pool status."""
        try:
            pool = self.engine.pool
            
            # Get pool configuration
            pool_size = getattr(pool, '_pool_size', pool.size())
            max_overflow = getattr(pool, '_max_overflow', 0)
            
            # Get current pool state
            checked_in = pool.checkedin()
            checked_out = pool.checkedout()
            overflow = pool.overflow()
            
            # Calculate total connections (base pool + overflow)
            total_connections = pool_size + max(0, overflow)
            max_total_connections = pool_size + max_overflow
            
            # Calculate utilization based on max possible connections
            utilization = round((checked_out / max_total_connections) * 100, 2) if max_total_connections > 0 else 0
            
            status = {
                'pool_size': pool_size,
                'max_overflow': max_overflow,
                'checked_in': checked_in,
                'checked_out': checked_out,
                'overflow': overflow,
                'total_connections': total_connections,
                'max_total_connections': max_total_connections,
                'available_connections': checked_in,
                'utilization_percent': utilization,
                'status': 'healthy' if utilization < 80 else 'warning' if utilization < 95 else 'critical'
            }
            
            # Log warning if utilization is high
            if utilization > 80:
                self.logger.warning(f"High connection pool utilization: {utilization}% ({checked_out}/{max_total_connections})")
            
            return status
        except Exception as e:
            self.logger.error(f"Failed to get pool status: {e}")
            return {'error': str(e)}


# Global database service instance
db_service = None

def get_database_service() -> DatabaseService:
    """Get global database service instance."""
    global db_service
    if db_service is None:
        db_service = DatabaseService()
    return db_service