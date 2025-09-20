from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
from .pydantic_models import ThreadStatus #, SolutionStatus
from datetime import datetime
from typing import Optional, List
import os

# Try to import pgvector, but fallback gracefully if not available
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
    print("✅ pgvector SQLAlchemy integration loaded")
except ImportError:
    PGVECTOR_AVAILABLE = False
    print("⚠️  pgvector not available - using JSON fallback for embeddings")
    # Create a mock Vector type that falls back to JSON
    def Vector(dimension):
        from sqlalchemy.dialects.postgresql import JSON
        return JSON

Base = declarative_base()
class Author(Base):
    """Discord author model"""
    __tablename__ = 'authors'
    author_id = Column(String(50), primary_key=True)
    author_name = Column(String(50), nullable=False)
    author_type = Column(String(50), nullable=False)
    
    # One-to-many relationship: one author has many messages
    messages = relationship("Message", back_populates="author", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Author(author_id='{self.author_id}', name='{self.author_name}', type='{self.author_type}')>"
    
class Message(Base):
    """Discord message model with hierarchical parent-child relationships"""
    __tablename__ = 'messages'

    message_id = Column(String(50), primary_key=True)
    
    # Hierarchical parent-child relationship within a thread (no FK constraint for flexibility)
    parent_id = Column(String(50), nullable=True, index=True)
    
    author_id = Column(String(50), ForeignKey('authors.author_id'), nullable=False, index=True)
    content = Column(Text, nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False, index=True)
    referenced_message_id = Column(String(50), nullable=True, index=True)
    
    # Thread relationship - many messages belong to one thread
    thread_id = Column(String(50), ForeignKey('threads.topic_id'), nullable=True, index=True)
    
    # Self-referential relationships for message hierarchy (manual join via parent_id)
    parent_message = relationship("Message", remote_side=[message_id], 
                                 primaryjoin="foreign(Message.parent_id)==Message.message_id",
                                 back_populates="child_messages")
    child_messages = relationship("Message", 
                                 primaryjoin="Message.message_id==foreign(Message.parent_id)",
                                 back_populates="parent_message")
    
    # Author relationship - many messages belong to one author
    author = relationship("Author", back_populates="messages")
    
    # Thread relationship
    thread = relationship("Thread", back_populates="messages")
    
    # Processing relationships
    processing_steps = relationship("MessageProcessing", back_populates="message", cascade="all, delete-orphan")
    annotations = relationship("MessageAnnotation", back_populates="message", cascade="all, delete-orphan")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_message_datetime_author', 'datetime', 'author_id'),
        Index('idx_message_referenced', 'referenced_message_id'),
        Index('idx_message_thread_id', 'thread_id'),
        Index('idx_message_parent_id', 'parent_id'),
    )

    def __repr__(self):
        return f"<Message(message_id='{self.message_id}', author_id='{self.author_id}', thread_id='{self.thread_id}', parent_id='{self.parent_id}')>"
    
    @property
    def is_thread_root(self):
        """Check if this message is the root message of its thread (topic starter)"""
        return self.parent_id is None
    
    def get_all_descendants(self, session):
        """Get all descendant messages (children, grandchildren, etc.) recursively"""
        def _get_descendants(message):
            descendants = []
            for child in message.child_messages:
                descendants.append(child)
                descendants.extend(_get_descendants(child))
            return descendants
        
        return _get_descendants(self)
    
    def get_message_path(self, session):
        """Get the path from root message to this message"""
        path = []
        current = self
        while current is not None:
            path.insert(0, current)
            current = current.parent_message
        return path
    
    def get_thread_tree(self, session):
        """Get the entire message tree for this thread, starting from root"""
        if self.thread_id is None:
            return []
        
        # Find the root message(s) of this thread
        root_messages = session.query(Message).filter(
            Message.thread_id == self.thread_id,
            Message.parent_id == None
        ).order_by(Message.datetime).all()
        
        return root_messages


class Thread(Base):
    """Conversation thread model"""
    __tablename__ = 'threads'

    topic_id = Column(String(50), primary_key=True)
    header = Column(Text, nullable=True)
    actual_date = Column(DateTime(timezone=True), nullable=False, index=True)
    answer_id = Column(String(50), nullable=True)
    label = Column(String(20), nullable=True, index=True)  # resolved, unresolved, suggestion, outside
    solution = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Processing status
    status = Column(String(20), default=ThreadStatus.NEW, index=True)  # new, modified, persisted
    is_technical = Column(Boolean, default=False, index=True)
    is_processed = Column(Boolean, default=False, index=True)
    
    # Relationships - one thread has many messages
    # CRITICAL: No cascade delete on messages - messages must NEVER be deleted when thread is deleted
    messages = relationship("Message", back_populates="thread")
    # Solutions can be cascade deleted as they are derived from threads
    solutions = relationship("Solution", back_populates="thread", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Thread(topic_id='{self.topic_id}', header='{self.header[:50]}...')>"


# ThreadMessage junction table removed - using direct many-to-one relationship


class Solution(Base):
    """Extracted solution model"""
    __tablename__ = 'solutions'

    id = Column(Integer, primary_key=True)
    thread_id = Column(String(50), ForeignKey('threads.topic_id'), nullable=False, unique=True)
    header = Column(Text, nullable=False)
    solution = Column(Text, nullable=False)
    label = Column(String(20), nullable=False)  # resolved, unresolved, suggestion, outside
    confidence_score = Column(Integer, nullable=True)  # 0-100
    
    # Duplicate tracking fields
    is_duplicate = Column(Boolean, default=False, nullable=False, index=True)
    duplicate_count = Column(Integer, default=0, nullable=False)  # Count of duplicates pointing to this solution
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    version = Column(Integer, default=1)
    
    # Relationships
    thread = relationship("Thread", back_populates="solutions")
    embeddings = relationship("SolutionEmbedding", back_populates="solution", cascade="all, delete-orphan")
    similarities = relationship("SolutionSimilarity", back_populates="solution_a", 
                               foreign_keys="SolutionSimilarity.solution_a_id", 
                               cascade="all, delete-orphan")
    # Duplicate relationships
    duplicates = relationship("SolutionDuplicate", back_populates="solution", 
                             foreign_keys="SolutionDuplicate.solution_id",
                             cascade="all, delete-orphan")
    original_of = relationship("SolutionDuplicate", back_populates="original_solution",
                              foreign_keys="SolutionDuplicate.original_solution_id")
    
    def __repr__(self):
        return f"<solution(id={self.id}, header='{self.header[:50]}...')>"


class SolutionDuplicate(Base):
    """Track duplicate relationships between solutions"""
    __tablename__ = 'solution_duplicates'

    id = Column(Integer, primary_key=True)
    solution_id = Column(Integer, ForeignKey('solutions.id'), nullable=False)
    original_solution_id = Column(Integer, ForeignKey('solutions.id'), nullable=False)
    similarity_score = Column(String(10), nullable=False)  # Cosine similarity as string
    status = Column(String(20), default='pending_review', nullable=False, index=True)  
    # Status: 'pending_review', 'confirmed_duplicate', 'false_positive'
    
    # Admin review fields
    reviewed_by = Column(String(100), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    notes = Column(Text, nullable=True)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    solution = relationship("Solution", back_populates="duplicates", foreign_keys=[solution_id])
    original_solution = relationship("Solution", back_populates="original_of", foreign_keys=[original_solution_id])
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_duplicate_solution', 'solution_id'),
        Index('idx_duplicate_original', 'original_solution_id'),
        Index('idx_duplicate_status', 'status'),
        Index('idx_duplicate_similarity', 'similarity_score'),
        # Prevent duplicate entries for the same solution pair
        Index('idx_unique_duplicate_pair', 'solution_id', 'original_solution_id', unique=True),
    )
    
    def __repr__(self):
        return f"<SolutionDuplicate(solution_id={self.solution_id}, original_id={self.original_solution_id}, status='{self.status}')>"


class SolutionEmbedding(Base):
    """Vector embeddings for solutions"""
    __tablename__ = 'solution_embeddings'

    id = Column(Integer, primary_key=True)
    solution_id = Column(Integer, ForeignKey('solutions.id'), nullable=False, unique=True)
    embedding = Column(Vector(768), nullable=False)  # Will be Vector(768) or JSON based on availability
    embedding_model = Column(String(100), nullable=False, default="text-embedding-004")
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    solution = relationship("Solution", back_populates="embeddings")
    
    # Index for vector similarity search (only if pgvector is available)
    if PGVECTOR_AVAILABLE:
        __table_args__ = (
            Index('idx_solution_embedding_vector', 'embedding', postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'}),
        )
    else:
        __table_args__ = (
            Index('idx_solution_embedding_json', 'embedding'),
        )


class SolutionSimilarity(Base):
    """Pre-computed similarity scores between solutions"""
    __tablename__ = 'solution_similarities'

    id = Column(Integer, primary_key=True)
    solution_a_id = Column(Integer, ForeignKey('solutions.id'), nullable=False)
    solution_b_id = Column(Integer, ForeignKey('solutions.id'), nullable=False)
    similarity_score = Column(String(10), nullable=False)  # cosine similarity score as string
    computed_at = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    solution_a = relationship("Solution", back_populates="similarities", foreign_keys=[solution_a_id])
    
    # Unique constraint and indexes
    __table_args__ = (
        Index('idx_similarity_pair', 'solution_a_id', 'solution_b_id', unique=True),
        Index('idx_similarity_score', 'similarity_score'),
    )


class ProcessingBatch(Base):
    """Track batch processing history"""
    __tablename__ = 'processing_batches'

    id = Column(Integer, primary_key=True)
    batch_type = Column(String(20), nullable=False)  # first, next
    start_date = Column(DateTime(timezone=True), nullable=False)
    end_date = Column(DateTime(timezone=True), nullable=False)
    lookback_date = Column(DateTime(timezone=True), nullable=True)
    
    # Execution info
    started_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default='running')  # running, completed, failed
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ProcessingBatch(id={self.id}, type='{self.batch_type}', status='{self.status}')>"


class MessageProcessing(Base):
    """Track each processing step applied to individual messages"""
    __tablename__ = 'message_processing'

    id = Column(Integer, primary_key=True)
    message_id = Column(String(50), ForeignKey('messages.message_id'), nullable=False)
    processing_step = Column(String(50), nullable=False, index=True)
    step_order = Column(Integer, nullable=False)
    result = Column(JSON, nullable=True)
    confidence_score = Column(String(10), nullable=True)  # Store as string for consistency
    processing_metadata = Column(JSON, nullable=True)
    processed_at = Column(DateTime(timezone=True), default=func.now())
    processing_version = Column(String(20), nullable=True)
    
    # Relationships
    message = relationship("Message", back_populates="processing_steps")
    
    # Indexes
    __table_args__ = (
        Index('idx_message_processing', 'message_id', 'processing_step'),
        Index('idx_processing_step_order', 'processing_step', 'step_order'),
        Index('idx_processing_timestamp', 'processed_at'),
    )
    
    def __repr__(self):
        return f"<MessageProcessing(message_id={self.message_id}, step='{self.processing_step}')>"


class ProcessingPipeline(Base):
    """Define and track processing pipeline configuration"""
    __tablename__ = 'processing_pipeline'

    id = Column(Integer, primary_key=True)
    pipeline_name = Column(String(100), nullable=False, index=True)
    step_order = Column(Integer, nullable=False)
    step_name = Column(String(50), nullable=False)
    step_config = Column(JSON, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_pipeline_order', 'pipeline_name', 'step_order'),
        Index('idx_pipeline_active', 'pipeline_name', 'is_active'),
    )
    
    def __repr__(self):
        return f"<ProcessingPipeline(pipeline='{self.pipeline_name}', step='{self.step_name}')>"


class MessageAnnotation(Base):
    """Store message-level classifications and annotations"""
    __tablename__ = 'message_annotations'

    id = Column(Integer, primary_key=True)
    message_id = Column(String(50), ForeignKey('messages.message_id'), nullable=False)
    annotation_type = Column(String(50), nullable=False, index=True)
    annotation_value = Column(JSON, nullable=True)
    confidence_score = Column(String(10), nullable=True)
    annotated_by = Column(String(50), nullable=False, index=True)  # 'gemini_ai', 'manual', 'rule_based'
    annotated_at = Column(DateTime(timezone=True), default=func.now())
    
    # Relationships
    message = relationship("Message", back_populates="annotations")
    
    # Indexes
    __table_args__ = (
        Index('idx_message_annotations', 'message_id', 'annotation_type'),
        Index('idx_annotation_type_confidence', 'annotation_type', 'confidence_score'),
        Index('idx_annotation_timestamp', 'annotated_at'),
    )
    
    def __repr__(self):
        return f"<MessageAnnotation(message_id={self.message_id}, type='{self.annotation_type}')>"


class LLMCache(Base):
    """Cache for LLM responses to reduce API calls"""
    __tablename__ = 'llm_cache'

    id = Column(Integer, primary_key=True)
    request_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA256 hash of request
    request_type = Column(String(50), nullable=False, index=True)  # step_1, step_2, step_3, revision
    response_data = Column(JSON, nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), default=func.now())
    last_accessed = Column(DateTime(timezone=True), default=func.now())
    access_count = Column(Integer, default=1)
    ttl_expires_at = Column(DateTime(timezone=True), nullable=True, index=True)
    
    def __repr__(self):
        return f"<LLMCache(id={self.id}, type='{self.request_type}', accessed={self.access_count} times)>"