from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON, Index, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import ARRAY
from datetime import datetime
from typing import Optional, List
import os

# Try to import pgvector, but fallback gracefully if not available
try:
    from pgvector.sqlalchemy import Vector
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False
    # Create a mock Vector type that falls back to JSON
    def Vector(dimension):
        return JSON

Base = declarative_base()


class Message(Base):
    """Discord message model"""
    __tablename__ = 'messages'

    id = Column(Integer, primary_key=True)
    message_id = Column(String(50), unique=True, nullable=False, index=True)
    author_id = Column(String(50), nullable=False, index=True)
    content = Column(Text, nullable=False)
    datetime = Column(DateTime(timezone=True), nullable=False, index=True)
    dated_message = Column(Text, nullable=False)
    referenced_message_id = Column(String(50), nullable=True, index=True)
    
    # Relationships
    threads = relationship("ThreadMessage", back_populates="message")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_message_datetime_author', 'datetime', 'author_id'),
        Index('idx_message_referenced', 'referenced_message_id'),
    )

    def __repr__(self):
        return f"<Message(message_id='{self.message_id}', author_id='{self.author_id}')>"


class Thread(Base):
    """Conversation thread model"""
    __tablename__ = 'threads'

    id = Column(Integer, primary_key=True)
    topic_id = Column(String(50), unique=True, nullable=False, index=True)
    header = Column(Text, nullable=True)
    actual_date = Column(DateTime(timezone=True), nullable=False, index=True)
    answer_id = Column(String(50), nullable=True)
    label = Column(String(20), nullable=True, index=True)  # resolved, unresolved, suggestion, outside
    solution = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())
    
    # Processing status
    status = Column(String(20), default='new', index=True)  # new, modified, persisted
    is_technical = Column(Boolean, default=False, index=True)
    is_processed = Column(Boolean, default=False, index=True)
    
    # Relationships
    messages = relationship("ThreadMessage", back_populates="thread", cascade="all, delete-orphan")
    solutions = relationship("Solution", back_populates="thread", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Thread(topic_id='{self.topic_id}', header='{self.header[:50]}...')>"


class ThreadMessage(Base):
    """Many-to-many relationship between threads and messages"""
    __tablename__ = 'thread_messages'

    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('threads.id'), nullable=False)
    message_id = Column(Integer, ForeignKey('messages.id'), nullable=False)
    order_in_thread = Column(Integer, nullable=False)
    
    # Relationships
    thread = relationship("Thread", back_populates="messages")
    message = relationship("Message", back_populates="threads")
    
    # Unique constraint
    __table_args__ = (
        Index('idx_thread_message_unique', 'thread_id', 'message_id', unique=True),
        Index('idx_thread_order', 'thread_id', 'order_in_thread'),
    )


class Solution(Base):
    """Extracted solution model"""
    __tablename__ = 'solutions'

    id = Column(Integer, primary_key=True)
    thread_id = Column(Integer, ForeignKey('threads.id'), nullable=False, unique=True)
    header = Column(Text, nullable=False)
    solution = Column(Text, nullable=False)
    label = Column(String(20), nullable=False)  # resolved, unresolved, suggestion, outside
    confidence_score = Column(Integer, nullable=True)  # 0-100
    
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
    
    def __repr__(self):
        return f"<Solution(id={self.id}, header='{self.header[:50]}...')>"


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
    
    # Processing stats
    messages_processed = Column(Integer, default=0)
    threads_created = Column(Integer, default=0)
    threads_modified = Column(Integer, default=0)
    technical_threads = Column(Integer, default=0)
    solutions_added = Column(Integer, default=0)
    
    # Execution info
    started_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(String(20), default='running')  # running, completed, failed
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<ProcessingBatch(id={self.id}, type='{self.batch_type}', status='{self.status}')>"


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