"""
Processing tracking service for recording each processing step and maintaining audit trails.

This service implements the layered processing strategy where original messages remain
unchanged and each processing step adds metadata layers.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union
from sqlalchemy.orm import Session

from app.services.database import get_database_service
from app.models.db_models import (
    Message, Thread, Solution, MessageProcessing, MessageAnnotation, 
    ProcessingPipeline
)


class ProcessingTracker:
    """Service for tracking and recording all processing steps."""
    
    def __init__(self, processing_version: str = "1.0"):
        self.logger = logging.getLogger(__name__)
        self.db_service = get_database_service()
        self.processing_version = processing_version
        
        # Define processing step constants
        self.STEPS = {
            'DATA_LOADING': 'data_loading',
            'THREAD_GROUPING': 'thread_grouping', 
            'TECHNICAL_FILTERING': 'technical_filtering',
            'SOLUTION_EXTRACTION': 'solution_extraction',
            'DUPLICATE_DETECTION': 'duplicate_detection',
            'RAG_PROCESSING': 'rag_processing',
            'MANUAL_ANNOTATION': 'manual_annotation'
        }
        
        # Annotation types
        self.ANNOTATION_TYPES = {
            'TECHNICAL': 'technical',
            'QUESTION': 'question',
            'ANSWER': 'answer',
            'FOLLOW_UP': 'follow_up',
            'CLARIFICATION': 'clarification',
            'SOLUTION': 'solution',
            'DUPLICATE': 'duplicate'
        }
    
    def record_processing_step(self, session: Session, message_id: int, 
                             processing_step: str, result: Dict[str, Any] = None,
                             confidence_score: float = None, metadata: Dict[str, Any] = None,
                             step_order: int = None) -> Optional[MessageProcessing]:
        """
        Record a processing step for a specific message.
        
        Args:
            session: Database session
            message_id: ID of the message being processed
            processing_step: Name of the processing step
            result: Results/output from the processing step
            confidence_score: Confidence score for the processing result
            metadata: Additional metadata about the processing
            step_order: Order of this step in the processing pipeline
        
        Returns:
            MessageProcessing record if successful, None otherwise
        """
        try:
            # Get the message to ensure it exists
            message = session.query(Message).filter(Message.id == message_id).first()
            if not message:
                self.logger.error(f"Message with ID {message_id} not found")
                return None
            
            # Determine step order if not provided
            if step_order is None:
                # Get the highest step order for this message and increment
                last_step = session.query(MessageProcessing).filter(
                    MessageProcessing.message_id == message_id
                ).order_by(MessageProcessing.step_order.desc()).first()
                step_order = (last_step.step_order + 1) if last_step else 1
            
            # Create processing record
            processing_record = MessageProcessing(
                message_id=message_id,
                processing_step=processing_step,
                step_order=step_order,
                result=result or {},
                confidence_score=f"{confidence_score:.6f}" if confidence_score is not None else None,
                processing_metadata=metadata or {},
                processing_version=self.processing_version
            )
            
            session.add(processing_record)
            
            # Update message processing status
            if not message.processing_status:
                message.processing_status = {}
            
            message.processing_status[processing_step] = {
                'completed_at': datetime.now(timezone.utc).isoformat(),
                'confidence_score': confidence_score,
                'step_order': step_order
            }
            message.last_processed_at = datetime.now(timezone.utc)
            message.processing_version = self.processing_version
            
            session.flush()
            
            self.logger.info(f"Recorded processing step '{processing_step}' for message {message_id}")
            return processing_record
            
        except Exception as e:
            self.logger.error(f"Failed to record processing step: {e}")
            return None
    
    def annotate_message(self, session: Session, message_id: int, 
                        annotation_type: str, annotation_value: Any = None,
                        confidence_score: float = None, annotated_by: str = 'gemini_ai') -> Optional[MessageAnnotation]:
        """
        Add an annotation to a message.
        
        Args:
            session: Database session
            message_id: ID of the message being annotated
            annotation_type: Type of annotation (technical, question, answer, etc.)
            annotation_value: Value/content of the annotation
            confidence_score: Confidence score for the annotation
            annotated_by: Source of the annotation (gemini_ai, manual, rule_based)
        
        Returns:
            MessageAnnotation record if successful, None otherwise
        """
        try:
            # Get the message to ensure it exists
            message = session.query(Message).filter(Message.id == message_id).first()
            if not message:
                self.logger.error(f"Message with ID {message_id} not found")
                return None
            
            # Create annotation
            annotation = MessageAnnotation(
                message_id=message_id,
                annotation_type=annotation_type,
                annotation_value=annotation_value,
                confidence_score=f"{confidence_score:.6f}" if confidence_score is not None else None,
                annotated_by=annotated_by
            )
            
            session.add(annotation)
            session.flush()
            
            self.logger.info(f"Added annotation '{annotation_type}' to message {message_id}")
            return annotation
            
        except Exception as e:
            self.logger.error(f"Failed to annotate message: {e}")
            return None
    
    def record_thread_processing(self, session: Session, thread_id: int,
                               processing_step: str, result: Dict[str, Any] = None,
                               confidence_scores: Dict[str, float] = None, 
                               metadata: Dict[str, Any] = None):
        """
        Record processing information for a thread.
        
        Args:
            session: Database session
            thread_id: ID of the thread being processed
            processing_step: Name of the processing step
            result: Results from the processing step
            confidence_scores: Confidence scores for various aspects
            metadata: Additional processing metadata
        """
        try:
            thread = session.query(Thread).filter(Thread.id == thread_id).first()
            if not thread:
                self.logger.error(f"Thread with ID {thread_id} not found")
                return
            
            # Update processing history
            if not thread.processing_history:
                thread.processing_history = []
            
            history_entry = {
                'step': processing_step,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'result': result or {},
                'metadata': metadata or {},
                'version': self.processing_version
            }
            thread.processing_history.append(history_entry)
            
            # Update confidence scores
            if confidence_scores:
                if not thread.confidence_scores:
                    thread.confidence_scores = {}
                thread.confidence_scores.update({
                    k: float(v) for k, v in confidence_scores.items()
                })
            
            # Update processing metadata
            if metadata:
                if not thread.processing_metadata:
                    thread.processing_metadata = {}
                thread.processing_metadata.update(metadata)
            
            thread.updated_at = datetime.now(timezone.utc)
            session.flush()
            
            self.logger.info(f"Recorded thread processing step '{processing_step}' for thread {thread_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record thread processing: {e}")
    
    def record_solution_extraction(self, session: Session, solution_id: int,
                                 extraction_metadata: Dict[str, Any] = None,
                                 source_message_ids: List[int] = None,
                                 processing_steps: List[Dict[str, Any]] = None):
        """
        Record solution extraction metadata.
        
        Args:
            session: Database session
            solution_id: ID of the solution
            extraction_metadata: Metadata about the extraction process
            source_message_ids: List of message IDs that contributed to this solution
            processing_steps: List of processing steps taken to generate solution
        """
        try:
            solution = session.query(Solution).filter(Solution.id == solution_id).first()
            if not solution:
                self.logger.error(f"Solution with ID {solution_id} not found")
                return
            
            # Update extraction metadata
            if extraction_metadata:
                if not solution.extraction_metadata:
                    solution.extraction_metadata = {}
                solution.extraction_metadata.update(extraction_metadata)
            
            # Update source messages
            if source_message_ids:
                solution.source_messages = source_message_ids
            
            # Update processing steps
            if processing_steps:
                solution.processing_steps = processing_steps
            
            solution.updated_at = datetime.now(timezone.utc)
            session.flush()
            
            self.logger.info(f"Recorded solution extraction metadata for solution {solution_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to record solution extraction: {e}")
    
    def get_message_processing_history(self, session: Session, message_id: int) -> Dict[str, Any]:
        """Get complete processing history for a message."""
        try:
            # Get message
            message = session.query(Message).filter(Message.id == message_id).first()
            if not message:
                return {'error': 'Message not found'}
            
            # Get processing steps
            processing_steps = session.query(MessageProcessing).filter(
                MessageProcessing.message_id == message_id
            ).order_by(MessageProcessing.step_order).all()
            
            # Get annotations
            annotations = session.query(MessageAnnotation).filter(
                MessageAnnotation.message_id == message_id
            ).order_by(MessageAnnotation.annotated_at).all()
            
            return {
                'message_id': message.message_id,
                'processing_status': message.processing_status,
                'last_processed_at': message.last_processed_at.isoformat() if message.last_processed_at else None,
                'processing_version': message.processing_version,
                'processing_steps': [
                    {
                        'step': ps.processing_step,
                        'order': ps.step_order,
                        'result': ps.result,
                        'confidence_score': float(ps.confidence_score) if ps.confidence_score else None,
                        'metadata': ps.processing_metadata,
                        'processed_at': ps.processed_at.isoformat(),
                        'version': ps.processing_version
                    }
                    for ps in processing_steps
                ],
                'annotations': [
                    {
                        'type': ann.annotation_type,
                        'value': ann.annotation_value,
                        'confidence_score': float(ann.confidence_score) if ann.confidence_score else None,
                        'annotated_by': ann.annotated_by,
                        'annotated_at': ann.annotated_at.isoformat()
                    }
                    for ann in annotations
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get message processing history: {e}")
            return {'error': str(e)}
    
    def get_thread_processing_history(self, session: Session, thread_id: int) -> Dict[str, Any]:
        """Get complete processing history for a thread."""
        try:
            thread = session.query(Thread).filter(Thread.id == thread_id).first()
            if not thread:
                return {'error': 'Thread not found'}
            
            return {
                'thread_id': thread.topic_id,
                'processing_history': thread.processing_history,
                'confidence_scores': thread.confidence_scores,
                'processing_metadata': thread.processing_metadata,
                'status': thread.status,
                'is_technical': thread.is_technical,
                'is_processed': thread.is_processed,
                'created_at': thread.created_at.isoformat(),
                'updated_at': thread.updated_at.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get thread processing history: {e}")
            return {'error': str(e)}
    
    def create_processing_pipeline(self, session: Session, pipeline_name: str, 
                                 steps: List[Dict[str, Any]]) -> bool:
        """
        Create or update a processing pipeline configuration.
        
        Args:
            session: Database session
            pipeline_name: Name of the pipeline
            steps: List of step configurations
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove existing pipeline steps
            session.query(ProcessingPipeline).filter(
                ProcessingPipeline.pipeline_name == pipeline_name
            ).delete()
            
            # Add new steps
            for step_info in steps:
                pipeline_step = ProcessingPipeline(
                    pipeline_name=pipeline_name,
                    step_order=step_info['order'],
                    step_name=step_info['name'],
                    step_config=step_info.get('config', {}),
                    is_active=step_info.get('is_active', True)
                )
                session.add(pipeline_step)
            
            session.flush()
            self.logger.info(f"Created processing pipeline '{pipeline_name}' with {len(steps)} steps")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create processing pipeline: {e}")
            return False
    
    def get_processing_statistics(self, session: Session) -> Dict[str, Any]:
        """Get statistics about processing across the system."""
        try:
            # Message processing stats
            total_messages = session.query(Message).count()
            processed_messages = session.query(Message).filter(
                Message.last_processed_at.isnot(None)
            ).count()
            
            # Step counts
            step_counts = {}
            for step_name in self.STEPS.values():
                count = session.query(MessageProcessing).filter(
                    MessageProcessing.processing_step == step_name
                ).count()
                step_counts[step_name] = count
            
            # Annotation stats
            annotation_counts = {}
            for ann_type in self.ANNOTATION_TYPES.values():
                count = session.query(MessageAnnotation).filter(
                    MessageAnnotation.annotation_type == ann_type
                ).count()
                annotation_counts[ann_type] = count
            
            return {
                'total_messages': total_messages,
                'processed_messages': processed_messages,
                'processing_coverage': processed_messages / total_messages if total_messages > 0 else 0,
                'step_counts': step_counts,
                'annotation_counts': annotation_counts,
                'processing_version': self.processing_version
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get processing statistics: {e}")
            return {'error': str(e)}


# Global processing tracker instance
processing_tracker = None

def get_processing_tracker() -> ProcessingTracker:
    """Get global processing tracker instance."""
    global processing_tracker
    if processing_tracker is None:
        processing_tracker = ProcessingTracker()
    return processing_tracker