
from fastapi import APIRouter, HTTPException
from app.utils.analytics import get_analytics_service
from app.services.database import get_database_service
from app.models.db_models import Solution, Message, Thread, MessageProcessing, MessageAnnotation, ProcessingPipeline
from app.services.processing_tracker import get_processing_tracker
import logging

router = APIRouter()

@router.get("/admin/duplicates")
def get_pending_duplicates(limit: int = 50, offset: int = 0):
    """Get duplicates pending admin review."""
    try:
        analytics_service = get_analytics_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            pending_duplicates = analytics_service.get_pending_duplicates_for_review(session, limit)
            return {
                "pending_duplicates": pending_duplicates,
                "total_count": len(pending_duplicates),
                "limit": limit,
                "offset": offset
            }
    except Exception as e:
        logging.error(f"Error fetching pending duplicates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/duplicates/{solution_id}")
def get_solution_duplicates(solution_id: int):
    """Get duplicate chain for a specific solution."""
    try:
        analytics_service = get_analytics_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            duplicate_chain = analytics_service.get_solution_duplicate_chain(session, solution_id)
            return duplicate_chain
    except Exception as e:
        logging.error(f"Error fetching solution duplicates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/admin/duplicates/{duplicate_id}/review")
def review_duplicate(duplicate_id: int, status: str, reviewed_by: str, notes: str = None):
    """Review and update duplicate status."""
    try:
        valid_statuses = ['confirmed_duplicate', 'false_positive', 'pending_review']
        if status not in valid_statuses:
            raise HTTPException(status_code=400, detail=f"Invalid status. Must be one of: {valid_statuses}")
        
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            success = db_service.update_duplicate_status(session, duplicate_id, status, reviewed_by, notes)
            
            if not success:
                raise HTTPException(status_code=404, detail="Duplicate record not found")
            
            session.commit()
            
            return {
                "duplicate_id": duplicate_id,
                "status": status,
                "reviewed_by": reviewed_by,
                "notes": notes,
                "message": "Duplicate status updated successfully"
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error updating duplicate status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/frequency-analysis")
def get_frequency_analysis(limit: int = 20, min_duplicates: int = 2):
    """Get problem frequency analysis."""
    try:
        analytics_service = get_analytics_service()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            frequent_problems = analytics_service.get_frequent_problems(session, limit, min_duplicates)
            duplicate_stats = analytics_service.get_duplicate_statistics(session)
            trends = analytics_service.get_duplicate_trends(session, days_back=30)
            categories = analytics_service.get_problem_categories_analysis(session)
            
            return {
                "frequent_problems": frequent_problems,
                "statistics": duplicate_stats,
                "trends": trends,
                "categories": categories
            }
    except Exception as e:
        logging.error(f"Error fetching frequency analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/solutions/{topic_id}/duplicates")
def get_topic_duplicates(topic_id: str):
    """Get all duplicates of a specific solution by topic ID."""
    try:
        db_service = get_database_service()
        analytics_service = get_analytics_service()
        
        with db_service.get_session() as session:
            thread = db_service.get_thread_by_topic_id(session, topic_id)
            if not thread:
                raise HTTPException(status_code=404, detail=f"Thread with topic_id {topic_id} not found")
            
            solution = session.query(Solution).filter(Solution.thread_id == thread.id).first()
            if not solution:
                raise HTTPException(status_code=404, detail=f"Solution for topic_id {topic_id} not found")
            
            duplicate_chain = analytics_service.get_solution_duplicate_chain(session, solution.id)
            return duplicate_chain
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching topic duplicates: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/admin/duplicate-statistics")
def get_duplicate_statistics():
    """Get comprehensive duplicate statistics for admin dashboard."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            stats = db_service.get_duplicate_statistics(session)
            return stats
    except Exception as e:
        logging.error(f"Error fetching duplicate statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/admin/recalculate-duplicate-counts")
def recalculate_duplicate_counts():
    """Recalculate duplicate counts for all solutions."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            updated_count = db_service.bulk_update_duplicate_counts(session)
            session.commit()
            
            return {
                "message": f"Successfully updated duplicate counts for {updated_count} solutions"
            }
    except Exception as e:
        logging.error(f"Error recalculating duplicate counts: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/processing/messages/{message_id}")
def get_message_processing_history(message_id: str):
    """Get complete processing history for a message by message_id."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            message = session.query(Message).filter(Message.message_id == message_id).first()
            if not message:
                raise HTTPException(status_code=404, detail=f"Message with ID {message_id} not found")
            
            processing_history = processing_tracker.get_message_processing_history(session, message.id)
            return processing_history
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching message processing history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/threads/{topic_id}")
def get_thread_processing_history(topic_id: str):
    """Get complete processing history for a thread by topic_id."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            thread = session.query(Thread).filter(Thread.topic_id == topic_id).first()
            if not thread:
                raise HTTPException(status_code=404, detail=f"Thread with topic_id {topic_id} not found")
            
            processing_history = processing_tracker.get_thread_processing_history(session, thread.id)
            return processing_history
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching thread processing history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/pipeline")
def get_processing_pipeline():
    """Get current processing pipeline configuration."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            
            pipeline_steps = session.query(ProcessingPipeline).filter(
                ProcessingPipeline.is_active == True
            ).order_by(ProcessingPipeline.step_order).all()
            
            pipeline = {
                'pipeline_name': 'default',
                'steps': [
                    {
                        'order': step.step_order,
                        'name': step.step_name,
                        'config': step.step_config,
                        'created_at': step.created_at.isoformat()
                    }
                    for step in pipeline_steps
                ]
            }
            
            return pipeline
    except Exception as e:
        logging.error(f"Error fetching processing pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processing/pipeline")
def create_processing_pipeline(pipeline_data: dict):
    """Create or update processing pipeline configuration."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        pipeline_name = pipeline_data.get('pipeline_name', 'default')
        steps = pipeline_data.get('steps', [])
        
        if not steps:
            raise HTTPException(status_code=400, detail="Steps are required")
        
        with db_service.get_session() as session:
            success = processing_tracker.create_processing_pipeline(session, pipeline_name, steps)
            
            if not success:
                raise HTTPException(status_code=500, detail="Failed to create processing pipeline")
            
            session.commit()
            
            return {
                "message": f"Processing pipeline '{pipeline_name}' created successfully",
                "pipeline_name": pipeline_name,
                "steps_count": len(steps)
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating processing pipeline: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/messages/{message_id}/annotations")
def get_message_annotations(message_id: str):
    """Get all annotations for a message."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            message = session.query(Message).filter(Message.message_id == message_id).first()
            if not message:
                raise HTTPException(status_code=404, detail=f"Message with ID {message_id} not found")
            
            annotations = session.query(MessageAnnotation).filter(
                MessageAnnotation.message_id == message.id
            ).order_by(MessageAnnotation.annotated_at).all()
            
            return {
                'message_id': message_id,
                'annotations': [
                    {
                        'id': ann.id,
                        'type': ann.annotation_type,
                        'value': ann.annotation_value,
                        'confidence_score': float(ann.confidence_score) if ann.confidence_score else None,
                        'annotated_by': ann.annotated_by,
                        'annotated_at': ann.annotated_at.isoformat()
                    }
                    for ann in annotations
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching message annotations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/processing/messages/{message_id}/annotate")
def annotate_message(message_id: str, annotation_data: dict):
    """Add manual annotation to a message."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        annotation_type = annotation_data.get('type')
        annotation_value = annotation_data.get('value')
        confidence_score = annotation_data.get('confidence_score')
        annotated_by = annotation_data.get('annotated_by', 'manual')
        
        if not annotation_type:
            raise HTTPException(status_code=400, detail="Annotation type is required")
        
        with db_service.get_session() as session:
            message = session.query(Message).filter(Message.message_id == message_id).first()
            if not message:
                raise HTTPException(status_code=404, detail=f"Message with ID {message_id} not found")
            
            annotation = processing_tracker.annotate_message(
                session=session,
                message_id=message.id,
                annotation_type=annotation_type,
                annotation_value=annotation_value,
                confidence_score=confidence_score,
                annotated_by=annotated_by
            )
            
            if not annotation:
                raise HTTPException(status_code=500, detail="Failed to create annotation")
            
            session.commit()
            
            return {
                "message": "Annotation created successfully",
                "annotation_id": annotation.id,
                "message_id": message_id,
                "type": annotation_type
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating message annotation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/annotations/search")
def search_annotations(annotation_type: str = None, annotated_by: str = None, limit: int = 100):
    """Search messages by annotation type or annotated_by."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            
            query = session.query(MessageAnnotation, Message).join(
                Message, MessageAnnotation.message_id == Message.id
            )
            
            if annotation_type:
                query = query.filter(MessageAnnotation.annotation_type == annotation_type)
            
            if annotated_by:
                query = query.filter(MessageAnnotation.annotated_by == annotated_by)
            
            results = query.order_by(MessageAnnotation.annotated_at.desc()).limit(limit).all()
            
            return {
                'results': [
                    {
                        'annotation_id': ann.id,
                        'message_id': msg.message_id,
                        'annotation_type': ann.annotation_type,
                        'annotation_value': ann.annotation_value,
                        'confidence_score': float(ann.confidence_score) if ann.confidence_score else None,
                        'annotated_by': ann.annotated_by,
                        'annotated_at': ann.annotated_at.isoformat(),
                        'message_content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,
                        'message_datetime': msg.datetime.isoformat()
                    }
                    for ann, msg in results
                ]
            }
    except Exception as e:
        logging.error(f"Error searching annotations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/processing/statistics")
def get_processing_statistics():
    """Get comprehensive processing statistics."""
    try:
        processing_tracker = get_processing_tracker()
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            stats = processing_tracker.get_processing_statistics(session)
            return stats
    except Exception as e:
        logging.error(f"Error fetching processing statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/processing/{date}")
def get_processing_audit_log(date: str):
    """Get processing audit log for a specific date (YYYY-MM-DD)."""
    try:
        from datetime import datetime, timedelta
        
        try:
            target_date = datetime.strptime(date, '%Y-%m-%d').date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
        
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            
            # Get processing steps for the specified date
            start_datetime = datetime.combine(target_date, datetime.min.time())
            end_datetime = start_datetime + timedelta(days=1)
            
            processing_steps = session.query(MessageProcessing, Message).join(
                Message, MessageProcessing.message_id == Message.id
            ).filter(
                MessageProcessing.processed_at >= start_datetime,
                MessageProcessing.processed_at < end_datetime
            ).order_by(MessageProcessing.processed_at).all()
            
            return {
                'date': date,
                'total_processing_steps': len(processing_steps),
                'processing_steps': [
                    {
                        'message_id': msg.message_id,
                        'processing_step': ps.processing_step,
                        'step_order': ps.step_order,
                        'result': ps.result,
                        'confidence_score': float(ps.confidence_score) if ps.confidence_score else None,
                        'processed_at': ps.processed_at.isoformat(),
                        'processing_version': ps.processing_version
                    }
                    for ps, msg in processing_steps
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching processing audit log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit/confidence/low")
def get_low_confidence_classifications(threshold: float = 0.7, limit: int = 50):
    """Get classifications with low confidence scores for review."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            
            # Get low confidence processing steps
            low_confidence_processing = session.query(MessageProcessing, Message).join(
                Message, MessageProcessing.message_id == Message.id
            ).filter(
                MessageProcessing.confidence_score.isnot(None),
                MessageProcessing.confidence_score.cast(db_service.engine.dialect.numeric_type) < threshold
            ).order_by(MessageProcessing.confidence_score).limit(limit // 2).all()
            
            # Get low confidence annotations
            low_confidence_annotations = session.query(MessageAnnotation, Message).join(
                Message, MessageAnnotation.message_id == Message.id
            ).filter(
                MessageAnnotation.confidence_score.isnot(None),
                MessageAnnotation.confidence_score.cast(db_service.engine.dialect.numeric_type) < threshold
            ).order_by(MessageAnnotation.confidence_score).limit(limit // 2).all()
            
            return {
                'threshold': threshold,
                'low_confidence_processing': [
                    {
                        'message_id': msg.message_id,
                        'processing_step': ps.processing_step,
                        'confidence_score': float(ps.confidence_score),
                        'result': ps.result,
                        'processed_at': ps.processed_at.isoformat(),
                        'message_content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content
                    }
                    for ps, msg in low_confidence_processing
                ],
                'low_confidence_annotations': [
                    {
                        'message_id': msg.message_id,
                        'annotation_type': ann.annotation_type,
                        'confidence_score': float(ann.confidence_score),
                        'annotation_value': ann.annotation_value,
                        'annotated_at': ann.annotated_at.isoformat(),
                        'message_content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content
                    }
                    for ann, msg in low_confidence_annotations
                ]
            }
    except Exception as e:
        logging.error(f"Error fetching low confidence classifications: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
