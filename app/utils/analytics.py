"""
Analytics utilities for duplicate solution tracking and frequency analysis.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import text, func, desc, and_, or_
import pandas as pd

from app.models.db_models import Solution, SolutionDuplicate, Thread
from app.services.database import get_database_service


class AnalyticsService:
    """Service for analyzing solution duplicates and frequencies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_service = get_database_service()
    
    def get_duplicate_statistics(self, session: Session) -> Dict[str, Any]:
        """Get overall statistics about duplicates in the system."""
        try:
            # Total solutions and duplicates
            total_solutions = session.query(Solution).count()
            total_duplicates = session.query(SolutionDuplicate).count()
            
            # Duplicate statuses
            status_counts = session.query(
                SolutionDuplicate.status,
                func.count(SolutionDuplicate.id)
            ).group_by(SolutionDuplicate.status).all()
            
            # Solutions with most duplicates
            top_duplicated = session.query(
                Solution.id,
                Solution.header,
                Solution.duplicate_count,
                func.count(SolutionDuplicate.original_solution_id).label('actual_duplicate_count')
            ).outerjoin(
                SolutionDuplicate, Solution.id == SolutionDuplicate.original_solution_id
            ).group_by(
                Solution.id, Solution.header, Solution.duplicate_count
            ).having(
                func.count(SolutionDuplicate.original_solution_id) > 0
            ).order_by(
                desc('actual_duplicate_count')
            ).limit(10).all()
            
            return {
                'total_solutions': total_solutions,
                'total_duplicates': total_duplicates,
                'duplicate_rate': total_duplicates / total_solutions if total_solutions > 0 else 0,
                'status_breakdown': {status: count for status, count in status_counts},
                'top_duplicated_solutions': [
                    {
                        'solution_id': sol.id,
                        'header': sol.header[:100] + ('...' if len(sol.header) > 100 else ''),
                        'stored_count': sol.duplicate_count,
                        'actual_count': sol.actual_duplicate_count
                    }
                    for sol in top_duplicated
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get duplicate statistics: {e}")
            return {}
    
    def get_frequent_problems(self, session: Session, limit: int = 20, 
                            min_duplicates: int = 2) -> List[Dict[str, Any]]:
        """Get most frequently occurring problems based on duplicate count."""
        try:
            # Query solutions with their duplicate counts
            frequent_problems = session.query(
                Solution.id,
                Solution.header,
                Solution.solution,
                Solution.label,
                Thread.topic_id,
                Thread.actual_date,
                func.count(SolutionDuplicate.original_solution_id).label('duplicate_count')
            ).join(
                Thread, Solution.thread_id == Thread.id
            ).outerjoin(
                SolutionDuplicate, Solution.id == SolutionDuplicate.original_solution_id
            ).group_by(
                Solution.id, Solution.header, Solution.solution, Solution.label,
                Thread.topic_id, Thread.actual_date
            ).having(
                func.count(SolutionDuplicate.original_solution_id) >= min_duplicates
            ).order_by(
                desc('duplicate_count')
            ).limit(limit).all()
            
            result = []
            for problem in frequent_problems:
                result.append({
                    'solution_id': problem.id,
                    'topic_id': problem.topic_id,
                    'header': problem.header,
                    'solution': problem.solution,
                    'label': problem.label,
                    'first_occurrence': problem.actual_date.isoformat() if problem.actual_date else None,
                    'duplicate_count': problem.duplicate_count,
                    'total_occurrences': problem.duplicate_count + 1  # Include original
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get frequent problems: {e}")
            return []
    
    def get_duplicate_trends(self, session: Session, days_back: int = 30) -> Dict[str, Any]:
        """Analyze duplicate trends over time."""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_back)
            
            # Duplicates created over time
            daily_duplicates = session.query(
                func.date(SolutionDuplicate.created_at).label('date'),
                func.count(SolutionDuplicate.id).label('count')
            ).filter(
                SolutionDuplicate.created_at >= cutoff_date
            ).group_by(
                func.date(SolutionDuplicate.created_at)
            ).order_by('date').all()
            
            # Most active periods (when most duplicates are detected)
            hourly_pattern = session.query(
                func.extract('hour', SolutionDuplicate.created_at).label('hour'),
                func.count(SolutionDuplicate.id).label('count')
            ).filter(
                SolutionDuplicate.created_at >= cutoff_date
            ).group_by(
                func.extract('hour', SolutionDuplicate.created_at)
            ).order_by('hour').all()
            
            return {
                'period_days': days_back,
                'daily_duplicates': [
                    {'date': day.date.isoformat(), 'count': day.count}
                    for day in daily_duplicates
                ],
                'hourly_pattern': [
                    {'hour': int(hour.hour), 'count': hour.count}
                    for hour in hourly_pattern
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get duplicate trends: {e}")
            return {}
    
    def get_problem_categories_analysis(self, session: Session) -> Dict[str, Any]:
        """Analyze duplicates by problem categories (labels)."""
        try:
            # Duplicates by label
            category_duplicates = session.query(
                Solution.label,
                func.count(SolutionDuplicate.original_solution_id).label('duplicate_count'),
                func.count(Solution.id).label('total_solutions')
            ).outerjoin(
                SolutionDuplicate, Solution.id == SolutionDuplicate.original_solution_id
            ).group_by(
                Solution.label
            ).order_by(
                desc('duplicate_count')
            ).all()
            
            return {
                'categories': [
                    {
                        'label': cat.label,
                        'duplicate_count': cat.duplicate_count,
                        'total_solutions': cat.total_solutions,
                        'duplicate_rate': cat.duplicate_count / cat.total_solutions if cat.total_solutions > 0 else 0
                    }
                    for cat in category_duplicates
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get category analysis: {e}")
            return {}
    
    def get_pending_duplicates_for_review(self, session: Session, limit: int = 50) -> List[Dict[str, Any]]:
        """Get duplicates pending admin review, ordered by similarity score."""
        try:
            pending_duplicates = session.query(
                SolutionDuplicate,
                Solution.header.label('duplicate_header'),
                Solution.solution.label('duplicate_solution'),
                Thread.topic_id.label('duplicate_topic_id'),
                Thread.actual_date.label('duplicate_date')
            ).join(
                Solution, SolutionDuplicate.solution_id == Solution.id
            ).join(
                Thread, Solution.thread_id == Thread.id
            ).filter(
                SolutionDuplicate.status == 'pending_review'
            ).order_by(
                desc(SolutionDuplicate.similarity_score)
            ).limit(limit).all()
            
            result = []
            for dup in pending_duplicates:
                # Get original solution details
                original_solution = session.query(Solution, Thread).join(
                    Thread, Solution.thread_id == Thread.id
                ).filter(Solution.id == dup.SolutionDuplicate.original_solution_id).first()
                
                if original_solution:
                    result.append({
                        'duplicate_id': dup.SolutionDuplicate.id,
                        'similarity_score': float(dup.SolutionDuplicate.similarity_score),
                        'created_at': dup.SolutionDuplicate.created_at.isoformat(),
                        'duplicate': {
                            'topic_id': dup.duplicate_topic_id,
                            'header': dup.duplicate_header,
                            'solution': dup.duplicate_solution,
                            'date': dup.duplicate_date.isoformat() if dup.duplicate_date else None
                        },
                        'original': {
                            'topic_id': original_solution.Thread.topic_id,
                            'header': original_solution.Solution.header,
                            'solution': original_solution.Solution.solution,
                            'date': original_solution.Thread.actual_date.isoformat() if original_solution.Thread.actual_date else None
                        }
                    })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get pending duplicates: {e}")
            return []
    
    def get_solution_duplicate_chain(self, session: Session, solution_id: int) -> Dict[str, Any]:
        """Get the complete duplicate chain for a solution (original + all duplicates)."""
        try:
            # Get the solution
            solution = session.query(Solution, Thread).join(
                Thread, Solution.thread_id == Thread.id
            ).filter(Solution.id == solution_id).first()
            
            if not solution:
                return {'error': 'Solution not found'}
            
            # Get all duplicates of this solution
            duplicates = session.query(
                SolutionDuplicate,
                Solution.header.label('dup_header'),
                Solution.solution.label('dup_solution'),
                Thread.topic_id.label('dup_topic_id'),
                Thread.actual_date.label('dup_date')
            ).join(
                Solution, SolutionDuplicate.solution_id == Solution.id
            ).join(
                Thread, Solution.thread_id == Thread.id
            ).filter(
                SolutionDuplicate.original_solution_id == solution_id
            ).order_by(SolutionDuplicate.created_at).all()
            
            return {
                'original': {
                    'solution_id': solution.Solution.id,
                    'topic_id': solution.Thread.topic_id,
                    'header': solution.Solution.header,
                    'solution': solution.Solution.solution,
                    'label': solution.Solution.label,
                    'date': solution.Thread.actual_date.isoformat() if solution.Thread.actual_date else None,
                    'duplicate_count': solution.Solution.duplicate_count
                },
                'duplicates': [
                    {
                        'duplicate_id': dup.SolutionDuplicate.id,
                        'topic_id': dup.dup_topic_id,
                        'header': dup.dup_header,
                        'solution': dup.dup_solution,
                        'similarity_score': float(dup.SolutionDuplicate.similarity_score),
                        'status': dup.SolutionDuplicate.status,
                        'date': dup.dup_date.isoformat() if dup.dup_date else None,
                        'created_at': dup.SolutionDuplicate.created_at.isoformat(),
                        'reviewed_by': dup.SolutionDuplicate.reviewed_by,
                        'reviewed_at': dup.SolutionDuplicate.reviewed_at.isoformat() if dup.SolutionDuplicate.reviewed_at else None,
                        'notes': dup.SolutionDuplicate.notes
                    }
                    for dup in duplicates
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get duplicate chain: {e}")
            return {'error': str(e)}
    
    def export_frequency_report(self, session: Session, output_format: str = 'json') -> Dict[str, Any]:
        """Export comprehensive frequency analysis report."""
        try:
            report = {
                'generated_at': datetime.utcnow().isoformat(),
                'statistics': self.get_duplicate_statistics(session),
                'frequent_problems': self.get_frequent_problems(session, limit=50),
                'trends': self.get_duplicate_trends(session, days_back=30),
                'categories': self.get_problem_categories_analysis(session),
                'pending_reviews': len(self.get_pending_duplicates_for_review(session, limit=1000))
            }
            
            if output_format == 'csv':
                # Convert frequent problems to CSV-friendly format
                import io
                csv_data = io.StringIO()
                df = pd.DataFrame(report['frequent_problems'])
                df.to_csv(csv_data, index=False)
                report['csv_data'] = csv_data.getvalue()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to export frequency report: {e}")
            return {'error': str(e)}


# Global analytics service instance
analytics_service = None

def get_analytics_service() -> AnalyticsService:
    """Get global analytics service instance."""
    global analytics_service
    if analytics_service is None:
        analytics_service = AnalyticsService()
    return analytics_service