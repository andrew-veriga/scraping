import logging
import yaml
from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.services.database import get_database_service
from app.services import gemini_service
from app.models.db_models import Thread, Message, ThreadMessage, LLMCache, Solution
from app.models.pydantic_models import RawThreadList, TechnicalTopics, ThreadList


class LLMOptimizer:
    """Optimizes LLM API calls by caching responses and batching requests."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.db_service = get_database_service()
        
        # LLM optimization configuration
        self.batch_size = self.config.get('llm', {}).get('batch_size', 10)
        self.max_context_length = self.config.get('llm', {}).get('max_context_length', 8000)
        self.cache_ttl_hours = self.config.get('llm', {}).get('cache_ttl', 24)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _create_request_hash(self, request_data: Dict[str, Any]) -> str:
        """Create hash for caching LLM requests."""
        # Create a deterministic hash from request data
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()
    
    def _prepare_messages_context(self, session: Session, thread_ids: List[str], 
                                max_messages_per_thread: int = 20) -> str:
        """Prepare minimal message context for LLM instead of full CSV."""
        context_data = []
        
        for thread_id in thread_ids:
            # Get thread
            thread = session.query(Thread).filter(Thread.topic_id == thread_id).first()
            if not thread:
                continue
            
            # Get messages for this thread (limit to most relevant)
            messages = session.query(Message).join(ThreadMessage).filter(
                ThreadMessage.thread_id == thread.topic_id
            ).order_by(Message.datetime).limit(max_messages_per_thread).all()
            
            if messages:
                # Create compact representation
                thread_data = {
                    'topic_id': thread_id,
                    'actual_date': thread.actual_date.isoformat() if thread.actual_date else None,
                    'Messages': [
                        {
                            'message_id': msg.message_id,
                            'author_id': msg.author_id,
                            'content': msg.content[:200] + '...' if len(msg.content) > 200 else msg.content,  # Truncate long messages
                            'datetime': msg.datetime.isoformat() if msg.datetime else None,
                            'referenced_message_id': msg.referenced_message_id or ''
                        }
                        for msg in messages
                    ]
                }
                context_data.append(thread_data)
        
        return json.dumps(context_data, indent=2)
    
    def optimized_thread_gathering(self, session: Session, messages_df: pd.DataFrame, 
                                 save_path: str, batch_type: str = "first") -> str:
        """Optimized version of thread gathering with caching and minimal context."""
        try:
            # Check if we can use cached result
            cache_key_data = {
                'operation': 'thread_gathering',
                'batch_type': batch_type,
                'message_count': len(messages_df),
                'date_range': f"{messages_df['DateTime'].min()}_{messages_df['DateTime'].max()}"
            }
            
            cached_response = self._get_cached_response(session, cache_key_data, 'thread_gathering')
            if cached_response:
                self.logger.info("Using cached thread gathering result")
                return self._save_cached_result(cached_response, save_path, 'thread_gathering')
            
            # Use existing logic but with optimizations
            logs_csv = messages_df.to_csv(index=False)
            
            response = gemini_service.generate_content(
                contents=[
                    logs_csv,
                    gemini_service.system_prompt,
                    gemini_service.prompt_start_step_1
                ],
                config=gemini_service.config_step1,
            )
            
            threads_list_dict = [thread.model_dump() for thread in response.threads]
            
            # Cache the result
            self._cache_response(session, cache_key_data, 'thread_gathering', 
                               {'threads': threads_list_dict})
            
            # Save result
            output_filename = f'{batch_type}_group_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
            full_path = self._save_result(threads_list_dict, save_path, output_filename)
            
            self.logger.info(f"Optimized thread gathering: processed {len(messages_df)} messages")
            return full_path
            
        except Exception as e:
            self.logger.error(f"Optimized thread gathering failed: {e}")
            raise
    
    def optimized_technical_filtering(self, session: Session, thread_ids: List[str], 
                                    messages_df: pd.DataFrame, save_path: str, 
                                    batch_type: str = "first") -> str:
        """Optimized technical topic filtering using minimal context."""
        try:
            # Prepare minimal context instead of full thread data
            context_data = self._prepare_messages_context(session, thread_ids, max_messages_per_thread=10)
            
            # Check cache
            cache_key_data = {
                'operation': 'technical_filtering',
                'batch_type': batch_type,
                'thread_count': len(thread_ids),
                'context_hash': hashlib.md5(context_data.encode()).hexdigest()
            }
            
            cached_response = self._get_cached_response(session, cache_key_data, 'technical_filtering')
            if cached_response:
                self.logger.info("Using cached technical filtering result")
                return self._save_cached_result(cached_response, save_path, 'technical_filtering')
            
            # Call LLM with minimal context
            response = gemini_service.generate_content(
                contents=[
                    context_data,
                    gemini_service.system_prompt,
                    gemini_service.prompt_step_2
                ],
                config=gemini_service.config_step2,
            )
            
            # Cache result
            result_data = {'technical_topics': response.technical_topics}
            self._cache_response(session, cache_key_data, 'technical_filtering', result_data)
            
            # Get full thread data for technical threads only
            technical_threads = []
            with self.db_service.get_session() as db_session:
                for topic_id in response.technical_topics:
                    thread = db_session.query(Thread).filter(Thread.topic_id == topic_id).first()
                    if thread:
                        # Get messages for this thread
                        messages = db_session.query(Message).join(ThreadMessage).filter(
                            ThreadMessage.thread_id == thread.topic_id
                        ).order_by(Message.datetime).all()
                        
                        thread_data = {
                            'topic_id': topic_id,
                            'actual_date': thread.actual_date.isoformat() if thread.actual_date else None,
                            'answer_id': thread.answer_id,
                            'whole_thread': [msg.message_id for msg in messages],
                            'status': thread.status
                        }
                        technical_threads.append(thread_data)
            
            # Save result
            output_filename = f'{batch_type}_technical_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
            full_path = self._save_result(technical_threads, save_path, output_filename)
            
            self.logger.info(f"Optimized technical filtering: {len(response.technical_topics)} technical from {len(thread_ids)} threads")
            return full_path
            
        except Exception as e:
            self.logger.error(f"Optimized technical filtering failed: {e}")
            raise
    
    def optimized_solution_generation(self, session: Session, technical_threads: List[Dict[str, Any]], 
                                    save_path: str, batch_type: str = "first") -> str:
        """Optimized solution generation with batching and caching."""
        try:
            # Process in batches to avoid token limits
            all_solutions = []
            
            for i in range(0, len(technical_threads), self.batch_size):
                batch = technical_threads[i:i + self.batch_size]
                
                # Check cache for this batch
                cache_key_data = {
                    'operation': 'solution_generation',
                    'batch_type': batch_type,
                    'batch_index': i // self.batch_size,
                    'thread_ids': [t['topic_id'] for t in batch]
                }
                
                cached_response = self._get_cached_response(session, cache_key_data, 'solution_generation')
                if cached_response:
                    self.logger.info(f"Using cached solution generation for batch {i // self.batch_size}")
                    batch_solutions = cached_response.get('solutions', [])
                else:
                    # Generate solutions for this batch
                    batch_context = json.dumps(batch, indent=2)
                    
                    # Check context length
                    if len(batch_context) > self.max_context_length:
                        self.logger.warning(f"Batch context too long ({len(batch_context)} chars), splitting further")
                        # Split batch further if needed
                        half_batch_size = len(batch) // 2
                        first_half = batch[:half_batch_size]
                        second_half = batch[half_batch_size:]
                        
                        batch_solutions = []
                        for sub_batch in [first_half, second_half]:
                            if sub_batch:
                                sub_solutions = self._generate_solutions_for_batch(sub_batch, batch_type)
                                batch_solutions.extend(sub_solutions)
                    else:
                        batch_solutions = self._generate_solutions_for_batch(batch, batch_type)
                    
                    # Cache the result
                    self._cache_response(session, cache_key_data, 'solution_generation', 
                                       {'solutions': batch_solutions})
                
                all_solutions.extend(batch_solutions)
                self.logger.info(f"Processed batch {i // self.batch_size + 1}/{(len(technical_threads) - 1) // self.batch_size + 1}")
            
            # Save all solutions
            output_filename = f'{batch_type}_solutions_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.json'
            full_path = self._save_result(all_solutions, save_path, output_filename)
            
            self.logger.info(f"Optimized solution generation: {len(all_solutions)} solutions from {len(technical_threads)} threads")
            return full_path
            
        except Exception as e:
            self.logger.error(f"Optimized solution generation failed: {e}")
            raise
    
    def _generate_solutions_for_batch(self, batch: List[Dict[str, Any]], batch_type: str) -> List[Dict[str, Any]]:
        """Generate solutions for a single batch."""
        try:
            response_solutions = gemini_service.generate_content(
                contents=[
                    json.dumps(batch, indent=2),
                    gemini_service.system_prompt,
                    gemini_service.prompt_step_3
                ],
                config=gemini_service.solution_config,
            )
            
            solutions_list = [thread.model_dump() for thread in response_solutions.threads]
            return solutions_list
            
        except Exception as e:
            self.logger.error(f"Failed to generate solutions for batch: {e}")
            return []
    
    def _get_cached_response(self, session: Session, cache_key_data: Dict[str, Any], 
                           request_type: str) -> Optional[Dict[str, Any]]:
        """Get cached LLM response if available and not expired."""
        try:
            return self.db_service.get_cached_llm_response(session, cache_key_data, request_type)
        except Exception as e:
            self.logger.error(f"Failed to get cached response: {e}")
            return None
    
    def _cache_response(self, session: Session, cache_key_data: Dict[str, Any], 
                       request_type: str, response_data: Dict[str, Any]):
        """Cache LLM response."""
        try:
            self.db_service.cache_llm_response(session, cache_key_data, request_type, 
                                             response_data, self.cache_ttl_hours)
        except Exception as e:
            self.logger.error(f"Failed to cache response: {e}")
    
    def _save_result(self, data: List[Dict[str, Any]], save_path: str, filename: str) -> str:
        """Save result to file."""
        import os
        import json
        from app.utils.file_utils import convert_datetime_to_str
        
        full_path = os.path.join(save_path, filename)
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=2, default=convert_datetime_to_str)
        
        return full_path
    
    def _save_cached_result(self, cached_data: Dict[str, Any], save_path: str, 
                           operation_type: str) -> str:
        """Save cached result to file."""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"cached_{operation_type}_{timestamp}.json"
        
        # Extract the relevant data based on operation type
        if operation_type == 'thread_gathering':
            data = cached_data.get('threads', [])
        elif operation_type == 'technical_filtering':
            data = cached_data.get('technical_topics', [])
        elif operation_type == 'solution_generation':
            data = cached_data.get('solutions', [])
        else:
            data = cached_data
        
        return self._save_result(data, save_path, filename)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get LLM cache statistics."""
        try:
            with self.db_service.get_session() as session:
                total_cache_entries = session.query(LLMCache).count()
                
                # Cache by type
                cache_by_type = {}
                for cache_type in ['thread_gathering', 'technical_filtering', 'solution_generation', 'revision']:
                    count = session.query(LLMCache).filter(LLMCache.request_type == cache_type).count()
                    cache_by_type[cache_type] = count
                
                # Cache hit rate (would need to track misses separately)
                total_access_count = session.query(func.sum(LLMCache.access_count)).scalar() or 0
                
                return {
                    'total_cache_entries': total_cache_entries,
                    'cache_by_type': cache_by_type,
                    'total_access_count': total_access_count,
                    'cache_ttl_hours': self.cache_ttl_hours,
                    'batch_size': self.batch_size
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {e}")
            return {}
    
    def cleanup_expired_cache(self) -> int:
        """Clean up expired cache entries."""
        try:
            with self.db_service.get_session() as session:
                expired_count = self.db_service.cleanup_expired_cache(session)
                session.commit()
                self.logger.info(f"Cleaned up {expired_count} expired cache entries")
                return expired_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired cache: {e}")
            return 0
    
    def preprocess_threads_for_classification(self, session: Session, 
                                            thread_ids: List[str]) -> List[str]:
        """Preprocess threads to identify which ones need LLM classification."""
        threads_needing_classification = []
        
        for thread_id in thread_ids:
            thread = session.query(Thread).filter(Thread.topic_id == thread_id).first()
            if thread and not thread.is_processed:
                # Check if thread has enough content for classification
                message_count = session.query(ThreadMessage).filter(
                    ThreadMessage.thread_id == thread.topic_id
                ).count()
                
                if message_count >= 2:  # Minimum messages for meaningful classification
                    threads_needing_classification.append(thread_id)
        
        self.logger.info(f"Identified {len(threads_needing_classification)} threads needing classification from {len(thread_ids)} total")
        return threads_needing_classification


# Global LLM optimizer instance
llm_optimizer = None

def get_llm_optimizer() -> LLMOptimizer:
    """Get global LLM optimizer instance."""
    global llm_optimizer
    if llm_optimizer is None:
        llm_optimizer = LLMOptimizer()
    return llm_optimizer