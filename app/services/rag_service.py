import os
import logging
import yaml
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text, func
import hashlib
import json

try:
    from google import genai
except ImportError:
    # Fallback for development/testing
    genai = None

from app.services.database import get_database_service
from app.models.db_models import Solution, SolutionEmbedding, SolutionSimilarity, SolutionDuplicate


class RAGService:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.db_service = get_database_service()
        self.gemini_client = None
        self._initialize_gemini_client()
        
        # RAG configuration
        self.similarity_threshold = self.config.get('rag', {}).get('similarity_threshold', 0.85)
        self.embedding_model = self.config.get('rag', {}).get('embedding_model', 'text-embedding-004')
        self.embedding_dimension = self.config.get('rag', {}).get('embedding_dimension', 768)
        self.max_similar_results = self.config.get('rag', {}).get('max_similar_results', 5)
        
        # Check if pgvector is available in database service
        self.pgvector_available = getattr(self.db_service, 'pgvector_available', False)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _initialize_gemini_client(self):
        """Initialize Gemini client for embeddings."""
        try:
            if genai is None:
                self.logger.warning("Google Gemini not available, embeddings will be disabled")
                self.gemini_client = None
                return
                
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            if not gemini_api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            
            self.gemini_client = genai.Client(api_key=gemini_api_key)
            self.logger.info("Gemini client initialized for RAG service")
        except Exception as e:
            self.logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for given text using Gemini."""
        try:
            if not self.gemini_client:
                self.logger.warning("Gemini client not available, cannot generate embedding")
                return None
                
            if not text or text.strip() == "":
                self.logger.warning("Empty text provided for embedding generation")
                return None
            
            # Clean and prepare text
            cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
            
            # Generate embedding using Google GenAI SDK
            result = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=[cleaned_text]
            )
            
            if result and hasattr(result, 'embeddings') and result.embeddings:
                embedding = result.embeddings[0].values
                self.logger.debug(f"Generated embedding with dimension: {len(embedding)}")
                return embedding
            else:
                self.logger.error("No embedding returned from Gemini API")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def create_solution_text(self, solution: Solution) -> str:
        """Create combined text from solution for embedding."""
        return f"Problem: {solution.header}\nSolution: {solution.solution}\nStatus: {solution.label}"
    
    def add_solution_embedding(self, session: Session, solution: Solution) -> Optional[SolutionEmbedding]:
        """Generate and store embedding for a solution."""
        try:
            # Check if embedding already exists
            existing_embedding = session.query(SolutionEmbedding).filter(
                SolutionEmbedding.solution_id == solution.id
            ).first()
            
            if existing_embedding:
                self.logger.debug(f"Embedding already exists for solution {solution.id}")
                return existing_embedding
            
            # Generate embedding
            solution_text = self.create_solution_text(solution)
            embedding = self.generate_embedding(solution_text)
            
            if not embedding:
                self.logger.error(f"Could not generate embedding for solution {solution.id}")
                return None
            
            # Create embedding record
            solution_embedding = SolutionEmbedding(
                solution_id=solution.id,
                embedding=embedding,
                embedding_model=self.embedding_model
            )
            
            session.add(solution_embedding)
            session.flush()  # Get the ID without committing
            
            self.logger.info(f"Created embedding for solution {solution.id}")
            return solution_embedding
            
        except Exception as e:
            self.logger.error(f"Failed to add solution embedding: {e}")
            return None
    
    def calculate_cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def find_similar_solutions_by_embedding(self, session: Session, query_embedding: List[float], 
                                          exclude_solution_id: Optional[int] = None,
                                          min_similarity: Optional[float] = None) -> List[Tuple[Solution, float]]:
        """Find similar solutions using vector similarity search."""
        try:
            min_similarity = min_similarity or self.similarity_threshold
            
            if self.pgvector_available:
                # Use pgvector for efficient similarity search
                return self._find_similar_with_pgvector(session, query_embedding, exclude_solution_id, min_similarity)
            else:
                # Use JSON storage with Python-based similarity calculation
                return self._find_similar_with_json(session, query_embedding, exclude_solution_id, min_similarity)
            
        except Exception as e:
            self.logger.error(f"Failed to find similar solutions: {e}")
            return []
    
    def _find_similar_with_pgvector(self, session: Session, query_embedding: List[float],
                                   exclude_solution_id: Optional[int], min_similarity: float) -> List[Tuple[Solution, float]]:
        """Find similar solutions using pgvector extension."""
        # Build query to find similar embeddings using pgvector
        query = """
        SELECT s.*, se.embedding, 
               (se.embedding <=> %s::vector) AS distance
        FROM solutions s
        JOIN solution_embeddings se ON s.id = se.solution_id
        WHERE (se.embedding <=> %s::vector) < %s
        """
        
        params = [query_embedding, query_embedding, 1 - min_similarity]
        
        if exclude_solution_id:
            query += " AND s.id != %s"
            params.append(exclude_solution_id)
        
        query += " ORDER BY (se.embedding <=> %s::vector) LIMIT %s"
        params.extend([query_embedding, self.max_similar_results])
        
        # Execute raw SQL query
        # TODO: Всегда выдает ошибку "List argument must consist only of dictionaries" 
        result = session.execute(text(query), params)
        rows = result.fetchall()
        
        similar_solutions = []
        for row in rows:
            # Reconstruct solution object
            solution = session.query(Solution).filter(Solution.id == row[0]).first()
            if solution:
                # Convert distance to similarity (distance = 1 - cosine_similarity)
                similarity = 1 - float(row[-1])
                similar_solutions.append((solution, similarity))
        
        self.logger.info(f"Found {len(similar_solutions)} similar solutions using pgvector")
        return similar_solutions
    
    def _find_similar_with_json(self, session: Session, query_embedding: List[float],
                               exclude_solution_id: Optional[int], min_similarity: float) -> List[Tuple[Solution, float]]:
        """Find similar solutions using JSON storage and Python similarity calculation."""
        # Get all solution embeddings
        query = session.query(Solution, SolutionEmbedding).join(
            SolutionEmbedding, Solution.id == SolutionEmbedding.solution_id
        )
        
        if exclude_solution_id:
            query = query.filter(Solution.id != exclude_solution_id)
        
        embeddings_data = query.all()
        
        similar_solutions = []
        for solution, embedding_record in embeddings_data:
            # Extract embedding from JSON
            stored_embedding = embedding_record.embedding
            if isinstance(stored_embedding, list):
                # Calculate cosine similarity
                similarity = self.calculate_cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= min_similarity:
                    similar_solutions.append((solution, similarity))
        
        # Sort by similarity (highest first) and limit results
        similar_solutions.sort(key=lambda x: x[1], reverse=True)
        similar_solutions = similar_solutions[:self.max_similar_results]
        
        self.logger.info(f"Found {len(similar_solutions)} similar solutions using JSON storage")
        return similar_solutions
    
    def find_similar_solutions_by_text(self, session: Session, header: str, solution_text: str,
                                     exclude_solution_id: Optional[int] = None,
                                     min_similarity: Optional[float] = None) -> List[Tuple[Solution, float]]:
        """Find similar solutions by generating embedding from text."""
        try:
            # Create query text
            query_text = f"Problem: {header}\nSolution: {solution_text}"
            
            # Generate embedding for query
            query_embedding = self.generate_embedding(query_text)
            if not query_embedding:
                self.logger.error("Could not generate embedding for query text")
                return []
            
            return self.find_similar_solutions_by_embedding(
                session, query_embedding, exclude_solution_id, min_similarity
            )
            
        except Exception as e:
            self.logger.error(f"Failed to find similar solutions by text: {e}")
            return []
    
    def check_solution_uniqueness(self, session: Session, header: str, solution_text: str,
                                 label: str, exclude_solution_id: Optional[int] = None) -> Dict[str, Any]:
        """Check if a solution is unique or has similar existing solutions."""
        try:
            similar_solutions = self.find_similar_solutions_by_text(
                session, header, solution_text, exclude_solution_id
            )
            
            if not similar_solutions:
                return {
                    'is_unique': True,
                    'similar_solutions': [],
                    'recommendation': 'add',
                    'reason': 'No similar solutions found'
                }
            
            # Analyze similar solutions
            highest_similarity = max(sim for _, sim in similar_solutions)
            
            # Check if any similar solution has the same label (problem type)
            same_label_solutions = [
                (sol, sim) for sol, sim in similar_solutions 
                if sol.label == label
            ]
            
            # Decision logic - modified to mark duplicates instead of skipping
            if highest_similarity >= 0.95:
                recommendation = 'mark_duplicate'
                reason = f'Very similar solution exists (similarity: {highest_similarity:.3f}) - marking as duplicate'
            elif highest_similarity >= self.similarity_threshold and same_label_solutions:
                recommendation = 'merge'
                reason = f'Similar solution with same label exists (similarity: {highest_similarity:.3f})'
            elif highest_similarity >= self.similarity_threshold:
                recommendation = 'mark_duplicate'
                reason = f'Similar solution exists (similarity: {highest_similarity:.3f}) - marking as duplicate for review'
            else:
                recommendation = 'add'
                reason = f'Solutions exist but not similar enough (max similarity: {highest_similarity:.3f})'
            
            return {
                'is_unique': len(similar_solutions) == 0,
                'similar_solutions': similar_solutions,
                'recommendation': recommendation,
                'reason': reason,
                'highest_similarity': highest_similarity,
                'same_label_count': len(same_label_solutions)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to check solution uniqueness: {e}")
            return {
                'is_unique': True,
                'similar_solutions': [],
                'recommendation': 'add',
                'reason': f'Error checking uniqueness: {e}'
            }
    
    def merge_solution_suggestions(self, original_solution: Solution, 
                                 similar_solutions: List[Tuple[Solution, float]]) -> Dict[str, Any]:
        """Generate suggestions for merging solutions."""
        try:
            if not similar_solutions:
                return {'should_merge': False, 'suggestions': []}
            
            merge_candidates = []
            for similar_solution, similarity in similar_solutions:
                if similarity >= 0.9 and similar_solution.label == original_solution.label:
                    merge_candidates.append({
                        'solution': similar_solution,
                        'similarity': similarity,
                        'merge_strategy': 'combine_details'
                    })
                elif similarity >= 0.85:
                    merge_candidates.append({
                        'solution': similar_solution,
                        'similarity': similarity,
                        'merge_strategy': 'update_existing'
                    })
            
            return {
                'should_merge': len(merge_candidates) > 0,
                'candidates': merge_candidates,
                'suggestions': self._generate_merge_suggestions(merge_candidates)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate merge suggestions: {e}")
            return {'should_merge': False, 'suggestions': []}
    
    def _generate_merge_suggestions(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """Generate human-readable merge suggestions."""
        suggestions = []
        
        for candidate in candidates:
            similarity = candidate['similarity']
            strategy = candidate['merge_strategy']
            
            if strategy == 'combine_details':
                suggestions.append(
                    f"Consider combining with existing solution (similarity: {similarity:.3f}) "
                    f"to create a more comprehensive answer"
                )
            elif strategy == 'update_existing':
                suggestions.append(
                    f"Consider updating existing solution (similarity: {similarity:.3f}) "
                    f"with new information instead of creating duplicate"
                )
        
        return suggestions
    
    def batch_generate_embeddings(self, session: Session, solutions: List[Solution]) -> int:
        """Generate embeddings for multiple solutions in batch."""
        success_count = 0
        
        for solution in solutions:
            try:
                embedding = self.add_solution_embedding(session, solution)
                if embedding:
                    success_count += 1
                    
                    # Commit after each embedding to avoid losing work
                    session.commit()
                    
            except Exception as e:
                self.logger.error(f"Failed to process solution {solution.id}: {e}")
                session.rollback()
                continue
        
        self.logger.info(f"Successfully generated {success_count}/{len(solutions)} embeddings")
        return success_count
    
    def create_duplicate_record(self, session: Session, solution_id: int, original_solution_id: int, 
                              similarity_score: float) -> Optional[SolutionDuplicate]:
        """Create a duplicate record linking a solution to its original."""
        try:
            # Check if duplicate record already exists
            existing_duplicate = session.query(SolutionDuplicate).filter(
                and_(
                    SolutionDuplicate.solution_id == solution_id,
                    SolutionDuplicate.original_solution_id == original_solution_id
                )
            ).first()
            
            if existing_duplicate:
                self.logger.debug(f"Duplicate record already exists for solution {solution_id} -> {original_solution_id}")
                return existing_duplicate
            
            # Create new duplicate record
            duplicate_record = SolutionDuplicate(
                solution_id=solution_id,
                original_solution_id=original_solution_id,
                similarity_score=f"{similarity_score:.6f}",
                status='pending_review'
            )
            
            session.add(duplicate_record)
            
            # Update duplicate count on original solution
            original_solution = session.query(Solution).filter(Solution.id == original_solution_id).first()
            if original_solution:
                original_solution.duplicate_count = session.query(SolutionDuplicate).filter(
                    SolutionDuplicate.original_solution_id == original_solution_id
                ).count() + 1
            
            # Mark the duplicate solution
            duplicate_solution = session.query(Solution).filter(Solution.id == solution_id).first()
            if duplicate_solution:
                duplicate_solution.is_duplicate = True
            
            session.flush()  # Get the ID without committing
            
            self.logger.info(f"Created duplicate record: solution {solution_id} -> original {original_solution_id} (similarity: {similarity_score:.3f})")
            return duplicate_record
            
        except Exception as e:
            self.logger.error(f"Failed to create duplicate record: {e}")
            return None
    
    def get_duplicate_chain(self, session: Session, solution_id: int) -> List[Tuple[Solution, float]]:
        """Get all solutions in a duplicate chain (original + all duplicates)."""
        try:
            # First, check if this solution is itself a duplicate
            duplicate_record = session.query(SolutionDuplicate).filter(
                SolutionDuplicate.solution_id == solution_id
            ).first()
            
            if duplicate_record:
                # This is a duplicate, get the original
                original_id = duplicate_record.original_solution_id
            else:
                # This might be an original, use it as the root
                original_id = solution_id
            
            # Get the original solution
            original_solution = session.query(Solution).filter(Solution.id == original_id).first()
            if not original_solution:
                return []
            
            # Get all duplicates of the original
            duplicates = session.query(Solution, SolutionDuplicate).join(
                SolutionDuplicate, Solution.id == SolutionDuplicate.solution_id
            ).filter(
                SolutionDuplicate.original_solution_id == original_id
            ).all()
            
            # Build result list
            result = [(original_solution, 1.0)]  # Original has similarity of 1.0
            for solution, duplicate_record in duplicates:
                similarity = float(duplicate_record.similarity_score)
                result.append((solution, similarity))
            
            # Sort by similarity (highest first)
            result.sort(key=lambda x: x[1], reverse=True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get duplicate chain: {e}")
            return []
    
    def rebuild_vector_index(self, session: Session):
        """Rebuild vector index for better performance."""
        try:
            if self.pgvector_available:
                # Drop existing pgvector index
                session.execute(text(
                    "DROP INDEX IF EXISTS idx_solution_embedding_vector"
                ))
                
                # Create new pgvector index with appropriate parameters
                session.execute(text(
                    "CREATE INDEX idx_solution_embedding_vector "
                    "ON solution_embeddings USING ivfflat (embedding vector_cosine_ops) "
                    "WITH (lists = 100)"
                ))
                
                self.logger.info("Pgvector index rebuilt successfully")
            else:
                # Drop existing JSON index
                session.execute(text(
                    "DROP INDEX IF EXISTS idx_solution_embedding_json"
                ))
                
                # Create new JSON index (basic index for JSON column)
                session.execute(text(
                    "CREATE INDEX idx_solution_embedding_json "
                    "ON solution_embeddings (embedding)"
                ))
                
                self.logger.info("JSON embedding index rebuilt successfully")
            
            session.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild vector index: {e}")
            session.rollback()
    
    def get_embedding_statistics(self, session: Session) -> Dict[str, Any]:
        """Get statistics about embeddings in the database."""
        try:
            total_solutions = session.query(Solution).count()
            total_embeddings = session.query(SolutionEmbedding).count()
            
            # Get average similarity between all embeddings (sample)
            avg_similarity = 0
            if self.pgvector_available:
                try:
                    sample_similarities = session.execute(text("""
                        SELECT AVG(se1.embedding <=> se2.embedding) as avg_distance
                        FROM solution_embeddings se1, solution_embeddings se2
                        WHERE se1.id != se2.id
                        LIMIT 1000
                    """)).fetchone()
                    
                    avg_similarity = 1 - float(sample_similarities[0]) if sample_similarities[0] else 0
                except Exception:
                    avg_similarity = 0
            else:
                # For JSON storage, calculate average similarity using Python
                try:
                    embeddings = session.query(SolutionEmbedding).limit(50).all()  # Sample only
                    if len(embeddings) > 1:
                        similarities = []
                        for i, emb1 in enumerate(embeddings):
                            for emb2 in embeddings[i+1:i+10]:  # Compare with next 10 to avoid too many calculations
                                if isinstance(emb1.embedding, list) and isinstance(emb2.embedding, list):
                                    sim = self.calculate_cosine_similarity(emb1.embedding, emb2.embedding)
                                    similarities.append(sim)
                        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                except Exception:
                    avg_similarity = 0
            
            return {
                'total_solutions': total_solutions,
                'solutions_with_embeddings': total_embeddings,
                'embedding_coverage': total_embeddings / total_solutions if total_solutions > 0 else 0,
                'average_similarity': avg_similarity,
                'embedding_model': self.embedding_model,
                'embedding_dimension': self.embedding_dimension,
                'storage_type': 'pgvector' if self.pgvector_available else 'json'
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get embedding statistics: {e}")
            return {}


# Global RAG service instance
rag_service = None

def get_rag_service() -> RAGService:
    """Get global RAG service instance."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service