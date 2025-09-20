#!/usr/bin/env python3
"""
Data migration script to move existing JSON files to PostgreSQL database.
Run this script after setting up the database to migrate existing data.
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add app to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import get_database_service
from app.services.rag_service import get_rag_service
from app.services.data_loader import load_and_preprocess_data
from app.utils.file_utils import load_solutions_dict, convert_datetime_to_str
from app.models.db_models import Message, Thread, ThreadMessage, Solution, ProcessingBatch
from app.models.pydantic_models import ThreadStatus, SolutionStatus


class DataMigrator:
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.logger = logging.getLogger(__name__)
        self.db_service = get_database_service()
        self.rag_service = get_rag_service()
        
        # Load configuration
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.save_path = self.config.get('SAVE_PATH', '../results')
        self.messages_file_path = self.config.get('MESSAGES_FILE_PATH', 'data/discord_messages.xlsx')
        
        # Migration statistics
        self.stats = {
            'messages_migrated': 0,
            'threads_migrated': 0,
            'solutions_migrated': 0,
            'embeddings_generated': 0,
            'errors': []
        }
    
    def migrate_all_data(self):
        """Run complete data migration."""
        try:
            self.logger.info("Starting complete data migration...")
            
            # Step 1: Migrate messages
            self.migrate_messages()
            
            # Step 2: Migrate solutions (if solutions_dict.json exists)
            self.migrate_solutions()
            
            # Step 3: Generate embeddings for solutions
            self.generate_embeddings()
            
            # Step 4: Create processing batch records (if needed)
            self.create_processing_history()
            
            # Print final statistics
            self.print_migration_stats()
            
            self.logger.info("Data migration completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Data migration failed: {e}")
            raise
    
    def migrate_messages(self):
        """Migrate Discord messages from Excel file to database."""
        try:
            self.logger.info("Migrating messages from Excel file...")
            
            # Load messages using existing data loader
            messages_df = load_and_preprocess_data(self.messages_file_path)
            self.logger.info(f"Loaded {len(messages_df)} messages from Excel")
            
            # Convert DataFrame to list of dictionaries
            messages_data = []
            for _, row in messages_df.iterrows():
                message_data = {
                    'message_id': str(row['Message ID']),
                    'author_id': str(row['Author ID']),
                    'content': str(row['Content']),
                    'datetime': pd.to_datetime(row['DateTime']),
                    'referenced_message_id': str(row.get('Referenced Message ID', '')) if pd.notna(row.get('Referenced Message ID')) else ''
                }
                messages_data.append(message_data)
            
            # Bulk create messages
            created_count = self.db_service.bulk_create_messages(messages_data)
            self.stats['messages_migrated'] = created_count
            
            self.logger.info(f"Successfully migrated {created_count} messages")
            
        except Exception as e:
            self.logger.error(f"Failed to migrate messages: {e}")
            self.stats['errors'].append(f"Message migration: {e}")
            raise
    
    def migrate_solutions(self):
        """Migrate solutions from JSON files to database."""
        try:
            solutions_dict_path = os.path.join(self.save_path, 'solutions_dict.json')
            
            if not os.path.exists(solutions_dict_path):
                self.logger.info("No solutions_dict.json found, skipping solution migration")
                return
            
            self.logger.info("Migrating solutions from JSON file...")
            
            # Load existing solutions
            solutions_dict = load_solutions_dict('solutions_dict.json', self.save_path)
            self.logger.info(f"Loaded {len(solutions_dict)} solutions from JSON")
            
            with self.db_service.get_session() as session:
                for topic_id, solution_data in solutions_dict.items():
                    try:
                        # Create or get thread
                        thread = self._create_or_get_thread(session, topic_id, solution_data)
                        
                        # Create solution
                        solution = self._create_solution(session, thread, solution_data)
                        
                        # Add messages to thread
                        self._add_messages_to_thread(session, thread, solution_data.get('whole_thread', []))
                        
                        session.flush()  # Flush to get IDs
                        self.stats['threads_migrated'] += 1
                        self.stats['solutions_migrated'] += 1
                        
                    except Exception as e:
                        self.logger.error(f"Failed to migrate solution {topic_id}: {e}")
                        self.stats['errors'].append(f"solution {topic_id}: {e}")
                        session.rollback()
                        continue
                
                session.commit()
                self.logger.info(f"Successfully migrated {self.stats['solutions_migrated']} solutions")
                
        except Exception as e:
            self.logger.error(f"Failed to migrate solutions: {e}")
            self.stats['errors'].append(f"solution migration: {e}")
            raise
    
    def _create_or_get_thread(self, session, topic_id: str, solution_data: Dict[str, Any]) -> Thread:
        """Create or get existing thread."""
        # Check if thread already exists
        existing_thread = self.db_service.get_thread_by_topic_id(session, topic_id)
        if existing_thread:
            return existing_thread
        
        # Create new thread
        actual_date = solution_data.get('actual_date')
        if isinstance(actual_date, str):
            actual_date = pd.to_datetime(actual_date)
        
        thread_data = {
            'topic_id': topic_id,
            'header': solution_data.get('header'),
            'actual_date': actual_date,
            'answer_id': solution_data.get('answer_id'),
            'label': solution_data.get('label'),
            'solution': solution_data.get('solution'),
            'status': ThreadStatus.PERSISTED,  # These are from existing data
            'is_technical': True,  # Solutions dict only contains technical threads
            'is_processed': True   # These have been processed
        }
        
        return self.db_service.create_thread(session, thread_data)
    
    def _create_solution(self, session, thread: Thread, solution_data: Dict[str, Any]) -> Solution:
        """Create solution for thread."""
        solution = Solution(
            thread_id=thread.topic_id,
            header=solution_data.get('header', ''),
            solution=solution_data.get('solution', ''),
            label=solution_data.get('label', SolutionStatus.UNRESOLVED),
            confidence_score=None,  # Not available in old data
            version=1
        )
        
        session.add(solution)
        return solution
    
    def _add_messages_to_thread(self, session, thread: Thread, message_ids: List[str]):
        """Add messages to thread."""
        if not message_ids:
            return
        
        for order, message_id in enumerate(message_ids):
            message = self.db_service.get_message_by_message_id(str(message_id), session)
            if message:
                try:
                    thread_message = ThreadMessage(
                        thread_id=thread.topic_id,
                        message_id=message.id,
                        order_in_thread=order
                    )
                    session.add(thread_message)
                except Exception as e:
                    # Message might already be in thread
                    self.logger.debug(f"Could not add message {message_id} to thread: {e}")
    
    def generate_embeddings(self):
        """Generate embeddings for migrated solutions."""
        try:
            self.logger.info("Generating embeddings for solutions...")
            
            with self.db_service.get_session() as session:
                # Get all solutions without embeddings
                solutions = session.query(Solution).filter(
                    ~Solution.embeddings.any()
                ).all()
                
                if not solutions:
                    self.logger.info("No solutions need embeddings")
                    return
                
                self.logger.info(f"Generating embeddings for {len(solutions)} solutions...")
                
                success_count = self.rag_service.batch_generate_embeddings(session, solutions)
                self.stats['embeddings_generated'] = success_count
                
                self.logger.info(f"Generated {success_count}/{len(solutions)} embeddings")
                
        except Exception as e:
            self.logger.error(f"Failed to generate embeddings: {e}")
            self.stats['errors'].append(f"Embedding generation: {e}")
    
    def create_processing_history(self):
        """Create processing batch records for existing data."""
        try:
            self.logger.info("Creating processing history records...")
            
            with self.db_service.get_session() as session:
                # Check if we already have processing history
                existing_batches = session.query(ProcessingBatch).count()
                if existing_batches > 0:
                    self.logger.info("Processing history already exists")
                    return
                
                # Get date range of processed data
                first_message = session.query(Message).order_by(Message.datetime).first()
                last_message = session.query(Message).order_by(Message.datetime.desc()).first()
                
                if not first_message or not last_message:
                    self.logger.info("No messages found for processing history")
                    return
                
                # Create a migration batch record
                batch_data = {
                    'batch_type': 'migration',
                    'start_date': first_message.datetime,
                    'end_date': last_message.datetime,
                    'messages_processed': self.stats['messages_migrated'],
                    'threads_created': self.stats['threads_migrated'],
                    'threads_modified': 0,
                    'technical_threads': self.stats['solutions_migrated'],
                    'solutions_added': self.stats['solutions_migrated']
                }
                
                batch = self.db_service.create_processing_batch(session, batch_data)
                self.db_service.complete_processing_batch(session, batch.id, {})
                
                session.commit()
                self.logger.info("Created processing history record")
                
        except Exception as e:
            self.logger.error(f"Failed to create processing history: {e}")
            self.stats['errors'].append(f"Processing history: {e}")
    
    def print_migration_stats(self):
        """Print migration statistics."""
        self.logger.info("\n" + "="*50)
        self.logger.info("MIGRATION STATISTICS")
        self.logger.info("="*50)
        self.logger.info(f"Messages migrated: {self.stats['messages_migrated']}")
        self.logger.info(f"Threads migrated: {self.stats['threads_migrated']}")
        self.logger.info(f"Solutions migrated: {self.stats['solutions_migrated']}")
        self.logger.info(f"Embeddings generated: {self.stats['embeddings_generated']}")
        
        if self.stats['errors']:
            self.logger.info(f"\nErrors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][:5]:  # Show first 5 errors
                self.logger.info(f"  - {error}")
            if len(self.stats['errors']) > 5:
                self.logger.info(f"  ... and {len(self.stats['errors']) - 5} more errors")
        else:
            self.logger.info("\nNo errors encountered!")
        
        self.logger.info("="*50)
    
    def verify_migration(self) -> bool:
        """Verify that migration was successful."""
        try:
            stats = self.db_service.get_database_stats()
            
            self.logger.info("\nDatabase verification:")
            self.logger.info(f"Messages in database: {stats['messages']}")
            self.logger.info(f"Threads in database: {stats['threads']}")
            self.logger.info(f"Solutions in database: {stats['solutions']}")
            self.logger.info(f"Technical threads: {stats['technical_threads']}")
            
            # Basic validation
            if stats['messages'] == 0:
                self.logger.error("No messages found in database!")
                return False
            
            if stats['solutions'] > 0 and stats['technical_threads'] == 0:
                self.logger.error("Solutions exist but no technical threads marked!")
                return False
            
            self.logger.info("Database verification passed!")
            return True
            
        except Exception as e:
            self.logger.error(f"Database verification failed: {e}")
            return False


def main():
    """Main migration function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('migration.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Check if migration should proceed
        response = input("This will migrate existing data to PostgreSQL. Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("Migration cancelled by user")
            return
        
        # Run migration
        migrator = DataMigrator()
        migrator.migrate_all_data()
        
        # Verify migration
        if migrator.verify_migration():
            logger.info("Migration completed successfully!")
        else:
            logger.error("Migration verification failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()