"""
Database migration script to implement complete layered processing strategy.

This migration adds comprehensive processing tracking infrastructure:
1. New tables for message processing, annotations, and pipeline configuration
2. Enhanced existing tables with processing metadata
3. Creates indexes for performance
4. Initializes default processing pipeline

This migration builds upon the duplicate tracking migration and implements the
full layered processing strategy where original messages remain unchanged
and each processing step adds metadata layers.
"""

import os
import sys
import logging
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Index, func, JSON
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from app.services.database import get_database_service
from app.models.db_models import Base

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the layered processing migration."""
    try:
        logger.info("Starting layered processing migration...")
        
        # Get database service
        db_service = get_database_service()
        engine = db_service.engine
        
        with engine.connect() as connection:
            # Start transaction
            trans = connection.begin()
            
            try:
                # Check if migration has already been run
                if check_migration_status(connection):
                    logger.info("Migration has already been applied. Skipping.")
                    return True
                
                # Step 1: Add new columns to existing tables
                logger.info("Adding new columns to existing tables...")
                enhance_existing_tables(connection)
                
                # Step 2: Create new tables
                logger.info("Creating new tables...")
                create_new_tables(connection)
                
                # Step 3: Create indexes
                logger.info("Creating indexes...")
                create_indexes(connection)
                
                # Step 4: Initialize default processing pipeline
                logger.info("Initializing default processing pipeline...")
                initialize_default_pipeline(connection)
                
                # Step 5: Record migration
                record_migration(connection)
                
                # Commit transaction
                trans.commit()
                logger.info("Layered processing migration completed successfully!")
                return True
                
            except Exception as e:
                # Rollback on error
                trans.rollback()
                logger.error(f"Migration failed: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to run migration: {e}")
        return False

def check_migration_status(connection):
    """Check if this migration has already been applied."""
    try:
        # Try to query the message_processing table
        result = connection.execute(text("SELECT 1 FROM message_processing LIMIT 1"))
        return True  # Table exists, migration already applied
    except SQLAlchemyError:
        return False  # Table doesn't exist, migration not applied

def enhance_existing_tables(connection):
    """Add new columns to existing tables."""
    try:
        # Enhance messages table
        logger.info("Enhancing messages table...")
        connection.execute(text("""
            ALTER TABLE messages 
            ADD COLUMN IF NOT EXISTS processing_status JSON DEFAULT '{}' NOT NULL
        """))
        connection.execute(text("""
            ALTER TABLE messages 
            ADD COLUMN IF NOT EXISTS last_processed_at TIMESTAMP WITH TIME ZONE
        """))
        connection.execute(text("""
            ALTER TABLE messages 
            ADD COLUMN IF NOT EXISTS processing_version VARCHAR(20)
        """))
        
        # Create index on messages processing fields
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_message_processed 
            ON messages(last_processed_at)
        """))
        
        # Enhance threads table
        logger.info("Enhancing threads table...")
        connection.execute(text("""
            ALTER TABLE threads 
            ADD COLUMN IF NOT EXISTS processing_history JSON DEFAULT '[]' NOT NULL
        """))
        connection.execute(text("""
            ALTER TABLE threads 
            ADD COLUMN IF NOT EXISTS confidence_scores JSON DEFAULT '{}' NOT NULL
        """))
        connection.execute(text("""
            ALTER TABLE threads 
            ADD COLUMN IF NOT EXISTS processing_metadata JSON DEFAULT '{}' NOT NULL
        """))
        
        # Enhance solutions table
        logger.info("Enhancing solutions table...")
        connection.execute(text("""
            ALTER TABLE solutions 
            ADD COLUMN IF NOT EXISTS extraction_metadata JSON DEFAULT '{}' NOT NULL
        """))
        connection.execute(text("""
            ALTER TABLE solutions 
            ADD COLUMN IF NOT EXISTS processing_steps JSON DEFAULT '[]' NOT NULL
        """))
        connection.execute(text("""
            ALTER TABLE solutions 
            ADD COLUMN IF NOT EXISTS source_messages JSON DEFAULT '[]' NOT NULL
        """))
        
        logger.info("Successfully enhanced existing tables")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to enhance existing tables: {e}")
        raise

def create_new_tables(connection):
    """Create new tables for layered processing."""
    try:
        # Create message_processing table
        logger.info("Creating message_processing table...")
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS message_processing (
                id SERIAL PRIMARY KEY,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                processing_step VARCHAR(50) NOT NULL,
                step_order INTEGER NOT NULL,
                result JSON,
                confidence_score VARCHAR(10),
                processing_metadata JSON,
                processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                processing_version VARCHAR(20)
            )
        """))
        
        # Create processing_pipeline table
        logger.info("Creating processing_pipeline table...")
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS processing_pipeline (
                id SERIAL PRIMARY KEY,
                pipeline_name VARCHAR(100) NOT NULL,
                step_order INTEGER NOT NULL,
                step_name VARCHAR(50) NOT NULL,
                step_config JSON,
                is_active BOOLEAN DEFAULT TRUE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        
        # Create message_annotations table
        logger.info("Creating message_annotations table...")
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS message_annotations (
                id SERIAL PRIMARY KEY,
                message_id INTEGER NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                annotation_type VARCHAR(50) NOT NULL,
                annotation_value JSON,
                confidence_score VARCHAR(10),
                annotated_by VARCHAR(50) NOT NULL,
                annotated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        
        logger.info("Successfully created new tables")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create new tables: {e}")
        raise

def create_indexes(connection):
    """Create indexes for the new tables and columns."""
    try:
        indexes = [
            # Message processing indexes
            "CREATE INDEX IF NOT EXISTS idx_message_processing ON message_processing(message_id, processing_step)",
            "CREATE INDEX IF NOT EXISTS idx_processing_step_order ON message_processing(processing_step, step_order)",
            "CREATE INDEX IF NOT EXISTS idx_processing_timestamp ON message_processing(processed_at)",
            
            # Processing pipeline indexes
            "CREATE INDEX IF NOT EXISTS idx_pipeline_order ON processing_pipeline(pipeline_name, step_order)",
            "CREATE INDEX IF NOT EXISTS idx_pipeline_active ON processing_pipeline(pipeline_name, is_active)",
            
            # Message annotations indexes
            "CREATE INDEX IF NOT EXISTS idx_message_annotations ON message_annotations(message_id, annotation_type)",
            "CREATE INDEX IF NOT EXISTS idx_annotation_type_confidence ON message_annotations(annotation_type, confidence_score)",
            "CREATE INDEX IF NOT EXISTS idx_annotation_timestamp ON message_annotations(annotated_at)",
        ]
        
        for index_sql in indexes:
            connection.execute(text(index_sql))
        
        logger.info("Successfully created indexes")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create indexes: {e}")
        raise

def initialize_default_pipeline(connection):
    """Initialize the default processing pipeline configuration."""
    try:
        # Define default processing pipeline steps
        default_steps = [
            {
                'order': 1,
                'name': 'data_loading',
                'config': {'description': 'Load and preprocess Discord messages from Excel'},
                'is_active': True
            },
            {
                'order': 2,
                'name': 'thread_grouping',
                'config': {'description': 'Group messages into conversation threads using LLM'},
                'is_active': True
            },
            {
                'order': 3,
                'name': 'technical_filtering',
                'config': {'description': 'Identify technical discussions using keyword analysis'},
                'is_active': True
            },
            {
                'order': 4,
                'name': 'solution_extraction',
                'config': {'description': 'Extract solutions from technical threads'},
                'is_active': True
            },
            {
                'order': 5,
                'name': 'duplicate_detection',
                'config': {'description': 'Detect similar solutions using RAG similarity search'},
                'is_active': True
            },
            {
                'order': 6,
                'name': 'rag_processing',
                'config': {'description': 'Generate embeddings and build knowledge base'},
                'is_active': True
            }
        ]
        
        # Insert default pipeline steps
        for step in default_steps:
            # Check if this step already exists
            existing = connection.execute(text("""
                SELECT COUNT(*) FROM processing_pipeline 
                WHERE pipeline_name = 'default' AND step_order = :order
            """), {'order': step['order']}).fetchone()[0]
            
            if existing == 0:
                connection.execute(text("""
                    INSERT INTO processing_pipeline (pipeline_name, step_order, step_name, step_config, is_active)
                    VALUES ('default', :order, :name, :config, :is_active)
                """), {
                    'order': step['order'],
                    'name': step['name'],
                    'config': json.dumps(step['config']),
                    'is_active': step['is_active']
                })
        
        logger.info("Successfully initialized default processing pipeline")
        
    except Exception as e:
        logger.error(f"Failed to initialize default pipeline: {e}")
        raise

def record_migration(connection):
    """Record that this migration has been applied."""
    try:
        # Create migrations table if it doesn't exist
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) NOT NULL UNIQUE,
                applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        
        # Record this migration
        connection.execute(text("""
            INSERT INTO migrations (name) 
            VALUES ('add_layered_processing') 
            ON CONFLICT (name) DO NOTHING
        """))
        
        logger.info("Recorded migration in migrations table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to record migration: {e}")
        raise

def rollback_migration():
    """Rollback the layered processing migration."""
    try:
        logger.info("Starting layered processing migration rollback...")
        
        # Get database service
        db_service = get_database_service()
        engine = db_service.engine
        
        with engine.connect() as connection:
            trans = connection.begin()
            
            try:
                # Drop indexes
                logger.info("Dropping indexes...")
                indexes_to_drop = [
                    "DROP INDEX IF EXISTS idx_message_processing",
                    "DROP INDEX IF EXISTS idx_processing_step_order",
                    "DROP INDEX IF EXISTS idx_processing_timestamp",
                    "DROP INDEX IF EXISTS idx_pipeline_order",
                    "DROP INDEX IF EXISTS idx_pipeline_active",
                    "DROP INDEX IF EXISTS idx_message_annotations",
                    "DROP INDEX IF EXISTS idx_annotation_type_confidence",
                    "DROP INDEX IF EXISTS idx_annotation_timestamp",
                    "DROP INDEX IF EXISTS idx_message_processed"
                ]
                
                for drop_sql in indexes_to_drop:
                    connection.execute(text(drop_sql))
                
                # Drop new tables
                logger.info("Dropping new tables...")
                connection.execute(text("DROP TABLE IF EXISTS message_annotations"))
                connection.execute(text("DROP TABLE IF EXISTS processing_pipeline"))
                connection.execute(text("DROP TABLE IF EXISTS message_processing"))
                
                # Remove columns from existing tables
                logger.info("Removing columns from existing tables...")
                # Messages table
                connection.execute(text("ALTER TABLE messages DROP COLUMN IF EXISTS processing_status"))
                connection.execute(text("ALTER TABLE messages DROP COLUMN IF EXISTS last_processed_at"))
                connection.execute(text("ALTER TABLE messages DROP COLUMN IF EXISTS processing_version"))
                
                # Threads table
                connection.execute(text("ALTER TABLE threads DROP COLUMN IF EXISTS processing_history"))
                connection.execute(text("ALTER TABLE threads DROP COLUMN IF EXISTS confidence_scores"))
                connection.execute(text("ALTER TABLE threads DROP COLUMN IF EXISTS processing_metadata"))
                
                # Solutions table
                connection.execute(text("ALTER TABLE solutions DROP COLUMN IF EXISTS extraction_metadata"))
                connection.execute(text("ALTER TABLE solutions DROP COLUMN IF EXISTS processing_steps"))
                connection.execute(text("ALTER TABLE solutions DROP COLUMN IF EXISTS source_messages"))
                
                # Remove migration record
                connection.execute(text("DELETE FROM migrations WHERE name = 'add_layered_processing'"))
                
                trans.commit()
                logger.info("Migration rollback completed successfully!")
                return True
                
            except Exception as e:
                trans.rollback()
                logger.error(f"Rollback failed: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to rollback migration: {e}")
        return False

def check_database_ready():
    """Check if the database is ready for migration."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # Check if required tables exist
            result = session.execute(text("SELECT 1 FROM messages LIMIT 1"))
            result = session.execute(text("SELECT 1 FROM threads LIMIT 1"))
            result = session.execute(text("SELECT 1 FROM solutions LIMIT 1"))
            logger.info("Database is ready for layered processing migration")
            return True
            
    except Exception as e:
        logger.error(f"Database is not ready for migration: {e}")
        return False

def init_pipeline_only():
    """Initialize only the default pipeline steps."""
    try:
        logger.info("Initializing default pipeline steps...")
        
        # Get database service
        db_service = get_database_service()
        engine = db_service.engine
        
        with engine.connect() as connection:
            trans = connection.begin()
            
            try:
                # Initialize default processing pipeline
                initialize_default_pipeline(connection)
                
                trans.commit()
                logger.info("Default pipeline initialized successfully!")
                return True
                
            except Exception as e:
                trans.rollback()
                logger.error(f"Pipeline initialization failed: {e}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        return False

def verify_migration():
    """Verify that the migration was applied successfully."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            # Check new tables exist
            session.execute(text("SELECT 1 FROM message_processing LIMIT 1"))
            session.execute(text("SELECT 1 FROM message_annotations LIMIT 1"))
            session.execute(text("SELECT 1 FROM processing_pipeline LIMIT 1"))
            
            # Check new columns exist
            result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'messages' 
                AND column_name IN ('processing_status', 'last_processed_at', 'processing_version')
            """))
            message_columns = [row[0] for row in result.fetchall()]
            
            result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'threads' 
                AND column_name IN ('processing_history', 'confidence_scores', 'processing_metadata')
            """))
            thread_columns = [row[0] for row in result.fetchall()]
            
            result = session.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'solutions' 
                AND column_name IN ('extraction_metadata', 'processing_steps', 'source_messages')
            """))
            solution_columns = [row[0] for row in result.fetchall()]
            
            # Check default pipeline exists
            result = session.execute(text("""
                SELECT COUNT(*) FROM processing_pipeline WHERE pipeline_name = 'default'
            """))
            pipeline_count = result.fetchone()[0]
            
            logger.info("Migration verification results:")
            logger.info(f"  New tables: message_processing, message_annotations, processing_pipeline ✓")
            logger.info(f"  Messages columns added: {len(message_columns)}/3 ✓")
            logger.info(f"  Threads columns added: {len(thread_columns)}/3 ✓")
            logger.info(f"  Solutions columns added: {len(solution_columns)}/3 ✓")
            logger.info(f"  Default pipeline steps: {pipeline_count} ✓")
            
            if (len(message_columns) == 3 and len(thread_columns) == 3 and 
                len(solution_columns) == 3 and pipeline_count >= 6):
                logger.info("✅ Migration verification successful!")
                return True
            else:
                logger.error("❌ Migration verification failed!")
                return False
            
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        return False

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Layered processing migration script")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    parser.add_argument("--check", action="store_true", help="Check database status")
    parser.add_argument("--verify", action="store_true", help="Verify migration was applied correctly")
    parser.add_argument("--init-pipeline", action="store_true", help="Initialize default pipeline steps (even if migration already exists)")
    args = parser.parse_args()
    
    if args.check:
        if check_database_ready():
            print("✅ Database is ready for migration")
            sys.exit(0)
        else:
            print("❌ Database is not ready for migration")
            sys.exit(1)
    elif args.verify:
        if verify_migration():
            print("✅ Migration verification successful")
            sys.exit(0)
        else:
            print("❌ Migration verification failed")
            sys.exit(1)
    elif args.rollback:
        if rollback_migration():
            print("✅ Migration rollback completed successfully")
            sys.exit(0)
        else:
            print("❌ Migration rollback failed")
            sys.exit(1)
    elif args.init_pipeline:
        if init_pipeline_only():
            print("✅ Default pipeline initialized successfully")
            sys.exit(0)
        else:
            print("❌ Pipeline initialization failed")
            sys.exit(1)
    else:
        if run_migration():
            print("✅ Migration completed successfully")
            if verify_migration():
                print("✅ Migration verification passed")
                sys.exit(0)
            else:
                print("❌ Migration verification failed")
                sys.exit(1)
        else:
            print("❌ Migration failed")
            sys.exit(1)