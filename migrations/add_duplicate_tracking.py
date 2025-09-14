"""
Database migration script to add duplicate tracking functionality.

This script adds:
1. New SolutionDuplicate table for tracking duplicate relationships
2. New columns to solution table: is_duplicate and duplicate_count
3. Indexes for performance optimization

Run this script after updating the codebase to ensure database schema matches the new models.
"""

import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Index, func
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from app.services.database import get_database_service

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_migration():
    """Run the duplicate tracking migration."""
    try:
        logger.info("Starting duplicate tracking migration...")
        
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
                
                # Step 1: Add new columns to solutions table
                logger.info("Adding new columns to solutions table...")
                add_solution_columns(connection)
                
                # Step 2: Create solution_duplicates table
                logger.info("Creating solution_duplicates table...")
                create_duplicates_table(connection)
                
                # Step 3: Create indexes
                logger.info("Creating indexes...")
                create_indexes(connection)
                
                # Step 4: Initialize existing solutions
                logger.info("Initializing existing solutions...")
                initialize_existing_data(connection)
                
                # Step 5: Record migration
                record_migration(connection)
                
                # Commit transaction
                trans.commit()
                logger.info("Migration completed successfully!")
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
        # Try to query the solution_duplicates table
        result = connection.execute(text("SELECT 1 FROM solution_duplicates LIMIT 1"))
        return True  # Table exists, migration already applied
    except SQLAlchemyError:
        return False  # Table doesn't exist, migration not applied

def add_solution_columns(connection):
    """Add new columns to the solutions table."""
    try:
        # Add is_duplicate column
        connection.execute(text("""
            ALTER TABLE solutions 
            ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE NOT NULL
        """))
        
        # Add duplicate_count column
        connection.execute(text("""
            ALTER TABLE solutions 
            ADD COLUMN IF NOT EXISTS duplicate_count INTEGER DEFAULT 0 NOT NULL
        """))
        
        # Create index on is_duplicate
        connection.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_solution_is_duplicate 
            ON solutions(is_duplicate)
        """))
        
        logger.info("Successfully added columns to solutions table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to add solution columns: {e}")
        raise

def create_duplicates_table(connection):
    """Create the solution_duplicates table."""
    try:
        # Create solution_duplicates table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS solution_duplicates (
                id SERIAL PRIMARY KEY,
                solution_id INTEGER NOT NULL REFERENCES solutions(id) ON DELETE CASCADE,
                original_solution_id INTEGER NOT NULL REFERENCES solutions(id) ON DELETE CASCADE,
                similarity_score VARCHAR(10) NOT NULL,
                status VARCHAR(20) DEFAULT 'pending_review' NOT NULL,
                reviewed_by VARCHAR(100),
                reviewed_at TIMESTAMP WITH TIME ZONE,
                notes TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            )
        """))
        
        logger.info("Successfully created solution_duplicates table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create duplicates table: {e}")
        raise

def create_indexes(connection):
    """Create indexes for the new tables and columns."""
    try:
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_duplicate_solution ON solution_duplicates(solution_id)",
            "CREATE INDEX IF NOT EXISTS idx_duplicate_original ON solution_duplicates(original_solution_id)",
            "CREATE INDEX IF NOT EXISTS idx_duplicate_status ON solution_duplicates(status)",
            "CREATE INDEX IF NOT EXISTS idx_duplicate_similarity ON solution_duplicates(similarity_score)",
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_unique_duplicate_pair ON solution_duplicates(solution_id, original_solution_id)"
        ]
        
        for index_sql in indexes:
            connection.execute(text(index_sql))
        
        logger.info("Successfully created indexes")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create indexes: {e}")
        raise

def initialize_existing_data(connection):
    """Initialize existing solutions with default values."""
    try:
        # Set all existing solutions to is_duplicate = FALSE and duplicate_count = 0
        result = connection.execute(text("""
            UPDATE solutions 
            SET is_duplicate = FALSE, duplicate_count = 0
            WHERE is_duplicate IS NULL OR duplicate_count IS NULL
        """))
        
        updated_rows = result.rowcount
        logger.info(f"Initialized {updated_rows} existing solutions")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to initialize existing data: {e}")
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
            VALUES ('add_duplicate_tracking') 
            ON CONFLICT (name) DO NOTHING
        """))
        
        logger.info("Recorded migration in migrations table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to record migration: {e}")
        raise

def rollback_migration():
    """Rollback the duplicate tracking migration."""
    try:
        logger.info("Starting duplicate tracking migration rollback...")
        
        # Get database service
        db_service = get_database_service()
        engine = db_service.engine
        
        with engine.connect() as connection:
            trans = connection.begin()
            
            try:
                # Drop indexes
                logger.info("Dropping indexes...")
                indexes_to_drop = [
                    "DROP INDEX IF EXISTS idx_duplicate_solution",
                    "DROP INDEX IF EXISTS idx_duplicate_original", 
                    "DROP INDEX IF EXISTS idx_duplicate_status",
                    "DROP INDEX IF EXISTS idx_duplicate_similarity",
                    "DROP INDEX IF EXISTS idx_unique_duplicate_pair",
                    "DROP INDEX IF EXISTS idx_solution_is_duplicate"
                ]
                
                for drop_sql in indexes_to_drop:
                    connection.execute(text(drop_sql))
                
                # Drop solution_duplicates table
                logger.info("Dropping solution_duplicates table...")
                connection.execute(text("DROP TABLE IF EXISTS solution_duplicates"))
                
                # Remove columns from solutions table
                logger.info("Removing columns from solutions table...")
                connection.execute(text("ALTER TABLE solutions DROP COLUMN IF EXISTS is_duplicate"))
                connection.execute(text("ALTER TABLE solutions DROP COLUMN IF EXISTS duplicate_count"))
                
                # Remove migration record
                connection.execute(text("DELETE FROM migrations WHERE name = 'add_duplicate_tracking'"))
                
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
            # Check if solutions table exists
            result = session.execute(text("SELECT 1 FROM solutions LIMIT 1"))
            logger.info("Database is ready for migration")
            return True
            
    except Exception as e:
        logger.error(f"Database is not ready for migration: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Duplicate tracking migration script")
    parser.add_argument("--rollback", action="store_true", help="Rollback the migration")
    parser.add_argument("--check", action="store_true", help="Check database status")
    args = parser.parse_args()
    
    if args.check:
        if check_database_ready():
            print("✅ Database is ready for migration")
            sys.exit(0)
        else:
            print("❌ Database is not ready for migration")
            sys.exit(1)
    elif args.rollback:
        if rollback_migration():
            print("✅ Migration rollback completed successfully")
            sys.exit(0)
        else:
            print("❌ Migration rollback failed")
            sys.exit(1)
    else:
        if run_migration():
            print("✅ Migration completed successfully")
            sys.exit(0)
        else:
            print("❌ Migration failed")
            sys.exit(1)