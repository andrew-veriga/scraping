"""
Database migration script to add authors table.

This script adds:
1. New authors table for storing Discord author information
2. Indexes for performance optimization

Run this script to ensure database schema matches the Author model in db_models.py.
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
    """Run the authors table migration."""
    try:
        logger.info("Starting authors table migration...")
        
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
                
                # Step 1: Create authors table
                logger.info("Creating authors table...")
                create_authors_table(connection)
                
                # Step 2: Add foreign key constraint to messages table
                logger.info("Adding foreign key constraint to messages table...")
                add_foreign_key_constraint(connection)
                
                # Step 3: Create indexes
                logger.info("Creating indexes...")
                create_indexes(connection)
                
                # Step 4: Record migration
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
        # Try to query the authors table
        result = connection.execute(text("SELECT 1 FROM authors LIMIT 1"))
        return True  # Table exists, migration already applied
    except SQLAlchemyError:
        return False  # Table doesn't exist, migration not applied

def create_authors_table(connection):
    """Create the authors table."""
    try:
        # Create authors table
        connection.execute(text("""
            CREATE TABLE IF NOT EXISTS authors (
                author_id VARCHAR(50) PRIMARY KEY,
                author_name VARCHAR(50) NOT NULL,
                author_type VARCHAR(50) NOT NULL
            )
        """))
        
        logger.info("Successfully created authors table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create authors table: {e}")
        raise

def add_foreign_key_constraint(connection):
    """Add foreign key constraint to messages.author_id."""
    try:
        # Check if the constraint already exists
        result = connection.execute(text("""
            SELECT 1 FROM information_schema.table_constraints 
            WHERE constraint_name = 'fk_messages_author_id' 
            AND table_name = 'messages'
        """))
        
        if result.fetchone():
            logger.info("Foreign key constraint already exists, skipping...")
            return
        
        # Add foreign key constraint
        connection.execute(text("""
            ALTER TABLE messages 
            ADD CONSTRAINT fk_messages_author_id 
            FOREIGN KEY (author_id) REFERENCES authors(author_id) ON DELETE CASCADE
        """))
        
        logger.info("Successfully added foreign key constraint to messages table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to add foreign key constraint: {e}")
        raise

def create_indexes(connection):
    """Create indexes for the authors table."""
    try:
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_author_name ON authors(author_name)",
            "CREATE INDEX IF NOT EXISTS idx_author_type ON authors(author_type)"
        ]
        
        for index_sql in indexes:
            connection.execute(text(index_sql))
        
        logger.info("Successfully created indexes")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create indexes: {e}")
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
            VALUES ('add_authors_table') 
            ON CONFLICT (name) DO NOTHING
        """))
        
        logger.info("Recorded migration in migrations table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to record migration: {e}")
        raise

def rollback_migration():
    """Rollback the authors table migration."""
    try:
        logger.info("Starting authors table migration rollback...")
        
        # Get database service
        db_service = get_database_service()
        engine = db_service.engine
        
        with engine.connect() as connection:
            trans = connection.begin()
            
            try:
                # Drop indexes
                logger.info("Dropping indexes...")
                indexes_to_drop = [
                    "DROP INDEX IF EXISTS idx_author_name",
                    "DROP INDEX IF EXISTS idx_author_type"
                ]
                
                for drop_sql in indexes_to_drop:
                    connection.execute(text(drop_sql))
                
                # Drop foreign key constraint
                logger.info("Dropping foreign key constraint...")
                connection.execute(text("ALTER TABLE messages DROP CONSTRAINT IF EXISTS fk_messages_author_id"))
                
                # Drop authors table
                logger.info("Dropping authors table...")
                connection.execute(text("DROP TABLE IF EXISTS authors"))
                
                # Remove migration record
                connection.execute(text("DELETE FROM migrations WHERE name = 'add_authors_table'"))
                
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
            # Check if we can connect to the database
            result = session.execute(text("SELECT 1"))
            logger.info("Database is ready for migration")
            return True
            
    except Exception as e:
        logger.error(f"Database is not ready for migration: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Authors table migration script")
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
