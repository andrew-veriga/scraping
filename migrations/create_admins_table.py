#!/usr/bin/env python3
"""
Migration script to create admins table and insert admin IDs.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import get_database_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_admins_table(connection):
    """Create the admins table."""
    try:
        # Check if the table already exists
        result = connection.execute(text("""
            SELECT 1 FROM information_schema.tables 
            WHERE table_name = 'admins'
        """))
        
        if result.fetchone():
            logger.info("Admins table already exists, skipping creation...")
            return
        
        # Create the admins table
        connection.execute(text("""
            CREATE TABLE admins (
                author_id VARCHAR(50) PRIMARY KEY,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        logger.info("Successfully created admins table")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create admins table: {e}")
        raise

def insert_admin_ids(connection):
    """Insert admin IDs into the admins table."""
    try:
        # List of admin IDs from data_loader.py
        admin_ids = [
            '862550907349893151',
            '466815633347313664', 
            '997105563123064892',
            '457962750644060170'
        ]
        
        # Insert admin IDs
        for admin_id in admin_ids:
            try:
                connection.execute(text("""
                    INSERT INTO admins (author_id) 
                    VALUES (:admin_id)
                    ON CONFLICT (author_id) DO NOTHING
                """), {"admin_id": admin_id})
                logger.info(f"Inserted admin ID: {admin_id}")
            except SQLAlchemyError as e:
                logger.warning(f"Failed to insert admin ID {admin_id}: {e}")
                continue
        
        logger.info(f"Successfully inserted {len(admin_ids)} admin IDs")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to insert admin IDs: {e}")
        raise

def rollback_migration(connection):
    """Rollback the migration by dropping the admins table."""
    try:
        connection.execute(text("DROP TABLE IF EXISTS admins"))
        logger.info("Successfully dropped admins table")
    except SQLAlchemyError as e:
        logger.error(f"Failed to drop admins table: {e}")
        raise

def run_migration():
    """Run the migration to create admins table and insert data."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            connection = session.connection()
            
            logger.info("Starting admins table migration...")
            
            # Create the table
            create_admins_table(connection)
            
            # Insert admin IDs
            insert_admin_ids(connection)
            
            # Commit the transaction
            session.commit()
            
            logger.info("✅ Admins table migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migration()
