#!/usr/bin/env python3
"""
Migration script to create illustrated_messages view that replicates the illustrated_message function logic.
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

def create_illustrated_messages_view(connection):
    """Create the illustrated_messages view."""
    try:
        # Check if the view already exists
        result = connection.execute(text("""
            SELECT 1 FROM information_schema.views 
            WHERE table_name = 'illustrated_messages'
        """))
        
        if result.fetchone():
            logger.info("Illustrated messages view already exists, dropping and recreating...")
            connection.execute(text("DROP VIEW IF EXISTS illustrated_messages"))
        
        # Create the illustrated_messages view
        connection.execute(text("""
            CREATE VIEW illustrated_messages AS
            SELECT 
                m.message_id,
                m.content,
                m.datetime,
                m.referenced_message_id,
                m.parent_id,
                m.thread_id,
                a.author_id,
                a.author_name,
                a.author_type,
                -- Create the illustrated_message field with the same logic as the Python function
                CONCAT(
                    TO_CHAR(m.datetime, 'YYYY-MM-DD HH24:MI:SS TZ'),
                    ' ',
                    CASE 
                        WHEN EXISTS (SELECT 1 FROM admins ad WHERE ad.author_id = m.author_id) 
                        THEN CONCAT('Admin ', a.author_name)
                        ELSE CONCAT('User ', a.author_name)
                    END,
                    ': ',
                    CASE 
                        WHEN m.referenced_message_id IS NOT NULL THEN
                            CONCAT(
                                'reply to ',
                                CASE 
                                    WHEN EXISTS (SELECT 1 FROM admins ad WHERE ad.author_id = ref_author.author_id)
                                    THEN CONCAT('Admin ', ref_author.author_name)
                                    ELSE CONCAT('User ', ref_author.author_name)
                                END,
                                ' - '
                            )
                        ELSE ''
                    END,
                    COALESCE(m.content, '<empty>')
                ) AS illustrated_message
            FROM messages m
            LEFT JOIN authors a ON m.author_id = a.author_id
            LEFT JOIN messages ref_msg ON m.referenced_message_id = ref_msg.message_id
            LEFT JOIN authors ref_author ON ref_msg.author_id = ref_author.author_id
        """))
        
        logger.info("Successfully created illustrated_messages view")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create illustrated_messages view: {e}")
        raise

def create_illustrated_messages_view_with_regex(connection):
    """Create the illustrated_messages view with regex support for tagged users."""
    try:
        # Check if the view already exists
        result = connection.execute(text("""
            SELECT 1 FROM information_schema.views 
            WHERE table_name = 'illustrated_messages'
        """))
        
        if result.fetchone():
            logger.info("Illustrated messages view already exists, dropping and recreating...")
            connection.execute(text("DROP VIEW IF EXISTS illustrated_messages"))
        
        # Create the illustrated_messages view with regex support
        connection.execute(text("""
            CREATE VIEW illustrated_messages AS
            SELECT 
                m.message_id,
                m.content,
                m.datetime,
                m.referenced_message_id,
                m.parent_id,
                m.thread_id,
                a.author_id,
                a.author_name,
                a.author_type,
                -- Create the illustrated_message field with the same logic as the Python function
                CONCAT(
                    TO_CHAR(m.datetime, 'YYYY-MM-DD HH24:MI:SS TZ'),
                    ' ',
                    CASE 
                        WHEN EXISTS (SELECT 1 FROM admins ad WHERE ad.author_id = m.author_id) 
                        THEN CONCAT('Admin ', a.author_name)
                        ELSE CONCAT('User ', a.author_name)
                    END,
                    ': ',
                    CASE 
                        WHEN m.referenced_message_id IS NOT NULL THEN
                            CONCAT(
                                'reply to ',
                                CASE 
                                    WHEN EXISTS (SELECT 1 FROM admins ad WHERE ad.author_id = ref_author.author_id)
                                    THEN CONCAT('Admin ', ref_author.author_name)
                                    ELSE CONCAT('User ', ref_author.author_name)
                                END,
                                ' - '
                            )
                        ELSE ''
                    END,
                    -- Handle tagged users with regex (PostgreSQL regexp_replace)
                    COALESCE(
                        REGEXP_REPLACE(
                            m.content, 
                            '<@(\\d+)>', 
                            '<tagged ' || 
                            CASE 
                                WHEN EXISTS (SELECT 1 FROM admins ad WHERE ad.author_id = SUBSTRING(m.content FROM '<@(\\d+)>'))
                                THEN CONCAT('Admin ', COALESCE(tagged_author.author_name, SUBSTRING(m.content FROM '<@(\\d+)>')))
                                ELSE CONCAT('User ', COALESCE(tagged_author.author_name, SUBSTRING(m.content FROM '<@(\\d+)>')))
                            END || 
                            '>', 
                            'g'
                        ), 
                        '<empty>'
                    )
                ) AS illustrated_message
            FROM messages m
            LEFT JOIN authors a ON m.author_id = a.author_id
            LEFT JOIN messages ref_msg ON m.referenced_message_id = ref_msg.message_id
            LEFT JOIN authors ref_author ON ref_msg.author_id = ref_author.author_id
            LEFT JOIN authors tagged_author ON tagged_author.author_id = SUBSTRING(m.content FROM '<@(\\d+)>')
        """))
        
        logger.info("Successfully created illustrated_messages view with regex support")
        
    except SQLAlchemyError as e:
        logger.error(f"Failed to create illustrated_messages view with regex: {e}")
        raise

def rollback_migration(connection):
    """Rollback the migration by dropping the view."""
    try:
        connection.execute(text("DROP VIEW IF EXISTS illustrated_messages"))
        logger.info("Successfully dropped illustrated_messages view")
    except SQLAlchemyError as e:
        logger.error(f"Failed to drop illustrated_messages view: {e}")
        raise

def run_migration():
    """Run the migration to create illustrated_messages view."""
    try:
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            connection = session.connection()
            
            logger.info("Starting illustrated_messages view migration...")
            
            # Create the view (try with regex first, fallback to simple version)
            try:
                create_illustrated_messages_view_with_regex(connection)
            except SQLAlchemyError as e:
                logger.warning(f"Regex version failed, trying simple version: {e}")
                create_illustrated_messages_view(connection)
            
            # Commit the transaction
            session.commit()
            
            logger.info("✅ Illustrated messages view migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise

if __name__ == "__main__":
    run_migration()
