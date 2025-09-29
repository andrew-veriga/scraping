#!/usr/bin/env python3
"""
Migration script to add attachments column to messages table.
This script adds a new TEXT column 'attachments' to store attachment information from Discord messages.
"""

import logging
import sys
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Add the parent directory to the path to import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.database import get_database_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def add_attachments_column():
    """Add attachments column to messages table."""
    try:
        logger.info("Starting migration: Adding attachments column to messages table...")
        
        # Get database service
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            connection = session.connection()
            
            # Check if attachments column already exists
            logger.info("Checking if attachments column already exists...")
            result = connection.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'messages' 
                AND column_name = 'attachments'
            """)).fetchone()
            
            if result:
                logger.info("✅ Attachments column already exists in messages table")
                return True
            
            # Add attachments column to messages table
            logger.info("Adding attachments column to messages table...")
            connection.execute(text("""
                ALTER TABLE messages 
                ADD COLUMN attachments TEXT
            """))
            
            # Also add to illustrated_messages view if it exists
            logger.info("Checking if illustrated_messages view exists...")
            view_exists = connection.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.views 
                    WHERE table_name = 'illustrated_messages'
                )
            """)).fetchone()[0]
            
            if view_exists:
                logger.info("Updating illustrated_messages view to include attachments column...")
                # Drop and recreate the view with attachments column
                connection.execute(text("DROP VIEW IF EXISTS illustrated_messages"))
                
                # Recreate the view with attachments column
                connection.execute(text("""
                    CREATE OR REPLACE VIEW public.illustrated_messages AS
                    SELECT m.message_id,
                        m.content,
                        m.datetime,
                        m.referenced_message_id,
                        m.parent_id,
                        m.attachments,
                        m.thread_id,
                        a.author_id,
                        a.author_name,
                        a.author_type,
                        concat(to_char(m.datetime, 'YYYY-MM-DD HH24:MI:SS '::text), ' ',
                            CASE
                                WHEN (EXISTS ( SELECT 1
                                   FROM admins ad
                                  WHERE ad.author_id::text = m.author_id::text)) THEN concat('Admin ', a.author_name)
                                ELSE a.author_name
                            END, ': ',
                            CASE
                                WHEN (m.referenced_message_id IS NOT NULL) and (m.referenced_message_id<>'nan') THEN concat('reply to ',
                                CASE
                                    WHEN (EXISTS ( SELECT 1
                                       FROM admins ad
                                      WHERE ad.author_id::text = ref_author.author_id::text)) THEN concat('Admin ', ref_author.author_name)
                                    ELSE ref_author.author_name
                                END, ' - ')
                                ELSE ''::text
                            END, COALESCE(regexp_replace(m.content, '<@(\d+)>'::text, ('<# '::text ||
                            CASE
                                WHEN (EXISTS ( SELECT 1
                                   FROM admins ad
                                  WHERE ad.author_id::text = "substring"(m.content, '<@(\d+)>'::text))) THEN concat('Admin ', COALESCE(tagged_author.author_name, "substring"(m.content, '<@(\d+)>'::text)::character varying))
                                ELSE concat(COALESCE(tagged_author.author_name, "substring"(m.content, '<@(\d+)>'::text)::character varying))
                            END) || '>'::text, 'g'::text), '<empty>'::text)) AS illustrated_message
                    FROM messages m
                        LEFT JOIN authors a ON m.author_id::text = a.author_id::text
                        LEFT JOIN messages ref_msg ON m.referenced_message_id::text = ref_msg.message_id::text
                        LEFT JOIN authors ref_author ON ref_msg.author_id::text = ref_author.author_id::text
                        LEFT JOIN authors tagged_author ON tagged_author.author_id::text = "substring"(m.content, '<@(\d+)>'::text)
                """))
                
                logger.info("✅ Illustrated_messages view updated with attachments column")
            
            session.commit()
            logger.info("✅ Successfully added attachments column to messages table")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to add attachments column: {e}")
        if 'session' in locals():
            session.rollback()
        raise

def rollback_attachments_column():
    """Rollback: Remove attachments column from messages table."""
    try:
        logger.info("Starting rollback: Removing attachments column from messages table...")
        
        # Get database service
        db_service = get_database_service()
        
        with db_service.get_session() as session:
            connection = session.connection()
            
            # Check if attachments column exists
            result = connection.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'messages' 
                AND column_name = 'attachments'
            """)).fetchone()
            
            if not result:
                logger.info("✅ Attachments column does not exist in messages table")
                return True
            
            # Remove attachments column from messages table
            logger.info("Removing attachments column from messages table...")
            connection.execute(text("""
                ALTER TABLE messages 
                DROP COLUMN IF EXISTS attachments
            """))
            
            # Recreate illustrated_messages view without attachments column
            logger.info("Checking if illustrated_messages view exists...")
            view_exists = connection.execute(text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.views 
                    WHERE table_name = 'illustrated_messages'
                )
            """)).fetchone()[0]
            
            if view_exists:
                logger.info("Recreating illustrated_messages view without attachments column...")
                connection.execute(text("DROP VIEW IF EXISTS illustrated_messages"))
                
                # Recreate the view without attachments column (original version)
                connection.execute(text("""
                    CREATE OR REPLACE VIEW public.illustrated_messages AS
                    SELECT m.message_id,
                        m.content,
                        m.datetime,
                        m.referenced_message_id,
                        m.parent_id,
                        m.thread_id,
                        a.author_id,
                        a.author_name,
                        a.author_type,
                        concat(to_char(m.datetime, 'YYYY-MM-DD HH24:MI:SS '::text), ' ',
                            CASE
                                WHEN (EXISTS ( SELECT 1
                                   FROM admins ad
                                  WHERE ad.author_id::text = m.author_id::text)) THEN concat('Admin ', a.author_name)
                                ELSE a.author_name
                            END, ': ',
                            CASE
                                WHEN (m.referenced_message_id IS NOT NULL) and (m.referenced_message_id<>'nan') THEN concat('reply to ',
                                CASE
                                    WHEN (EXISTS ( SELECT 1
                                       FROM admins ad
                                      WHERE ad.author_id::text = ref_author.author_id::text)) THEN concat('Admin ', ref_author.author_name)
                                    ELSE ref_author.author_name
                                END, ' - ')
                                ELSE ''::text
                            END, COALESCE(regexp_replace(m.content, '<@(\d+)>'::text, ('<# '::text ||
                            CASE
                                WHEN (EXISTS ( SELECT 1
                                   FROM admins ad
                                  WHERE ad.author_id::text = "substring"(m.content, '<@(\d+)>'::text))) THEN concat('Admin ', COALESCE(tagged_author.author_name, "substring"(m.content, '<@(\d+)>'::text)::character varying))
                                ELSE concat(COALESCE(tagged_author.author_name, "substring"(m.content, '<@(\d+)>'::text)::character varying))
                            END) || '>'::text, 'g'::text), '<empty>'::text)) AS illustrated_message
                    FROM messages m
                        LEFT JOIN authors a ON m.author_id::text = a.author_id::text
                        LEFT JOIN messages ref_msg ON m.referenced_message_id::text = ref_msg.message_id::text
                        LEFT JOIN authors ref_author ON ref_msg.author_id::text = ref_author.author_id::text
                        LEFT JOIN authors tagged_author ON tagged_author.author_id::text = "substring"(m.content, '<@(\d+)>'::text)
                """))
                
                logger.info("✅ Illustrated_messages view recreated without attachments column")
            
            session.commit()
            logger.info("✅ Successfully removed attachments column from messages table")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to remove attachments column: {e}")
        if 'session' in locals():
            session.rollback()
        raise

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        rollback_attachments_column()
    else:
        add_attachments_column()
