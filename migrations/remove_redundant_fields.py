"""
Migration to remove redundant fields from database tables.
This migration removes fields that are duplicated or can be computed dynamically.

Changes:
1. Remove redundant fields from Message table
2. Remove redundant fields from Thread table  
3. Remove redundant fields from Solution table
4. Remove redundant fields from ProcessingBatch table
"""

import logging
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

def upgrade(connection):
    """Remove redundant fields from tables."""
    logger = logging.getLogger(__name__)
    
    try:
        # 1. Remove redundant fields from messages table
        logger.info("Removing redundant fields from messages table...")
        
        # Remove processing metadata fields (already stored in message_processing table)
        connection.execute(text("""
            ALTER TABLE messages 
            DROP COLUMN IF EXISTS processing_status,
            DROP COLUMN IF EXISTS last_processed_at,
            DROP COLUMN IF EXISTS processing_version
        """))
        
        # Remove hierarchical fields that can be computed dynamically
        connection.execute(text("""
            ALTER TABLE messages 
            DROP COLUMN IF EXISTS order_in_thread,
            DROP COLUMN IF EXISTS depth_level,
            DROP COLUMN IF EXISTS is_root_message
        """))
        
        # 2. Remove redundant fields from threads table
        logger.info("Removing redundant fields from threads table...")
        
        connection.execute(text("""
            ALTER TABLE threads 
            DROP COLUMN IF EXISTS processing_history,
            DROP COLUMN IF EXISTS confidence_scores,
            DROP COLUMN IF EXISTS processing_metadata
        """))
        
        # 3. Remove redundant fields from solutions table
        logger.info("Removing redundant fields from solutions table...")
        
        connection.execute(text("""
            ALTER TABLE solutions 
            DROP COLUMN IF EXISTS extraction_metadata,
            DROP COLUMN IF EXISTS processing_steps,
            DROP COLUMN IF EXISTS source_messages
        """))
        
        # 4. Remove redundant fields from processing_batches table
        logger.info("Removing redundant fields from processing_batches table...")
        
        connection.execute(text("""
            ALTER TABLE processing_batches 
            DROP COLUMN IF EXISTS messages_processed,
            DROP COLUMN IF EXISTS threads_created,
            DROP COLUMN IF EXISTS threads_modified,
            DROP COLUMN IF EXISTS technical_threads,
            DROP COLUMN IF EXISTS solutions_added
        """))
        
        # 5. Remove redundant indexes that are no longer needed
        logger.info("Removing redundant indexes...")
        
        # Remove indexes for dropped columns
        connection.execute(text("""
            DROP INDEX IF EXISTS idx_message_processed;
            DROP INDEX IF EXISTS idx_message_thread_order;
            DROP INDEX IF EXISTS idx_message_parent_hierarchy;
            DROP INDEX IF EXISTS idx_message_thread_hierarchy;
            DROP INDEX IF EXISTS idx_message_root_messages;
        """))
        
        connection.commit()
        logger.info("✅ Successfully removed redundant fields from all tables")
        
    except SQLAlchemyError as e:
        connection.rollback()
        logger.error(f"❌ Failed to remove redundant fields: {e}")
        raise

def downgrade(connection):
    """Restore redundant fields (if needed for rollback)."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Restoring redundant fields...")
        
        # Note: This is a destructive migration, so downgrade is not fully supported
        # as we cannot restore the lost data. This is intentional for cleanup.
        
        logger.warning("⚠️  Downgrade not supported - this migration removes redundant data")
        logger.warning("⚠️  If rollback is needed, restore from backup before this migration")
        
    except SQLAlchemyError as e:
        logger.error(f"❌ Downgrade failed: {e}")
        raise

def get_migration_info():
    """Get information about this migration."""
    return {
        'version': 'remove_redundant_fields',
        'description': 'Remove redundant fields from database tables to optimize storage',
        'changes': [
            'Remove processing metadata from Message table (stored in MessageProcessing)',
            'Remove hierarchical fields from Message table (can be computed)',
            'Remove processing metadata from Thread table (stored in MessageProcessing)',
            'Remove processing metadata from Solution table (can be derived)',
            'Remove statistics from ProcessingBatch table (can be computed)',
            'Remove redundant indexes'
        ],
        'tables_affected': ['messages', 'threads', 'solutions', 'processing_batches'],
        'data_loss': 'Yes - redundant data is permanently removed',
        'rollback_supported': 'No - data cannot be restored'
    }
