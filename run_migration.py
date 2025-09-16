#!/usr/bin/env python3
"""
Script to run the migration to remove redundant fields from database tables.
"""

import os
import sys
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

def run_migration():
    """Run the migration to remove redundant fields."""
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Get database URL from environment
        db_url = os.environ.get('PEERA_DB_URL')
        if not db_url:
            logger.error("PEERA_DB_URL environment variable not set")
            return False
        
        # Create database engine
        engine = create_engine(db_url)
        
        # Import and run migration
        from migrations.remove_redundant_fields import upgrade, get_migration_info
        
        # Show migration info
        info = get_migration_info()
        logger.info("=" * 60)
        logger.info(f"Migration: {info['version']}")
        logger.info(f"Description: {info['description']}")
        logger.info("Changes:")
        for change in info['changes']:
            logger.info(f"  - {change}")
        logger.info(f"Tables affected: {', '.join(info['tables_affected'])}")
        logger.info(f"Data loss: {info['data_loss']}")
        logger.info(f"Rollback supported: {info['rollback_supported']}")
        logger.info("=" * 60)
        
        # Confirm before proceeding
        response = input("\nDo you want to proceed with this migration? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Migration cancelled by user")
            return False
        
        # Run migration
        with engine.connect() as connection:
            upgrade(connection)
            logger.info("✅ Migration completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
