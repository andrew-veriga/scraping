#!/usr/bin/env python3
"""
Script to copy the production database with new optimized structure to test database.
"""

import os
import sys
import logging
import subprocess
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_database_urls():
    """Get production and test database URLs."""
    # Load production environment
    load_dotenv()
    prod_db_url = os.environ.get('PEERA_DB_URL')
    
    # Load test environment
    load_dotenv('.env.test')
    test_db_url = os.environ.get('PEERA_DB_URL_TEST')
    
    if not prod_db_url:
        raise ValueError("PEERA_DB_URL not found in environment variables")
    
    if not test_db_url:
        raise ValueError("PEERA_DB_URL_TEST not found in .env.test file")
    
    return prod_db_url, test_db_url

def create_test_database(test_db_url):
    """Create test database if it doesn't exist."""
    logger = logging.getLogger(__name__)
    
    # Extract database name from URL
    # Format: postgresql://user:pass@host:port/dbname
    db_name = test_db_url.split('/')[-1]
    base_url = test_db_url.rsplit('/', 1)[0]
    
    try:
        # Connect to postgres database to create test database
        engine = create_engine(base_url + '/postgres')
        
        with engine.connect() as conn:
            # Check if database exists
            result = conn.execute(text(
                "SELECT 1 FROM pg_database WHERE datname = :db_name"
            ), {"db_name": db_name})
            
            if result.fetchone():
                logger.info(f"Test database '{db_name}' already exists")
            else:
                # Create database
                conn.execute(text(f"CREATE DATABASE {db_name}"))
                conn.commit()
                logger.info(f"Created test database '{db_name}'")
                
    except Exception as e:
        logger.error(f"Failed to create test database: {e}")
        raise

def copy_database_structure(prod_db_url, test_db_url):
    """Copy database structure from production to test."""
    logger = logging.getLogger(__name__)
    
    try:
        # Create test database first
        create_test_database(test_db_url)
        
        # Use pg_dump and pg_restore for structure only
        logger.info("Copying database structure...")
        
        # Dump schema only (no data)
        dump_cmd = [
            'pg_dump',
            '--schema-only',
            '--no-owner',
            '--no-privileges',
            prod_db_url
        ]
        
        # Restore to test database
        restore_cmd = [
            'psql',
            test_db_url
        ]
        
        # Execute dump and restore
        dump_process = subprocess.Popen(dump_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        restore_process = subprocess.Popen(restore_cmd, stdin=dump_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        dump_process.stdout.close()
        restore_stdout, restore_stderr = restore_process.communicate()
        
        if restore_process.returncode != 0:
            logger.error(f"Failed to restore schema: {restore_stderr.decode()}")
            raise RuntimeError("Schema restore failed")
        
        logger.info("✅ Database structure copied successfully")
        
    except FileNotFoundError:
        logger.error("pg_dump or psql not found. Please install PostgreSQL client tools.")
        raise
    except Exception as e:
        logger.error(f"Failed to copy database structure: {e}")
        raise

def apply_migration_to_test(test_db_url):
    """Apply the optimization migration to test database."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Applying optimization migration to test database...")
        
        # Set test database URL for migration
        os.environ['PEERA_DB_URL'] = test_db_url
        
        # Import and run migration
        from migrations.remove_redundant_fields import upgrade
        
        engine = create_engine(test_db_url)
        with engine.connect() as connection:
            upgrade(connection)
        
        logger.info("✅ Migration applied to test database")
        
    except Exception as e:
        logger.error(f"Failed to apply migration: {e}")
        raise

def verify_test_database(test_db_url):
    """Verify test database structure."""
    logger = logging.getLogger(__name__)
    
    try:
        engine = create_engine(test_db_url)
        
        with engine.connect() as connection:
            # Check if tables exist
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            
            result = connection.execute(tables_query)
            tables = [row[0] for row in result.fetchall()]
            
            expected_tables = [
                'messages', 'threads', 'solutions', 'solution_duplicates',
                'solution_embeddings', 'solution_similarities', 'processing_batches',
                'message_processing', 'message_annotations', 'processing_pipeline',
                'llm_cache'
            ]
            
            missing_tables = set(expected_tables) - set(tables)
            if missing_tables:
                logger.warning(f"Missing tables: {missing_tables}")
            else:
                logger.info("✅ All expected tables present")
            
            # Check if redundant fields were removed
            messages_columns_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'messages' 
                AND table_schema = 'public'
            """)
            
            result = connection.execute(messages_columns_query)
            message_columns = [row[0] for row in result.fetchall()]
            
            redundant_fields = [
                'processing_status', 'last_processed_at', 'processing_version',
                'order_in_thread', 'depth_level', 'is_root_message'
            ]
            
            remaining_redundant = set(redundant_fields) & set(message_columns)
            if remaining_redundant:
                logger.warning(f"Redundant fields still present in messages: {remaining_redundant}")
            else:
                logger.info("✅ Redundant fields removed from messages table")
            
            logger.info(f"Test database verification complete. Tables: {len(tables)}")
            
    except Exception as e:
        logger.error(f"Failed to verify test database: {e}")
        raise

def main():
    """Main function to copy database to test environment."""
    logger = setup_logging()
    
    try:
        logger.info("=" * 60)
        logger.info("Copying production database to test environment")
        logger.info("=" * 60)
        
        # Get database URLs
        prod_db_url, test_db_url = get_database_urls()
        logger.info(f"Production DB: {prod_db_url.split('@')[1] if '@' in prod_db_url else 'configured'}")
        logger.info(f"Test DB: {test_db_url.split('@')[1] if '@' in test_db_url else 'configured'}")
        
        # Copy database structure
        copy_database_structure(prod_db_url, test_db_url)
        
        # Apply optimization migration
        apply_migration_to_test(test_db_url)
        
        # Verify test database
        verify_test_database(test_db_url)
        
        logger.info("=" * 60)
        logger.info("✅ Database copy completed successfully!")
        logger.info("Test database is ready with optimized structure")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database copy failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
