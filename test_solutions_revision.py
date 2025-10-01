#!/usr/bin/env python3
"""
Test runner for process_solutions_revision_and_update function.
This script tests the solutions revision process with specific files.
"""

import os
import sys
import logging
import pandas as pd
from datetime import datetime, timezone
import json

# Add the app directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.processing import process_solutions_update
from app.utils.file_utils import load_solutions_dict, convert_str_to_Timestamp
from app.services.database import get_database_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config():
    """Load configuration from config file."""
    import yaml
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_test_batch_record():
    """Create a test batch record in the database."""
    db_service = get_database_service()
    with db_service.get_session() as session:
        batch_data = {
            'batch_type': 'test_batch',
            'start_date': datetime.now(timezone.utc),
            'end_date': datetime.now(timezone.utc)
        }
        processing_batch = db_service.create_processing_batch(session, batch_data)
        session.commit()
        return processing_batch.id

def test_solutions_revision():
    """Test the solutions revision process."""
    
    # File paths
    next_solutions_filename = "./results/next_2025-07-21-2025-07-22_solutions.json"
    next_technical_filename = "./results/next_2025-07-21-2025-07-22_technical.json"
    solutions_dict_file = "results/solutions_dict.json"
    
    # Check if files exist
    if not os.path.exists(next_solutions_filename):
        logging.error(f"Solutions file not found: {next_solutions_filename}")
        return False
        
    if not os.path.exists(next_technical_filename):
        logging.error(f"Technical file not found: {next_technical_filename}")
        return False
        
    if not os.path.exists(solutions_dict_file):
        logging.error(f"Solutions dict file not found: {solutions_dict_file}")
        return False
    
    try:
        # Load configuration
        config = load_config()
        logging.info("✅ Configuration loaded")
        
        # Load solutions dictionary
        solutions_dict = load_solutions_dict(config)
        logging.info(f"✅ Loaded solutions dictionary with {len(solutions_dict)} solutions")
        
        # Create test batch record
        batch_id = create_test_batch_record()
        logging.info(f"✅ Created test batch record with ID: {batch_id}")
        
        # Set lookback date (you may need to adjust this based on your data)
        lookback_date = pd.Timestamp('2025-07-19', tz='UTC')
        logging.info(f"✅ Set lookback date: {lookback_date}")
        
        # Log initial state
        logging.info(f"📊 Initial solutions count: {len(solutions_dict)}")
        
        # Run the solutions revision process
        logging.info("🚀 Starting solutions revision process...")
        process_solutions_update(
            next_solutions_filename=next_solutions_filename,
            next_technical_filename=next_technical_filename,
            solutions_dict=solutions_dict,
            lookback_date=lookback_date,
            config=config,
            batch_id=batch_id
        )
        
        # Log final state
        logging.info(f"📊 Final solutions count: {len(solutions_dict)}")
        logging.info("✅ Solutions revision process completed successfully!")
        
        return True
        
    except Exception as e:
        logging.error(f"❌ Error during solutions revision process: {e}", exc_info=True)
        return False

def main():
    """Main function to run the test."""
    logging.info("🧪 Starting solutions revision test...")
    
    success = test_solutions_revision()
    
    if success:
        logging.info("🎉 Test completed successfully!")
        sys.exit(0)
    else:
        logging.error("💥 Test failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
