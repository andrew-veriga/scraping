"""
Test for the new get_latest_threads_from_actual_date function
Run with: python test_get_latest_threads.py
"""

import os
import sys
from datetime import datetime, timezone
import pandas as pd

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.database import DatabaseService
from app.models.db_models import Base, Message, Thread
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.test')

def test_get_latest_threads_from_actual_date():
    """Test the get_latest_threads_from_actual_date function"""
    print("🧪 Testing get_latest_threads_from_actual_date function...")
    
    # Use test database
    test_db_url = os.getenv('TEST_DB_URL', 'postgresql://postgres:password@localhost:5432/test_llmthreads')
    engine = create_engine(test_db_url, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Create tables
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    session = SessionLocal()
    db_service = DatabaseService()
    
    try:
        # Test 1: No threads - should return empty list
        print("Test 1: No threads in database")
        lookback_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = db_service.get_latest_threads_from_actual_date(session, lookback_date)
        assert result == [], f"Expected empty list, got {result}"
        print("✅ Test 1 passed: Returns empty list when no threads")
        
        # Test 2: Create some test threads with different dates
        print("Test 2: Multiple threads with different dates")
        
        # Thread 1: Before lookback date
        thread1 = Thread(
            topic_id="thread_1",
            header="Old thread",
            actual_date=datetime(2023, 12, 15, tzinfo=timezone.utc),
            status="new"
        )
        
        # Thread 2: On lookback date
        thread2 = Thread(
            topic_id="thread_2", 
            header="Thread on lookback date",
            actual_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            status="new"
        )
        
        # Thread 3: After lookback date
        thread3 = Thread(
            topic_id="thread_3",
            header="Recent thread",
            actual_date=datetime(2024, 1, 15, tzinfo=timezone.utc),
            status="new"
        )
        
        # Thread 4: Much later
        thread4 = Thread(
            topic_id="thread_4",
            header="Latest thread",
            actual_date=datetime(2024, 2, 1, tzinfo=timezone.utc),
            status="new"
        )
        
        session.add_all([thread1, thread2, thread3, thread4])
        session.commit()
        
        # Test with lookback_date = 2024-01-01
        lookback_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = db_service.get_latest_threads_from_actual_date(session, lookback_date)
        
        assert len(result) == 3, f"Expected 3 threads, got {len(result)}"
        assert result[0].topic_id == "thread_2", f"Expected thread_2 first, got {result[0].topic_id}"
        assert result[1].topic_id == "thread_3", f"Expected thread_3 second, got {result[1].topic_id}"
        assert result[2].topic_id == "thread_4", f"Expected thread_4 third, got {result[2].topic_id}"
        print("✅ Test 2 passed: Returns threads with actual_date >= lookback_date, ordered by actual_date")
        
        # Test 3: Different lookback date
        print("Test 3: Different lookback date")
        lookback_date = datetime(2024, 1, 20, tzinfo=timezone.utc)
        result = db_service.get_latest_threads_from_actual_date(session, lookback_date)
        
        assert len(result) == 1, f"Expected 1 thread, got {len(result)}"
        assert result[0].topic_id == "thread_4", f"Expected thread_4, got {result[0].topic_id}"
        print("✅ Test 3 passed: Returns only threads after the lookback date")
        
        # Test 4: Lookback date in the future
        print("Test 4: Lookback date in the future")
        lookback_date = datetime(2024, 3, 1, tzinfo=timezone.utc)
        result = db_service.get_latest_threads_from_actual_date(session, lookback_date)
        
        assert result == [], f"Expected empty list, got {result}"
        print("✅ Test 4 passed: Returns empty list when lookback date is in the future")
        
        # Test 5: Verify ordering
        print("Test 5: Verify threads are ordered by actual_date")
        lookback_date = datetime(2023, 1, 1, tzinfo=timezone.utc)  # Include all threads
        result = db_service.get_latest_threads_from_actual_date(session, lookback_date)
        
        assert len(result) == 4, f"Expected 4 threads, got {len(result)}"
        
        # Verify they are ordered by actual_date (ascending)
        for i in range(len(result) - 1):
            assert result[i].actual_date <= result[i + 1].actual_date, f"Threads not ordered correctly: {result[i].actual_date} > {result[i + 1].actual_date}"
        
        print("✅ Test 5 passed: Threads are correctly ordered by actual_date")
        
        print("\n🎉 All tests passed! get_latest_threads_from_actual_date function is working correctly.")
        print("\n📋 Summary of what the function does:")
        print("   1. Queries the Thread table for threads with actual_date >= lookback_date")
        print("   2. Orders results by actual_date (ascending)")
        print("   3. Returns a list of Thread objects")
        print("   4. Returns empty list if no threads match the criteria")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

if __name__ == "__main__":
    test_get_latest_threads_from_actual_date()
