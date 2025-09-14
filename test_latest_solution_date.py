"""
Test for the new get_latest_solution_date function
Run with: python test_latest_solution_date.py
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

def test_get_latest_solution_date():
    """Test the get_latest_solution_date function"""
    print("🧪 Testing get_latest_solution_date function...")
    
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
        # Test 1: No messages - should return None
        print("Test 1: No messages in database")
        result = db_service.get_latest_solution_date(session)
        assert result is None, f"Expected None, got {result}"
        print("✅ Test 1 passed: Returns None when no messages")
        
        # Test 2: Single message
        print("Test 2: Single message")
        # test_date = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        # message = Message(
        #     message_id="test_msg_1",
        #     content="Test message",
        #     datetime=test_date,
        #     author_id="test_user",
        #     thread_id="test_thread_1",
        #     dated_message="Test message",
        #     depth_level=0,
        #     is_root_message=True,
        #     order_in_thread=1
        # )
        # session.add(message)
        # session.commit()
        
        result = db_service.get_latest_solution_date(session)
        expected = pd.Timestamp(test_date).normalize() + pd.Timedelta(days=1)
        expected = expected.to_pydatetime()
        
        assert result is not None, "Expected a date, got None"
        assert result.date() == expected.date(), f"Expected {expected.date()}, got {result.date()}"
        print(f"✅ Test 2 passed: Single message date normalized and incremented by 1 day")
        print(f"   Original: {test_date}")
        print(f"   Result: {result}")
        
        # Test 3: Multiple messages - should return latest
        print("Test 3: Multiple messages")
        test_date2 = datetime(2024, 1, 20, 9, 15, 0, tzinfo=timezone.utc)
        message2 = Message(
            message_id="test_msg_2",
            content="Later message",
            datetime=test_date2,
            author_id="test_user",
            thread_id="test_thread_2"
        )
        session.add(message2)
        session.commit()
        
        result = db_service.get_latest_solution_date(session)
        expected = pd.Timestamp(test_date2).normalize() + pd.Timedelta(days=1)
        expected = expected.to_pydatetime()
        
        assert result is not None, "Expected a date, got None"
        assert result.date() == expected.date(), f"Expected {expected.date()}, got {result.date()}"
        print(f"✅ Test 3 passed: Latest message date used")
        print(f"   Latest message: {test_date2}")
        print(f"   Result: {result}")
        
        # Test 4: Message with time components - should normalize to midnight
        print("Test 4: Message with time components")
        test_date3 = datetime(2024, 1, 25, 23, 59, 59, tzinfo=timezone.utc)
        message3 = Message(
            message_id="test_msg_3",
            content="Late night message",
            datetime=test_date3,
            author_id="test_user",
            thread_id="test_thread_3"
        )
        session.add(message3)
        session.commit()
        
        result = db_service.get_latest_solution_date(session)
        expected = pd.Timestamp(test_date3).normalize() + pd.Timedelta(days=1)
        expected = expected.to_pydatetime()
        
        assert result is not None, "Expected a date, got None"
        assert result.date() == expected.date(), f"Expected {expected.date()}, got {result.date()}"
        assert result.hour == 0 and result.minute == 0 and result.second == 0, f"Expected normalized time (00:00:00), got {result.time()}"
        print(f"✅ Test 4 passed: Time normalized to midnight and incremented by 1 day")
        print(f"   Original: {test_date3}")
        print(f"   Result: {result}")
        
        print("\n🎉 All tests passed! get_latest_solution_date function is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()

if __name__ == "__main__":
    test_get_latest_solution_date()
