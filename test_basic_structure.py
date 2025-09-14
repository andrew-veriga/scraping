"""
Basic database structure test - no pgvector dependencies
Run with: python test_basic_structure.py
"""

import os
import sys
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Load environment variables
load_dotenv('.env.test')

def test_basic_structure():
    """Test basic database structure without pgvector dependencies"""
    
    print("🧪 Testing basic database structure...")
    
    # Import only core models
    from app.models.db_models import Base, Message, Thread
    from app.models.pydantic_models import ThreadStatus
    
    # Connect to database
    test_db_url = os.getenv('TEST_DB_URL')
    if not test_db_url:
        print("❌ TEST_DB_URL not found in .env.test")
        return False
    
    print(f"🔗 Connecting to: {test_db_url}")
    engine = create_engine(test_db_url, echo=False)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        # Create only the core tables (Thread first, then Message due to FK)
        print("📦 Creating core tables...")
        
        # Drop tables in reverse dependency order
        Message.__table__.drop(engine, checkfirst=True)
        Thread.__table__.drop(engine, checkfirst=True)
        
        # Create tables in dependency order (Thread first, then Message)
        Thread.__table__.create(engine)
        Message.__table__.create(engine) 
        
        print("✅ Core tables created successfully")
        
        # Test basic operations
        session = SessionLocal()
        
        try:
            # Create a thread
            thread = Thread(
                topic_id="test_thread_001",
                header="Test thread for basic structure",
                actual_date=datetime.now(timezone.utc),
                status=ThreadStatus.NEW,
                is_technical=True
            )
            session.add(thread)
            
            # Create root message
            root_msg = Message(
                message_id="msg_001",
                author_id="test_user",
                content="This is a test root message",
                datetime=datetime.now(timezone.utc),
                dated_message="Test message",
                thread_id="test_thread_001",
                parent_id=None,
                depth_level=0,
                is_root_message=True,
                order_in_thread=1
            )
            session.add(root_msg)
            
            # Create reply message
            reply_msg = Message(
                message_id="msg_002",
                author_id="test_helper",
                content="This is a test reply",
                datetime=datetime.now(timezone.utc),
                dated_message="Test reply",
                thread_id="test_thread_001",
                parent_id="msg_001",
                depth_level=1,
                is_root_message=False,
                order_in_thread=2
            )
            session.add(reply_msg)
            
            # Commit changes
            session.commit()
            print("✅ Test data created successfully")
            
            # Test relationships
            saved_thread = session.get(Thread, "test_thread_001")
            assert saved_thread is not None
            assert len(saved_thread.messages) == 2
            print("✅ Thread-Message relationship works")
            
            # Test hierarchical relationship
            saved_root = session.get(Message, "msg_001")
            saved_reply = session.get(Message, "msg_002")
            
            assert saved_root.is_thread_root == True
            assert len(saved_root.child_messages) == 1
            assert saved_reply.parent_message.message_id == "msg_001"
            print("✅ Hierarchical Message relationships work")
            
            # Test utility methods
            descendants = saved_root.get_all_descendants(session)
            assert len(descendants) == 1
            assert descendants[0].message_id == "msg_002"
            print("✅ Utility methods work")
            
            path = saved_reply.get_message_path(session)
            assert len(path) == 2
            assert path[0].message_id == "msg_001"
            assert path[1].message_id == "msg_002"
            print("✅ Message path finding works")
            
            print("\\n🎉 All basic structure tests passed!")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"❌ Test failed: {e}")
            return False
        finally:
            session.close()
            
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    finally:
        engine.dispose()

def main():
    print("🚀 Basic Database Structure Test")
    print("=" * 40)
    
    success = test_basic_structure()
    
    if success:
        print("\\n✅ Basic structure is working perfectly!")
        print("🚀 Ready to run full tests with:")
        print("   python setup_test_database.py  # Set up pgvector")
        print("   python test_database_operations.py  # Full test suite")
    else:
        print("\\n❌ Basic structure test failed")
        print("💡 Check your database connection and credentials")

if __name__ == "__main__":
    main()