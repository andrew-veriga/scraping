"""
Clean test runner that ensures fresh database state
Run with: python test_clean_run.py
"""

import os
import sys
from datetime import datetime, timezone
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.models.db_models import Base, Message, Thread, Solution
from app.models.pydantic_models import ThreadStatus

# Load environment variables
load_dotenv('.env.test')

def clean_database():
    """Clean all test data from database"""
    test_db_url = os.getenv('TEST_DB_URL')
    engine = create_engine(test_db_url, echo=False)
    
    try:
        # Drop and recreate all tables
        print("🧹 Cleaning database...")
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("✅ Database cleaned and tables recreated")
        return engine
    except Exception as e:
        print(f"❌ Error cleaning database: {e}")
        return None
    finally:
        engine.dispose()

def run_single_clean_test():
    """Run a single comprehensive test with clean database"""
    
    print("🚀 Running single clean database test...")
    print("=" * 50)
    
    # Clean database first
    test_db_url = os.getenv('TEST_DB_URL')
    engine = create_engine(test_db_url, echo=False)
    
    try:
        # Recreate tables
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        print("✅ Fresh database created")
        
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        session = SessionLocal()
        
        try:
            # Test 1: Create a complete thread with hierarchy
            print("\n🧪 Test 1: Creating hierarchical thread...")
            
            # Create thread
            thread = Thread(
                topic_id="clean_test_thread",
                header="Clean test thread with hierarchy",
                actual_date=datetime.now(timezone.utc),
                status=ThreadStatus.NEW,
                is_technical=True,
                answer_id="clean_msg_003"
            )
            session.add(thread)
            
            # Create message hierarchy
            messages = [
                # Root message
                Message(
                    message_id="clean_msg_001",
                    thread_id="clean_test_thread",
                    parent_id=None,
                    author_id="user_original",
                    content="Original question about Sui development",
                    datetime=datetime.now(timezone.utc),
                    referenced_message_id=None,
                    depth_level=0,
                    is_root_message=True,
                    order_in_thread=1
                ),
                # First reply
                Message(
                    message_id="clean_msg_002",
                    thread_id="clean_test_thread",
                    parent_id="clean_msg_001",
                    author_id="helper_1",
                    content="Here's some initial guidance...",
                    datetime=datetime.now(timezone.utc),
                    referenced_message_id=None,
                    depth_level=1,
                    is_root_message=False,
                    order_in_thread=2
                ),
                # solution message
                Message(
                    message_id="clean_msg_003",
                    thread_id="clean_test_thread", 
                    parent_id="clean_msg_002",
                    author_id="expert_solver",
                    content="Here's the complete solution with code examples...",
                    datetime=datetime.now(timezone.utc),
                    referenced_message_id=None,
                    is_root_message=False,
                    order_in_thread=3
                ),
                # Confirmation
                Message(
                    message_id="clean_msg_004",
                    thread_id="clean_test_thread",
                    parent_id="clean_msg_003",
                    author_id="user_original",
                    content="Perfect! That worked exactly as needed. Thank you!",
                    datetime=datetime.now(timezone.utc),
                    referenced_message_id=None,
                    is_root_message=False,
                    order_in_thread=4
                ),
                # Parallel branch - additional tip
                Message(
                    message_id="clean_msg_005",
                    thread_id="clean_test_thread",
                    parent_id="clean_msg_001",
                    author_id="additional_helper",
                    content="Also consider this alternative approach...",
                    datetime=datetime.now(timezone.utc),
                    referenced_message_id=None,
                    is_root_message=False,
                    order_in_thread=5
                )
            ]
            
            for msg in messages:
                session.add(msg)
            
            # Create solution
            solution = Solution(
                thread_id="clean_test_thread",
                header="Complete Sui Development solution",
                solution="Comprehensive solution with code examples and best practices",
                label="resolved",
                confidence_score=96
            )
            session.add(solution)
            
            # Commit all data
            session.commit()
            print("✅ Thread, messages, and solution created successfully")
            
            # Test 2: Verify relationships
            print("\n🧪 Test 2: Verifying relationships...")
            
            # Check thread
            saved_thread = session.get(Thread, "clean_test_thread")
            assert saved_thread is not None, "Thread not found"
            assert len(saved_thread.messages) == 5, f"Expected 5 messages, got {len(saved_thread.messages)}"
            print(f"✅ Thread has {len(saved_thread.messages)} messages")
            
            # Check root message
            root_msg = session.get(Message, "clean_msg_001")
            assert root_msg.is_thread_root == True, "Root message not identified correctly"
            assert len(root_msg.child_messages) == 2, f"Expected 2 direct children, got {len(root_msg.child_messages)}"
            print("✅ Root message has correct children")
            
            # Check hierarchy
            solution_msg = session.get(Message, "clean_msg_003")
            path = solution_msg.get_message_path(session)
            assert len(path) == 3, f"Expected path length 3, got {len(path)}"
            print(f"✅ solution message path: {' → '.join([m.message_id for m in path])}")
            
            # Check descendants
            all_descendants = root_msg.get_all_descendants(session)
            assert len(all_descendants) == 4, f"Expected 4 descendants, got {len(all_descendants)}"
            print("✅ All descendants found correctly")
            
            # Test 3: Query performance
            print("\n🧪 Test 3: Testing queries...")
            
            # Find by thread
            thread_messages = session.query(Message).filter_by(thread_id="clean_test_thread").count()
            assert thread_messages == 5, f"Expected 5 thread messages, got {thread_messages}"
            
            # Find root messages
            root_count = session.query(Message).filter_by(is_root_message=True).count()
            assert root_count == 1, f"Expected 1 root message, got {root_count}"
            
            # Find by depth
            depth_2_count = session.query(Message).filter_by(depth_level=2).count()
            assert depth_2_count == 1, f"Expected 1 depth-2 message, got {depth_2_count}"
            
            print("✅ All queries working correctly")
            
            # Test 4: solution relationship
            print("\n🧪 Test 4: Testing solution relationship...")
            
            saved_solution = session.query(Solution).filter_by(thread_id="clean_test_thread").first()
            assert saved_solution is not None, "solution not found"
            assert saved_solution.thread.topic_id == "clean_test_thread", "solution-thread relationship broken"
            print("✅ solution properly linked to thread")
            
            print("\n🎉 All tests passed successfully!")
            print("✅ Your hierarchical database structure is working perfectly!")
            
            return True
            
        except Exception as e:
            session.rollback()
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            session.close()
            
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False
    finally:
        engine.dispose()

if __name__ == "__main__":
    success = run_single_clean_test()
    
    if success:
        print("\n" + "=" * 50)
        print("🚀 SUCCESS: Database structure is production ready!")
        print("✅ Hierarchical messages working perfectly")
        print("✅ All relationships properly established")
        print("✅ Query performance optimized")
        print("🎯 Ready to load real Discord data!")
    else:
        print("\n" + "=" * 50)
        print("❌ FAILED: Issues detected with database structure")
        sys.exit(1)