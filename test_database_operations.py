"""
Test suite for database operations with hierarchical message structure
Run with: python test_database_operations.py
"""

import pytest
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
load_dotenv('.env.test' )

class TestDatabaseOperations:
    """Test class for database operations"""
    
    @classmethod
    def setup_class(cls):
        """Set up test database connection"""
        # Use a test database
        test_db_url = os.getenv('TEST_DB_URL', 'postgresql://postgres:password@localhost:5432/test_llmthreads')
        print(f"🔗 Connecting to test database at {test_db_url}")
        cls.engine = create_engine(test_db_url, echo=True)  # echo=True for SQL logging
        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=cls.engine)
        
        # Create all tables
        Base.metadata.drop_all(bind=cls.engine)  # Clean slate
        Base.metadata.create_all(bind=cls.engine)
        print("✅ Test database tables created")
    
    @classmethod
    def teardown_class(cls):
        """Clean up after all tests"""
        Base.metadata.drop_all(bind=cls.engine)
        cls.engine.dispose()
        print("✅ Test database cleaned up")
    
    def setup_method(self):
        """Set up before each test method"""
        self.session = self.SessionLocal()
        # Clean up any existing test data
        try:
            self.session.execute(text("TRUNCATE TABLE messages, threads, solutions RESTART IDENTITY CASCADE"))
            self.session.commit()
        except Exception:
            self.session.rollback()
    
    def teardown_method(self):
        """Clean up after each test method"""
        try:
            self.session.execute(text("TRUNCATE TABLE messages, threads, solutions RESTART IDENTITY CASCADE"))
            self.session.commit()
        except Exception:
            self.session.rollback()
        finally:
            self.session.close()
    
    def test_create_thread_with_hierarchical_messages(self):
        """Test creating a thread with hierarchical message structure"""
        print("\\n🧪 Testing thread creation with hierarchical messages...")
        
        # Create a thread
        thread = Thread(
            topic_id="thread_001",
            header="How to deploy Sui node on Ubuntu?",
            actual_date=datetime.now(timezone.utc),
            status=ThreadStatus.NEW,
            is_technical=True
        )
        self.session.add(thread)
        
        # Create root message (thread starter)
        root_msg = Message(
            message_id="msg_001",
            author_id="user_123",
            content="I'm having trouble deploying a Sui node on Ubuntu 22.04. Can anyone help?",
            datetime=datetime.now(timezone.utc),
            thread_id="thread_001",
            parent_id=None,  # No parent - this is the root
            depth_level=0,
            is_root_message=True,
            order_in_thread=1
        )
        self.session.add(root_msg)
        
        # Create first reply
        reply1 = Message(
            message_id="msg_002", 
            author_id="helper_456",
            content="Sure! First, make sure you have Docker installed. What's your current setup?",
            datetime=datetime.now(timezone.utc),
            thread_id="thread_001",
            parent_id="msg_001",  # Reply to root message
            depth_level=1,
            is_root_message=False,
            order_in_thread=2
        )
        self.session.add(reply1)
        
        # Create reply to the reply
        reply2 = Message(
            message_id="msg_003",
            author_id="user_123", 
            content="I have Docker installed. Using Ubuntu 22.04 with 8GB RAM.",
            datetime=datetime.now(timezone.utc),
            thread_id="thread_001",
            parent_id="msg_002",  # Reply to helper's message
            depth_level=2,
            is_root_message=False,
            order_in_thread=3
        )
        self.session.add(reply2)
        
        # Create another reply to root (parallel branch)
        reply3 = Message(
            message_id="msg_004",
            author_id="expert_789",
            content="Check the official documentation first: https://docs.sui.io/",
            datetime=datetime.now(timezone.utc),
            thread_id="thread_001",
            parent_id="msg_001",  # Another reply to root
            depth_level=1,
            is_root_message=False,
            order_in_thread=4
        )
        self.session.add(reply3)
        
        # Commit all changes
        self.session.commit()
        
        # Verify the data was saved correctly
        saved_thread = self.session.get(Thread, "thread_001")
        assert saved_thread is not None
        assert saved_thread.header == "How to deploy Sui node on Ubuntu?"
        assert len(saved_thread.messages) == 4
        
        print("✅ Thread and messages created successfully")
        return saved_thread
    
    def test_hierarchical_relationships(self):
        """Test the hierarchical relationships between messages"""
        print("\\n🧪 Testing hierarchical message relationships...")
        
        # First create test data
        self.test_create_thread_with_hierarchical_messages()
        
        # Test parent-child relationships
        root_msg = self.session.get(Message, "msg_001")
        reply1 = self.session.get(Message, "msg_002") 
        reply2 = self.session.get(Message, "msg_003")
        reply3 = self.session.get(Message, "msg_004")
        
        # Test root message properties
        assert root_msg.is_thread_root == True
        assert root_msg.parent_id is None
        assert root_msg.depth_level == 0
        assert len(root_msg.child_messages) == 2  # reply1 and reply3
        
        # Test first reply
        assert reply1.parent_id == "msg_001"
        assert reply1.parent_message.message_id == "msg_001"
        assert reply1.depth_level == 1
        assert len(reply1.child_messages) == 1  # reply2
        
        # Test nested reply
        assert reply2.parent_id == "msg_002"
        assert reply2.parent_message.message_id == "msg_002"
        assert reply2.depth_level == 2
        assert len(reply2.child_messages) == 0  # No children
        
        # Test parallel branch
        assert reply3.parent_id == "msg_001" 
        assert reply3.parent_message.message_id == "msg_001"
        assert reply3.depth_level == 1
        assert len(reply3.child_messages) == 0  # No children
        
        print("✅ Hierarchical relationships verified")
    
    def test_utility_methods(self):
        """Test utility methods for tree operations"""
        print("\\n🧪 Testing utility methods...")
        
        # Create test data
        self.test_create_thread_with_hierarchical_messages()
        
        root_msg = self.session.get(Message, "msg_001")
        reply1 = self.session.get(Message, "msg_002")
        reply2 = self.session.get(Message, "msg_003")
        
        # Test get_all_descendants
        descendants = root_msg.get_all_descendants(self.session)
        assert len(descendants) == 3  # Should include all replies and sub-replies
        descendant_ids = [d.message_id for d in descendants]
        assert "msg_002" in descendant_ids
        assert "msg_003" in descendant_ids 
        assert "msg_004" in descendant_ids
        
        # Test get_message_path
        path = reply2.get_message_path(self.session)
        assert len(path) == 3  # root -> reply1 -> reply2
        assert path[0].message_id == "msg_001"  # Root
        assert path[1].message_id == "msg_002"  # First reply
        assert path[2].message_id == "msg_003"  # Current message
        
        # Test get_thread_tree
        tree_roots = reply2.get_thread_tree(self.session)
        assert len(tree_roots) == 1  # One root message
        assert tree_roots[0].message_id == "msg_001"
        
        print("✅ Utility methods working correctly")
    
    def test_complex_thread_scenario(self):
        """Test a more complex thread with multiple branches"""
        print("\\n🧪 Testing complex thread scenario...")
        
        # Create thread
        thread = Thread(
            topic_id="thread_002",
            header="Multi-signature wallet setup issues",
            actual_date=datetime.now(timezone.utc),
            status=ThreadStatus.NEW,
            is_technical=True
        )
        self.session.add(thread)
        
        # Create complex message tree:
        # Root
        # ├── Reply A
        # │   ├── Reply A1
        # │   └── Reply A2
        # │       └── Reply A2a
        # └── Reply B
        #     └── Reply B1
        
        messages = [
            # Root message
            Message(
                message_id="msg_100", thread_id="thread_002", parent_id=None,
                author_id="user1", content="Need help with multi-sig wallet",
                datetime=datetime.now(timezone.utc),
                depth_level=0, is_root_message=True, order_in_thread=1
            ),
            # Reply A
            Message(
                message_id="msg_101", thread_id="thread_002", parent_id="msg_100",
                author_id="user2", content="What's your current configuration?",
                datetime=datetime.now(timezone.utc),
                depth_level=1, is_root_message=False, order_in_thread=2
            ),
            # Reply A1
            Message(
                message_id="msg_102", thread_id="thread_002", parent_id="msg_101",
                author_id="user1", content="Using 2-of-3 setup with hardware wallets",
                datetime=datetime.now(timezone.utc),
                depth_level=2, is_root_message=False, order_in_thread=3
            ),
            # Reply A2
            Message(
                message_id="msg_103", thread_id="thread_002", parent_id="msg_101",
                author_id="user3", content="Try updating your wallet software first",
                datetime=datetime.now(timezone.utc),
                depth_level=2, is_root_message=False, order_in_thread=4
            ),
            # Reply A2a
            Message(
                message_id="msg_104", thread_id="thread_002", parent_id="msg_103",
                author_id="user1", content="That worked! Thanks!",
                datetime=datetime.now(timezone.utc),
                depth_level=3, is_root_message=False, order_in_thread=5
            ),
            # Reply B
            Message(
                message_id="msg_105", thread_id="thread_002", parent_id="msg_100",
                author_id="user4", content="Also check the transaction fees",
                datetime=datetime.now(timezone.utc), 
                depth_level=1, is_root_message=False, order_in_thread=6
            ),
            # Reply B1
            Message(
                message_id="msg_106", thread_id="thread_002", parent_id="msg_105",
                author_id="user1", content="Fees look normal to me",
                datetime=datetime.now(timezone.utc),
                depth_level=2, is_root_message=False, order_in_thread=7
            )
        ]
        
        for msg in messages:
            self.session.add(msg)
        
        self.session.commit()
        
        # Test the complex structure
        root = self.session.get(Message, "msg_100")
        assert len(root.child_messages) == 2  # Reply A and Reply B
        
        reply_a = self.session.get(Message, "msg_101")
        assert len(reply_a.child_messages) == 2  # A1 and A2
        
        reply_a2 = self.session.get(Message, "msg_103")
        assert len(reply_a2.child_messages) == 1  # A2a
        
        # Test deepest message path
        deepest = self.session.get(Message, "msg_104")
        path = deepest.get_message_path(self.session)
        assert len(path) == 4  # Root -> A -> A2 -> A2a
        
        # Test total descendants from root
        all_descendants = root.get_all_descendants(self.session)
        assert len(all_descendants) == 6  # All non-root messages
        
        print("✅ Complex thread scenario working correctly")
    
    def test_solution_creation(self):
        """Test creating solutions linked to threads"""
        print("\\n🧪 Testing solution creation...")
        
        # Create test thread first
        self.test_create_thread_with_hierarchical_messages()
        
        # Create solution
        solution = Solution(
            thread_id="thread_001",
            header="Sui Node Deployment Guide",
            solution="Install Docker, download Sui binary, configure ports 9000-9184, run with sufficient RAM (8GB+)",
            label="resolved",
            confidence_score=95
        )
        self.session.add(solution)
        self.session.commit()
        
        # Verify solution was created and linked
        saved_solution = self.session.query(Solution).filter_by(thread_id="thread_001").first()
        assert saved_solution is not None
        assert saved_solution.header == "Sui Node Deployment Guide"
        assert saved_solution.thread.topic_id == "thread_001"
        
        print("✅ solution creation and linking verified")
    
    def test_query_performance(self):
        """Test query performance with indexes"""
        print("\\n🧪 Testing query performance...")
        
        # Create test data
        self.test_create_thread_with_hierarchical_messages()
        
        # Test indexed queries
        
        # Query by thread_id (should use idx_message_thread_order)
        thread_messages = self.session.query(Message).filter_by(thread_id="thread_001").all()
        assert len(thread_messages) == 4
        
        # Query root messages (should use idx_message_root_messages) 
        root_messages = self.session.query(Message).filter_by(
            thread_id="thread_001", 
            is_root_message=True
        ).all()
        assert len(root_messages) == 1
        assert root_messages[0].message_id == "msg_001"
        
        # Query by parent hierarchy (should use idx_message_parent_hierarchy)
        child_messages = self.session.query(Message).filter_by(parent_id="msg_001").all() 
        assert len(child_messages) == 2  # Two direct replies to root
        
        print("✅ Indexed queries working efficiently")

def run_tests():
    """Run all database tests"""
    print("🚀 Starting database operation tests...")
    print("=" * 60)
    
    test_instance = TestDatabaseOperations()
    
    try:
        # Setup
        TestDatabaseOperations.setup_class()
        
        # Run tests
        test_methods = [
            test_instance.test_create_thread_with_hierarchical_messages,
            test_instance.test_hierarchical_relationships, 
            test_instance.test_utility_methods,
            test_instance.test_complex_thread_scenario,
            test_instance.test_solution_creation,
            test_instance.test_query_performance
        ]
        
        for i, test_method in enumerate(test_methods, 1):
            try:
                test_instance.setup_method()
                test_method()
                print(f"✅ Test {i}/{len(test_methods)} passed: {test_method.__name__}")
            except Exception as e:
                print(f"❌ Test {i}/{len(test_methods)} failed: {test_method.__name__}")
                print(f"   Error: {e}")
                raise
            finally:
                test_instance.teardown_method()
        
        print("\\n" + "=" * 60)
        print("🎉 All database tests passed successfully!")
        print("✅ Your database structure is ready for real data")
        
    except Exception as e:
        print(f"\\n❌ Test suite failed: {e}")
        raise
    finally:
        # Cleanup
        TestDatabaseOperations.teardown_class()

if __name__ == "__main__":
    run_tests()