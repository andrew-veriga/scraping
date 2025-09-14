"""
Extended test suite using realistic sample data
Run with: python test_with_samples.py
"""

import os
import sys
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.models.db_models import Base, Message, Thread, Solution
from test_data_samples import TestDataGenerator

# Load environment variables
load_dotenv('.env.test')

class TestWithSampleData:
    """Test database operations using realistic sample data"""
    
    def __init__(self):
        # Use test database
        test_db_url = os.getenv('TEST_DB_URL', 'postgresql://postgres:password@localhost:5432/test_llmthreads')
        self.engine = create_engine(test_db_url, echo=False)  # Set echo=True for SQL debugging
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Recreate tables
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(bind=self.engine)
        print("✅ Test database tables recreated")
    
    def load_sample_data(self):
        """Load all sample threads into database"""
        print("\n📦 Loading sample data...")
        
        session = self.SessionLocal()
        try:
            sample_threads = TestDataGenerator.get_all_sample_threads()
            
            for thread_data in sample_threads:
                # Add thread
                session.add(thread_data['thread'])
                
                # Add all messages
                for message in thread_data['messages']:
                    session.add(message)
                
                # Add solution
                session.add(thread_data['solution'])
            
            session.commit()
            
            # Verify data was loaded
            thread_count = session.query(Thread).count()
            message_count = session.query(Message).count()
            solution_count = session.query(Solution).count()
            
            print(f"   ✅ Loaded {thread_count} threads")
            print(f"   ✅ Loaded {message_count} messages")
            print(f"   ✅ Loaded {solution_count} solutions")
            
        except Exception as e:
            session.rollback()
            print(f"   ❌ Error loading sample data: {e}")
            raise
        finally:
            session.close()
    
    def test_thread_structure_analysis(self):
        """Analyze the structure of each thread"""
        print("\n🔍 Analyzing thread structures...")
        
        session = self.SessionLocal()
        try:
            threads = session.query(Thread).all()
            
            for thread in threads:
                print(f"\n📋 Thread: {thread.header[:50]}...")
                print(f"   Topic ID: {thread.topic_id}")
                print(f"   Total Messages: {len(thread.messages)}")
                
                # Find root message
                root_messages = [m for m in thread.messages if m.is_root_message]
                print(f"   Root Messages: {len(root_messages)}")
                
                if root_messages:
                    root = root_messages[0]
                    descendants = root.get_all_descendants(session)
                    print(f"   Thread Depth: {max([m.depth_level for m in thread.messages])}")
                    print(f"   Total Replies: {len(descendants)}")
                    
                    # Show tree structure
                    print("   Message Tree:")
                    self._print_message_tree(root, session, indent="     ")
                
        finally:
            session.close()
    
    def _print_message_tree(self, message, session, indent=""):
        """Recursively print message tree structure"""
        author_short = message.author_id[:15] + ("..." if len(message.author_id) > 15 else "")
        content_preview = message.content[:40] + ("..." if len(message.content) > 40 else "")
        
        print(f"{indent}├── {message.message_id} ({author_short}): {content_preview}")
        
        # Print children
        for child in message.child_messages:
            self._print_message_tree(child, session, indent + "│   ")
    
    def test_hierarchical_queries(self):
        """Test various hierarchical queries"""
        print("\n🔎 Testing hierarchical queries...")
        
        session = self.SessionLocal()
        try:
            # Test 1: Find all root messages
            root_messages = session.query(Message).filter_by(
                is_root_message=True,
                parent_id=None
            ).all()
            print(f"   ✅ Found {len(root_messages)} root messages")
            
            # Test 2: Find messages by depth level
            for depth in range(0, 4):
                depth_messages = session.query(Message).filter_by(depth_level=depth).all()
                if depth_messages:
                    print(f"   ✅ Depth {depth}: {len(depth_messages)} messages")
            
            # Test 3: Find longest conversation chains
            deepest_message = session.query(Message).order_by(Message.depth_level.desc()).first()
            if deepest_message:
                path = deepest_message.get_message_path(session)
                print(f"   ✅ Longest chain: {len(path)} messages deep")
                print(f"      Path: {' -> '.join([m.message_id for m in path])}")
            
            # Test 4: Thread with most replies
            thread_reply_counts = []
            for thread in session.query(Thread).all():
                reply_count = session.query(Message).filter_by(thread_id=thread.topic_id).filter(Message.is_root_message == False).count()
                thread_reply_counts.append((thread.header[:30], reply_count))
            
            thread_reply_counts.sort(key=lambda x: x[1], reverse=True)
            print(f"   ✅ Most active thread: '{thread_reply_counts[0][0]}...' with {thread_reply_counts[0][1]} replies")
            
        finally:
            session.close()
    
    def test_solution_analysis(self):
        """Analyze solutions and their effectiveness"""
        print("\n💡 Analyzing solutions...")
        
        session = self.SessionLocal()
        try:
            solutions = session.query(Solution).all()
            
            status_counts = {}
            confidence_scores = []
            
            for solution in solutions:
                # Count by status
                status_counts[solution.label] = status_counts.get(solution.label, 0) + 1
                
                # Collect confidence scores
                if solution.confidence_score:
                    confidence_scores.append(solution.confidence_score)
                
                print(f"   📌 {solution.header[:40]}...")
                print(f"      Status: {solution.label}")
                print(f"      Confidence: {solution.confidence_score}%")
                print(f"      solution: {solution.solution[:60]}...")
                print()
            
            # Summary stats
            print(f"   📊 solution Status Distribution:")
            for status, count in status_counts.items():
                print(f"      {status}: {count}")
            
            if confidence_scores:
                avg_confidence = sum(confidence_scores) / len(confidence_scores)
                print(f"   📊 Average Confidence: {avg_confidence:.1f}%")
                
        finally:
            session.close()
    
    def test_performance_queries(self):
        """Test query performance with indexes"""
        print("\n⚡ Testing query performance...")
        
        session = self.SessionLocal()
        try:
            import time
            
            # Test indexed queries
            queries = [
                ("Find thread messages", lambda: session.query(Message).filter_by(thread_id="msg_1001").all()),
                ("Find root messages", lambda: session.query(Message).filter_by(is_root_message=True).all()),
                ("Find messages by author", lambda: session.query(Message).filter_by(author_id="user_newbie_123").all()),
                ("Find messages by parent", lambda: session.query(Message).filter_by(parent_id="msg_1001").all()),
                ("Complex hierarchy query", lambda: session.query(Message).filter(
                    Message.thread_id == "msg_2001",
                    Message.depth_level >= 2
                ).all())
            ]
            
            for query_name, query_func in queries:
                start_time = time.time()
                results = query_func()
                end_time = time.time()
                
                duration_ms = (end_time - start_time) * 1000
                print(f"   ✅ {query_name}: {len(results)} results in {duration_ms:.2f}ms")
                
        finally:
            session.close()
    
    def test_data_integrity(self):
        """Test data integrity and constraints"""
        print("\n🛡️  Testing data integrity...")
        
        session = self.SessionLocal()
        try:
            # Test 1: All messages in threads have valid thread references
            messages_with_threads = session.query(Message).filter(Message.thread_id.isnot(None)).all()
            for msg in messages_with_threads:
                thread = session.get(Thread, msg.thread_id)
                assert thread is not None, f"Message {msg.message_id} references non-existent thread {msg.thread_id}"
            print("   ✅ All message->thread references are valid")
            
            # Test 2: All parent references are valid
            messages_with_parents = session.query(Message).filter(Message.parent_id.isnot(None)).all()
            for msg in messages_with_parents:
                parent = session.get(Message, msg.parent_id)
                assert parent is not None, f"Message {msg.message_id} references non-existent parent {msg.parent_id}"
                assert parent.thread_id == msg.thread_id, f"Message {msg.message_id} and parent {msg.parent_id} are in different threads"
            print("   ✅ All parent->child relationships are valid")
            
            # Test 3: Root messages have correct properties
            root_messages = session.query(Message).filter_by(is_root_message=True).all()
            for root in root_messages:
                assert root.parent_id is None, f"Root message {root.message_id} has a parent"
                assert root.depth_level == 0, f"Root message {root.message_id} has non-zero depth"
            print("   ✅ All root message properties are correct")
            
            # Test 4: Depth levels are consistent
            all_messages = session.query(Message).all()
            for msg in all_messages:
                if msg.parent_id:
                    parent = session.get(Message, msg.parent_id)
                    expected_depth = parent.depth_level + 1
                    assert msg.depth_level == expected_depth, f"Message {msg.message_id} has incorrect depth level"
            print("   ✅ All depth levels are consistent")
            
            # Test 5: Solutions reference valid threads
            solutions = session.query(Solution).all()
            for solution in solutions:
                thread = session.get(Thread, solution.thread_id)
                assert thread is not None, f"solution references non-existent thread {solution.thread_id}"
            print("   ✅ All solution->thread references are valid")
            
        finally:
            session.close()
    
    def cleanup(self):
        """Clean up test database"""
        Base.metadata.drop_all(bind=self.engine)
        self.engine.dispose()
        print("✅ Test database cleaned up")

def run_sample_tests():
    """Run all tests with sample data"""
    print("🚀 Running comprehensive database tests with sample data...")
    print("=" * 70)
    
    test_suite = TestWithSampleData()
    
    try:
        # Load sample data
        test_suite.load_sample_data()
        
        # Run all tests
        test_methods = [
            test_suite.test_thread_structure_analysis,
            test_suite.test_hierarchical_queries,
            test_suite.test_solution_analysis,
            test_suite.test_performance_queries,
            test_suite.test_data_integrity
        ]
        
        for test_method in test_methods:
            try:
                test_method()
                print(f"✅ {test_method.__name__} passed")
            except Exception as e:
                print(f"❌ {test_method.__name__} failed: {e}")
                raise
        
        print("\n" + "=" * 70)
        print("🎉 All sample data tests passed!")
        print("✅ Your hierarchical database structure is working perfectly")
        print("🚀 Ready for real Discord data!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        raise
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    run_sample_tests()