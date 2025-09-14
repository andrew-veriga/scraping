# Database Testing Guide

This directory contains comprehensive tests for your new hierarchical database structure with `message_id` as primary key and proper parent-child relationships.

## Quick Start

1. **Set up test database**:
   ```bash
   # Create test database
   createdb test_llmthreads
   
   # Copy and configure test environment
   cp .env.test .env
   # Edit .env with your test database credentials
   ```

2. **Run basic tests**:
   ```bash
   python test_database_operations.py
   ```

3. **Run tests with realistic sample data**:
   ```bash
   python test_with_samples.py
   ```

## Test Files Overview

### `test_database_operations.py`
Core database functionality tests:
- ✅ Thread and message creation
- ✅ Hierarchical parent-child relationships
- ✅ Self-referential foreign keys
- ✅ Utility methods (get_all_descendants, get_message_path, etc.)
- ✅ Complex thread scenarios
- ✅ solution linking
- ✅ Query performance with indexes

### `test_with_samples.py`
Realistic scenario tests using sample Discord-like data:
- ✅ Thread structure analysis
- ✅ Hierarchical queries across multiple threads
- ✅ solution effectiveness analysis
- ✅ Performance testing with realistic data
- ✅ Data integrity validation

### `test_data_samples.py`
Sample data generator with three realistic scenarios:
1. **Technical Support Thread**: Node deployment issue with firewall solution
2. **Development Thread**: Complex Move language discussion with code examples
3. **Unresolved Issue**: Ongoing investigation of object ownership bug

### `run_db_tests.py`
Simple test runner for basic operations.

## Database Schema Changes

### New Message Structure
```python
class Message(Base):
    message_id = Column(String(50), primary_key=True)  # Changed from integer ID
    parent_id = Column(String(50), ForeignKey('messages.message_id'))  # Self-reference
    thread_id = Column(String(50), ForeignKey('threads.topic_id'))
    depth_level = Column(Integer, default=0)  # 0=root, 1+=replies
    is_root_message = Column(Boolean, default=False)  # Thread starter
    order_in_thread = Column(Integer)  # Message order
```

### New Thread Structure  
```python
class Thread(Base):
    topic_id = Column(String(50), primary_key=True)  # Changed from integer ID
    # Direct one-to-many relationship with messages
    messages = relationship("Message", back_populates="thread")
```

### Key Improvements
- ✅ **Natural primary keys**: Use meaningful `message_id` and `topic_id`
- ✅ **Direct relationships**: No junction table needed
- ✅ **Hierarchical support**: Self-referential parent-child structure
- ✅ **Performance optimized**: Proper indexes for hierarchy queries
- ✅ **Utility methods**: Built-in tree traversal functions

## Sample Data Structure

The test data creates realistic Discord thread scenarios:

```
Thread: "Sui Node deployment failing"
├── msg_1001 (root) - Problem description
    ├── msg_1002 - Helpful response about firewall
    │   └── msg_1003 - Detailed solution with commands
    │       └── msg_1004 - Confirmation from original poster
    └── msg_1005 - Additional tip (parallel branch)
```

## Running Tests

### Prerequisites
```bash
# Install dependencies
pip install sqlalchemy postgresql psycopg2-binary python-dotenv

# Ensure PostgreSQL is running
# Create test database: test_llmthreads
```

### Environment Setup
```bash
# Copy test environment file
cp .env.test .env

# Edit .env with your settings:
TEST_DB_URL=postgresql://username:password@localhost:5432/test_llmthreads
```

### Test Execution
```bash
# Run basic functionality tests
python test_database_operations.py

# Run comprehensive tests with sample data
python test_with_samples.py

# Or use the simple runner
python run_db_tests.py
```

## Expected Test Output

Successful test run will show:
```
🚀 Starting database operation tests...
✅ Test 1/6 passed: test_create_thread_with_hierarchical_messages
✅ Test 2/6 passed: test_hierarchical_relationships
✅ Test 3/6 passed: test_utility_methods
✅ Test 4/6 passed: test_complex_thread_scenario
✅ Test 5/6 passed: test_solution_creation
✅ Test 6/6 passed: test_query_performance

🎉 All database tests passed successfully!
✅ Your database structure is ready for real data
```

## Key Features Tested

1. **Message Hierarchy**:
   - Self-referential foreign keys work correctly
   - Parent-child relationships are bidirectional
   - Depth levels are calculated properly
   - Root messages are identified correctly

2. **Thread Management**:
   - One-to-many relationship between threads and messages
   - Thread-level metadata is preserved
   - Solutions can be linked to threads

3. **Query Performance**:
   - Indexed queries execute efficiently
   - Hierarchical traversal is optimized
   - Complex filtering works as expected

4. **Data Integrity**:
   - All foreign key constraints are enforced
   - Circular references are prevented
   - Orphaned records are handled properly

## Troubleshooting

### Common Issues

1. **Database Connection Error**:
   ```
   Could not connect to database
   ```
   solution: Check your `TEST_DB_URL` in `.env` file

2. **Table Creation Error**:
   ```
   Permission denied for schema public
   ```
   solution: Ensure your database user has CREATE privileges

3. **Import Error**:
   ```
   ModuleNotFoundError: No module named 'app.models'
   ```
   solution: Run tests from the project root directory

### Debug Mode
Set `echo=True` in the SQLAlchemy engine creation to see all SQL queries:

```python
self.engine = create_engine(test_db_url, echo=True)
```

## Next Steps

After all tests pass:

1. **Update your .env file** to point to your real 'LLMThreads' database
2. **Run Alembic migrations** to create the production tables:
   ```bash
   alembic upgrade head
   ```
3. **Start loading real Discord data** using the new structure

Your hierarchical message database structure is now fully tested and ready for production use! 🚀