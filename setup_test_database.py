"""
Database setup script with pgvector extension handling
Run this before running tests: python setup_test_database.py
"""

import os
import sys
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv('.env.test')

def setup_test_database():
    """Set up test database with proper pgvector handling"""
    
    test_db_url = os.getenv('TEST_DB_URL')
    if not test_db_url:
        print("❌ TEST_DB_URL not found in .env.test file")
        return False
    
    print(f"🔗 Connecting to: {test_db_url}")
    
    try:
        # Connect to database
        engine = create_engine(test_db_url)
        
        with engine.connect() as conn:
            # Try to create pgvector extension
            try:
                print("📦 Installing pgvector extension...")
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                conn.commit()
                print("✅ pgvector extension installed successfully")
                
                # Test if vector type is available
                result = conn.execute(text("SELECT typname FROM pg_type WHERE typname = 'vector';"))
                if result.fetchone():
                    print("✅ VECTOR type is available")
                    return True
                else:
                    print("⚠️  VECTOR type not found, will use JSON fallback")
                    return True
                    
            except Exception as e:
                print(f"⚠️  Could not install pgvector extension: {e}")
                print("📝 This is OK - tests will use JSON fallback for embeddings")
                return True
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Make sure your TEST_DB_URL is correct and database is accessible")
        return False

def check_database_requirements():
    """Check if all requirements are met"""
    
    print("🔍 Checking database requirements...")
    
    # Check if .env.test exists
    if not os.path.exists('.env.test'):
        print("❌ .env.test file not found")
        print("💡 Please create .env.test with your TEST_DB_URL")
        return False
    
    # Check if TEST_DB_URL is set
    load_dotenv('.env.test')
    test_db_url = os.getenv('TEST_DB_URL')
    if not test_db_url:
        print("❌ TEST_DB_URL not set in .env.test")
        return False
    
    print("✅ Configuration files found")
    return True

if __name__ == "__main__":
    print("🚀 Setting up test database...")
    print("=" * 50)
    
    if not check_database_requirements():
        sys.exit(1)
    
    if setup_test_database():
        print("\n✅ Test database setup complete!")
        print("🚀 You can now run the tests:")
        print("   python test_database_operations.py")
        print("   python test_with_samples.py")
    else:
        print("\n❌ Database setup failed")
        sys.exit(1)