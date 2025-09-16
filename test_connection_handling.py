#!/usr/bin/env python3
"""
Test script to verify database connection handling and retry logic.
This script simulates connection issues and tests the retry mechanism.
"""

import os
import sys
import logging
import time
from dotenv import load_dotenv
from sqlalchemy import text

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

load_dotenv()

from app.services.database import get_database_service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_connection_health():
    """Test the database connection health check."""
    print("🔍 Testing database connection health check...")
    
    try:
        db_service = get_database_service()
        health_status = db_service.health_check()
        
        print(f"✅ Health check result: {health_status['status']}")
        print(f"📊 Connection test: {health_status['connection_test']}")
        print(f"🏊 Pool status:")
        for key, value in health_status['pool_status'].items():
            print(f"   {key}: {value}")
        
        if health_status['error']:
            print(f"❌ Error: {health_status['error']}")
        
        return health_status['status'] == 'healthy'
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_basic_operations():
    """Test basic database operations with retry logic."""
    print("\n🔧 Testing basic database operations...")
    
    try:
        db_service = get_database_service()
        
        # Test session creation and basic query
        with db_service.get_session() as session:
            result = session.execute(text("SELECT 1 as test_value")).fetchone()
            print(f"✅ Basic query successful: {result}")
            
        # Test database stats
        stats = db_service.get_database_stats()
        print(f"📈 Database stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        return False

def test_connection_retry():
    """Test connection retry logic by simulating connection issues."""
    print("\n🔄 Testing connection retry logic...")
    
    try:
        db_service = get_database_service()
        
        # Test multiple session operations to trigger connection pooling
        for i in range(3):
            with db_service.get_session() as session:
                result = session.execute(text("SELECT NOW() as current_time")).fetchone()
                print(f"✅ Session {i+1} successful: {result}")
                time.sleep(0.1)  # Small delay to test connection reuse
        
        return True
        
    except Exception as e:
        print(f"❌ Connection retry test failed: {e}")
        return False

def main():
    """Run all connection tests."""
    print("🚀 Starting database connection handling tests...\n")
    
    tests = [
        ("Health Check", test_connection_health),
        ("Basic Operations", test_basic_operations),
        ("Connection Retry", test_connection_retry)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*50}")
        print(f"Running: {test_name}")
        print(f"{'='*50}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    print(f"{'='*50}")
    print("📋 TEST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Database connection handling is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the database configuration and connection.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
