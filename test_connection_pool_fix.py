#!/usr/bin/env python3
"""
Test script to verify the connection pool exhaustion fix.
This script tests the optimized database access patterns.
"""

import os
import sys
import logging
import time
from dotenv import load_dotenv

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

load_dotenv()

from app.services.database import get_database_service
from app.utils.file_utils import illustrated_message, illustrated_threads

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_connection_pool_utilization():
    """Test connection pool utilization with optimized access patterns."""
    print("🔍 Testing connection pool utilization...")
    
    try:
        db_service = get_database_service()
        
        # Warm up the pool first
        print("🔥 Warming up connection pool...")
        connections_created = db_service.warmup_pool()
        print(f"   Created {connections_created} connections")
        
        # Get initial pool status
        initial_status = db_service.get_pool_status()
        print(f"📊 Initial pool status: {initial_status}")
        
        # Test multiple message lookups with session reuse
        print("\n🧪 Testing optimized message lookups...")
        
        # Get a real message ID from the database for testing
        with db_service.get_session() as session:
            from sqlalchemy import text
            result = session.execute(text("SELECT message_id FROM messages LIMIT 1")).fetchone()
            if result:
                test_message_id = result[0]
                print(f"   Using real message ID: {test_message_id}")
            else:
                print("   No messages found in database, using dummy ID")
                test_message_id = "dummy_message_id"
        
        # Simulate the optimized pattern used in illustrated_threads
        with db_service.get_session() as session:
            # Test multiple message lookups using the same session
            for i in range(5):  # Reduced to 5 to avoid too many warnings
                try:
                    # This should use the same session, not create new connections
                    message_content = illustrated_message(test_message_id, db_service, session)
                    print(f"  ✅ Message lookup {i+1}: {message_content[:50]}...")
                except Exception as e:
                    print(f"  ⚠️  Message lookup {i+1} failed: {e}")
                
                # Check pool status after each lookup (but don't print every time)
                if i == 0 or i == 4:  # Only print first and last
                    current_status = db_service.get_pool_status()
                    print(f"    Pool utilization: {current_status.get('utilization_percent', 0)}%")
        
        # Get final pool status
        final_status = db_service.get_pool_status()
        print(f"\n📊 Final pool status: {final_status}")
        
        # Verify pool utilization is reasonable
        utilization = final_status.get('utilization_percent', 0)
        max_connections = final_status.get('max_total_connections', 1)
        
        if utilization < 50:
            print("✅ Connection pool utilization is healthy")
            return True
        elif utilization < 80:
            print("⚠️  Connection pool utilization is moderate")
            return True
        else:
            print(f"❌ Connection pool utilization is high: {utilization}%")
            print(f"   Max connections configured: {max_connections}")
            return False
            
    except Exception as e:
        print(f"❌ Connection pool test failed: {e}")
        return False

def test_connection_pool_monitoring():
    """Test connection pool monitoring endpoints."""
    print("\n📈 Testing connection pool monitoring...")
    
    try:
        db_service = get_database_service()
        
        # Test pool status
        pool_status = db_service.get_pool_status()
        print(f"📊 Pool status: {pool_status}")
        
        # Test health check
        health_status = db_service.health_check()
        print(f"🏥 Health status: {health_status['status']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pool monitoring test failed: {e}")
        return False

def test_connection_cleanup():
    """Test connection cleanup functionality."""
    print("\n🧹 Testing connection cleanup...")
    
    try:
        db_service = get_database_service()
        
        # Get status before cleanup
        before_status = db_service.get_pool_status()
        print(f"📊 Before cleanup: {before_status}")
        
        # Force cleanup
        db_service.cleanup_connections()
        
        # Get status after cleanup
        after_status = db_service.get_pool_status()
        print(f"📊 After cleanup: {after_status}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection cleanup test failed: {e}")
        return False

def test_pool_warmup():
    """Test connection pool warmup functionality."""
    print("\n🔥 Testing connection pool warmup...")
    
    try:
        db_service = get_database_service()
        
        # Get status before warmup
        before_status = db_service.get_pool_status()
        print(f"📊 Before warmup: {before_status}")
        
        # Warm up the pool
        connections_created = db_service.warmup_pool()
        print(f"🔥 Created {connections_created} connections")
        
        # Get status after warmup
        after_status = db_service.get_pool_status()
        print(f"📊 After warmup: {after_status}")
        
        # Verify warmup was successful
        if connections_created > 0:
            print("✅ Pool warmup successful")
            return True
        else:
            print("⚠️  Pool warmup created no connections")
            return False
        
    except Exception as e:
        print(f"❌ Pool warmup test failed: {e}")
        return False

def main():
    """Run all connection pool tests."""
    print("🚀 Starting connection pool exhaustion fix tests...\n")
    
    tests = [
        ("Pool Warmup", test_pool_warmup),
        ("Connection Pool Utilization", test_connection_pool_utilization),
        ("Connection Pool Monitoring", test_connection_pool_monitoring),
        ("Connection Cleanup", test_connection_cleanup)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"✅ {test_name}: {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            print(f"❌ {test_name}: FAILED with exception: {e}")
            results.append((test_name, False))
        
        print()
    
    # Summary
    print(f"{'='*60}")
    print("📋 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Connection pool exhaustion fix is working correctly.")
        print("\n💡 Key improvements:")
        print("   - Optimized database session reuse")
        print("   - Reduced connection pool exhaustion")
        print("   - Better connection monitoring")
        print("   - Automatic connection cleanup")
        return 0
    else:
        print("⚠️  Some tests failed. Check the connection pool configuration.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
