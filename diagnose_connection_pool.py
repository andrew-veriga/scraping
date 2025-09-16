#!/usr/bin/env python3
"""
Diagnostic script to understand connection pool behavior.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

load_dotenv()

from app.services.database import get_database_service

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def diagnose_connection_pool():
    """Diagnose connection pool configuration and behavior."""
    print("🔍 Diagnosing connection pool configuration...\n")
    
    try:
        db_service = get_database_service()
        
        # Get engine information
        engine = db_service.engine
        print(f"📊 Engine URL: {engine.url}")
        print(f"📊 Engine name: {engine.name}")
        
        # Get pool information
        pool = engine.pool
        print(f"\n🏊 Pool type: {type(pool).__name__}")
        print(f"🏊 Pool class: {pool.__class__}")
        
        # Get pool attributes
        pool_attrs = [attr for attr in dir(pool) if not attr.startswith('_') and not callable(getattr(pool, attr))]
        print(f"🏊 Pool attributes: {pool_attrs}")
        
        # Try to get pool configuration
        try:
            print(f"🏊 Pool size: {getattr(pool, '_pool_size', 'Not available')}")
            print(f"🏊 Max overflow: {getattr(pool, '_max_overflow', 'Not available')}")
            print(f"🏊 Pool timeout: {getattr(pool, '_timeout', 'Not available')}")
            print(f"🏊 Pool recycle: {getattr(pool, '_recycle', 'Not available')}")
        except Exception as e:
            print(f"⚠️  Could not get pool configuration: {e}")
        
        # Get current pool status
        print(f"\n📈 Current pool status:")
        status = db_service.get_pool_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
        
        # Test connection creation
        print(f"\n🧪 Testing connection creation...")
        try:
            with engine.connect() as conn:
                print(f"   ✅ Successfully created connection: {conn}")
                result = conn.execute("SELECT 1 as test").fetchone()
                print(f"   ✅ Query result: {result}")
        except Exception as e:
            print(f"   ❌ Connection test failed: {e}")
        
        # Test multiple connections
        print(f"\n🧪 Testing multiple connections...")
        connections = []
        try:
            for i in range(3):
                conn = engine.connect()
                connections.append(conn)
                print(f"   ✅ Created connection {i+1}: {conn}")
        except Exception as e:
            print(f"   ❌ Multiple connection test failed: {e}")
        finally:
            # Close connections
            for conn in connections:
                try:
                    conn.close()
                    print(f"   🔒 Closed connection: {conn}")
                except:
                    pass
        
        # Final pool status
        print(f"\n📈 Final pool status:")
        final_status = db_service.get_pool_status()
        for key, value in final_status.items():
            print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        return False

if __name__ == "__main__":
    success = diagnose_connection_pool()
    sys.exit(0 if success else 1)
