#!/usr/bin/env python3
"""
Connection Pool Monitor for Discord SUI Analyzer

This script monitors the database connection pool and can help diagnose
asyncio event loop issues caused by connection pool exhaustion.
"""

import time
import requests
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def monitor_connection_pool(base_url="http://localhost:8000", interval=30):
    """
    Monitor connection pool status and alert on issues.
    
    Args:
        base_url: Base URL of the FastAPI service
        interval: Monitoring interval in seconds
    """
    logger = logging.getLogger(__name__)
    
    while True:
        try:
            # Get pool status
            response = requests.get(f"{base_url}/pool-status", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                pool_status = data.get('pool_status', {})
                
                utilization = pool_status.get('utilization_percent', 0)
                status = pool_status.get('status', 'unknown')
                checked_out = pool_status.get('checked_out', 0)
                max_total = pool_status.get('max_total_connections', 0)
                
                # Log current status
                logger.info(f"Pool Status: {status} | Utilization: {utilization}% | "
                          f"Connections: {checked_out}/{max_total}")
                
                # Alert on high utilization
                if utilization > 80:
                    logger.warning(f"HIGH UTILIZATION: {utilization}% - "
                                 f"Consider increasing pool size or investigating connection leaks")
                
                if status == 'critical':
                    logger.error(f"CRITICAL POOL STATUS: {pool_status}")
                    
                    # Try emergency cleanup
                    try:
                        cleanup_response = requests.post(f"{base_url}/cleanup-connections", timeout=30)
                        if cleanup_response.status_code == 200:
                            logger.info("Emergency cleanup initiated")
                        else:
                            logger.error(f"Emergency cleanup failed: {cleanup_response.text}")
                    except Exception as e:
                        logger.error(f"Failed to initiate emergency cleanup: {e}")
                
            else:
                logger.error(f"Failed to get pool status: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Connection error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        
        time.sleep(interval)

def check_health(base_url="http://localhost:8000"):
    """Check overall service health."""
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"Service Health: {health_data.get('status', 'unknown')}")
            print(f"Database: {health_data.get('database', {}).get('status', 'unknown')}")
            return True
        else:
            print(f"Health check failed: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Discord SUI Analyzer connection pool")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the service")
    parser.add_argument("--interval", type=int, default=30,
                       help="Monitoring interval in seconds")
    parser.add_argument("--health-only", action="store_true",
                       help="Only check health once and exit")
    
    args = parser.parse_args()
    
    if args.health_only:
        check_health(args.url)
    else:
        print(f"Starting connection pool monitoring for {args.url}")
        print(f"Monitoring interval: {args.interval} seconds")
        print("Press Ctrl+C to stop")
        try:
            monitor_connection_pool(args.url, args.interval)
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
