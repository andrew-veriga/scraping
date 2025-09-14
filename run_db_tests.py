"""
Simple test runner script
Run with: python run_db_tests.py
"""

from test_database_operations import run_tests

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\\n❌ Test execution failed: {e}")
        exit(1)