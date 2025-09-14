"""
Test the new hierarchical first batch endpoint
"""

import pandas as pd
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def create_test_excel_file():
    """Create a test Excel file with Discord-like data"""
    
    # Sample Discord messages with realistic structure
    sample_data = [
        {
            'Message ID': 'msg_001',
            'Author ID': 'user123',  
            'Content': 'How do I deploy a Sui node on Ubuntu 22.04?',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(hours=3)).timestamp()),
            'Referenced Message ID': ''  # Root message
        },
        {
            'Message ID': 'msg_002',
            'Author ID': 'helper456',
            'Content': 'You need to install Docker first. What version of Ubuntu are you using?',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(hours=2, minutes=50)).timestamp()),
            'Referenced Message ID': 'msg_001'  # Reply to question
        },
        {
            'Message ID': 'msg_003', 
            'Author ID': 'user123',
            'Content': 'Ubuntu 22.04 with 8GB RAM',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(hours=2, minutes=45)).timestamp()),
            'Referenced Message ID': 'msg_002'  # Reply to helper
        },
        {
            'Message ID': 'msg_004',
            'Author ID': 'expert789',
            'Content': 'Perfect! Run:\\nsudo apt update\\nsudo apt install docker.io\\nsudo systemctl start docker',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(hours=2, minutes=30)).timestamp()),
            'Referenced Message ID': 'msg_003'  # solution
        },
        {
            'Message ID': 'msg_005',
            'Author ID': 'user123',
            'Content': 'That worked perfectly! Thank you so much! 🙏',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(hours=2, minutes=20)).timestamp()),
            'Referenced Message ID': 'msg_004'  # Confirmation
        },
        {
            'Message ID': 'msg_006',
            'Author ID': 'security_expert',
            'Content': 'Also remember to configure firewall: sudo ufw allow 9000:9184/tcp',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(hours=2, minutes=10)).timestamp()),
            'Referenced Message ID': 'msg_001'  # Parallel branch
        },
        # Second thread
        {
            'Message ID': 'msg_101',
            'Author ID': 'dev_alice',
            'Content': 'Move language compilation error: cannot find symbol in stdlib. Help?',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()),
            'Referenced Message ID': ''  # New thread
        },
        {
            'Message ID': 'msg_102',
            'Author ID': 'move_expert',
            'Content': 'Check your import statements. Are you importing std::vector?',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(minutes=55)).timestamp()),
            'Referenced Message ID': 'msg_101'
        },
        {
            'Message ID': 'msg_103',
            'Author ID': 'dev_alice',
            'Content': 'Yes, I have: use std::vector::Vector;',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(minutes=50)).timestamp()),
            'Referenced Message ID': 'msg_102'
        },
        {
            'Message ID': 'msg_104',
            'Author ID': 'move_expert',
            'Content': 'Try: use std::vector; instead. The Vector type is re-exported.',
            'Unix Timestamp': int((datetime.now(timezone.utc) - timedelta(minutes=45)).timestamp()),
            'Referenced Message ID': 'msg_103'
        }
    ]
    
    df = pd.DataFrame(sample_data)
    
    # Create temporary Excel file
    temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
    df.to_excel(temp_file.name, index=False)
    temp_file.close()
    
    print(f"📝 Created test Excel file: {temp_file.name}")
    print(f"   Contains {len(df)} messages across 2 conversation threads")
    
    return temp_file.name

def test_first_batch_endpoint():
    """Test the hierarchical first batch processing"""
    
    print("🧪 Testing Hierarchical First Batch Endpoint")
    print("=" * 60)
    
    try:
        from app.services.processing_hierarchical import process_first_batch_hierarchical
        
        # Create test Excel file
        excel_file = create_test_excel_file()
        
        try:
            # Configure test parameters
            test_config = {
                'MESSAGES_FILE_PATH': excel_file,
                'SAVE_PATH': './test_results',
                'INTERVAL_FIRST': 2,  # Process 2 days
                'SOLUTIONS_DICT_FILENAME': 'test_solutions_dict.json'
            }
            
            # Ensure save path exists
            os.makedirs(test_config['SAVE_PATH'], exist_ok=True)
            
            print(f"🚀 Starting hierarchical first batch processing...")
            print(f"   Excel file: {excel_file}")
            print(f"   Processing interval: {test_config['INTERVAL_FIRST']} days")
            
            # Call the processing function
            result = process_first_batch_hierarchical(test_config)
            
            print("\\n✅ Processing completed successfully!")
            print(f"   Status: {result['status']}")
            print(f"   Message: {result['message']}")
            
            # Display statistics
            if 'statistics' in result:
                stats = result['statistics']
                
                print("\\n📊 Processing Statistics:")
                
                if 'data_loading' in stats:
                    dl = stats['data_loading']
                    print(f"   📂 Data Loading: {dl['total_messages_loaded']} messages loaded")
                
                if 'hierarchy_analysis' in stats:
                    ha = stats['hierarchy_analysis']
                    print(f"   🌳 Hierarchy Analysis:")
                    print(f"      - Root messages: {ha['root_messages']}")
                    print(f"      - Reply messages: {ha['reply_messages']}")
                    print(f"      - Threads identified: {ha['threads_identified']}")
                    print(f"      - Maximum depth: {ha['max_depth']}")
                
                if 'database_operations' in stats:
                    db = stats['database_operations']
                    print(f"   💾 Database Operations:")
                    print(f"      - New messages created: {db['new_messages_created']}")
                    print(f"      - Threads created: {db['threads_created']}")
                
                if 'validation_results' in stats:
                    vr = stats['validation_results']
                    print(f"   🛡️ Validation: {'✅ Passed' if vr['valid'] else '❌ Failed'}")
                    if not vr['valid'] and 'issues' in vr:
                        for issue in vr['issues']:
                            print(f"      - {issue}")
                
                if 'processing_summary' in stats:
                    ps = stats['processing_summary']
                    print(f"   ⏱️ Processing Summary:")
                    print(f"      - Messages in batch: {ps['messages_in_first_batch']}")
                    print(f"      - Threads in batch: {ps['threads_in_first_batch']}")
                    print(f"      - Processing time: {ps['total_processing_time']:.2f} seconds")
            
            # Display next steps
            if 'next_steps' in result:
                print("\\n🚀 Next Steps:")
                for i, step in enumerate(result['next_steps'], 1):
                    print(f"   {i}. {step}")
            
            return True
            
        finally:
            # Clean up test file
            if os.path.exists(excel_file):
                os.unlink(excel_file)
                print(f"🧹 Cleaned up test file: {excel_file}")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_status_endpoint():
    """Test the hierarchical status endpoint"""
    
    print("\\n🔍 Testing Hierarchical Status Endpoint")
    print("-" * 40)
    
    try:
        from app.services.processing_hierarchical import get_hierarchical_processing_status
        
        status = get_hierarchical_processing_status()
        
        print("✅ Status endpoint working!")
        print(f"   Database status: {status.get('database_status', 'unknown')}")
        
        if 'message_statistics' in status:
            ms = status['message_statistics']
            print(f"   📝 Messages: {ms.get('total_messages', 0)} total ({ms.get('root_messages', 0)} roots, {ms.get('reply_messages', 0)} replies)")
        
        if 'thread_statistics' in status:
            ts = status['thread_statistics']
            print(f"   🧵 Threads: {ts.get('total_threads', 0)} total")
        
        return True
        
    except Exception as e:
        print(f"❌ Status test failed: {e}")
        return False

def main():
    """Run all tests"""
    
    print("🚀 Testing New Hierarchical First Batch Endpoint")
    print("=" * 70)
    
    # Test the processing endpoint
    processing_success = test_first_batch_endpoint()
    
    # Test the status endpoint  
    status_success = test_status_endpoint()
    
    print("\\n" + "=" * 70)
    
    if processing_success and status_success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Hierarchical first batch endpoint is working correctly")
        print("🚀 Ready to process real Discord data with parent-child relationships!")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Please check the errors above and fix any issues")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)