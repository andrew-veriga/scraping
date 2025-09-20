"""
Sample data generator for testing database operations
This creates realistic test data that mimics Discord thread structures
"""

from datetime import datetime, timezone, timedelta
from app.models.db_models import Message, Thread, Solution
from app.models.pydantic_models import ThreadStatus
import random

class TestDataGenerator:
    """Generates realistic test data for database testing"""
    
    @staticmethod
    def create_sample_thread_1():
        """Creates a technical support thread about node deployment"""
        thread_data = {
            'thread': Thread(
                topic_id="msg_1001",
                header="Sui Node deployment failing on Ubuntu 22.04",
                actual_date=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                status=ThreadStatus.NEW,
                is_technical=True,
                answer_id="msg_1003"  # Points to the solution message
            ),
            'messages': [
                # Root message (problem description)
                Message(
                    message_id="msg_1001",
                    thread_id="msg_1001", 
                    parent_id=None,
                    author_id="user_newbie_123",
                    content="Hi everyone! I'm trying to deploy a Sui validator node on Ubuntu 22.04 but getting connection errors. The node starts but can't sync with the network. Error: 'Connection refused on port 9000'. Any ideas?",
                    datetime=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                    referenced_message_id=None,
                    is_root_message=True,
                    order_in_thread=1
                ),
                # First helpful response
                Message(
                    message_id="msg_1002",
                    thread_id="msg_1001",
                    parent_id="msg_1001", 
                    author_id="helper_dev_456",
                    content="Sounds like a firewall issue. Did you open the required ports? Sui needs 9000-9184 to be open.",
                    datetime=datetime(2024, 1, 15, 10, 5, 0, tzinfo=timezone.utc),
                    depth_level=1,
                    is_root_message=False,
                    order_in_thread=2
                ),
                # solution message
                Message(
                    message_id="msg_1003",
                    thread_id="msg_1001",
                    parent_id="msg_1002",
                    author_id="expert_validator_789", 
                    content="Yes, firewall is the issue. Run these commands:\\n```bash\\nsudo ufw allow 9000:9184/tcp\\nsudo ufw allow 9000:9184/udp\\nsudo systemctl restart sui-node\\n```\\nAlso make sure your genesis.blob file is up to date.",
                    datetime=datetime(2024, 1, 15, 10, 10, 0, tzinfo=timezone.utc),
                    depth_level=2,
                    is_root_message=False,
                    order_in_thread=3
                ),
                # Confirmation from original poster
                Message(
                    message_id="msg_1004",
                    thread_id="msg_1001",
                    parent_id="msg_1003",
                    author_id="user_newbie_123",
                    content="Perfect! That solved it completely. Node is now syncing properly. Thanks so much! 🙏",
                    datetime=datetime(2024, 1, 15, 10, 25, 0, tzinfo=timezone.utc),
                    depth_level=3,
                    is_root_message=False,
                    order_in_thread=4
                ),
                # Additional helpful tip (parallel branch)
                Message(
                    message_id="msg_1005",
                    thread_id="msg_1001",
                    parent_id="msg_1001",
                    author_id="senior_dev_999",
                    content="For future reference, you can also check node logs with `docker logs sui-node-container` to diagnose network issues faster.",
                    datetime=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
                    depth_level=1,
                    is_root_message=False,
                    order_in_thread=5
                )
            ],
            'solution': Solution(
                thread_id="msg_1001",
                header="Sui Node Connection Issues - Firewall Configuration",
                solution="Open required ports (9000-9184) using ufw, restart the node service, and ensure genesis.blob is current. Use docker logs to diagnose network connectivity issues.",
                label="resolved",
                confidence_score=98
            )
        }
        return thread_data
    
    @staticmethod 
    def create_sample_thread_2():
        """Creates a complex thread about smart contract development"""
        thread_data = {
            'thread': Thread(
                topic_id="msg_2001", 
                header="Move language: How to implement custom transfer logic?",
                actual_date=datetime(2024, 1, 16, 14, 30, 0, tzinfo=timezone.utc),
                status=ThreadStatus.NEW,
                is_technical=True,
                answer_id="msg_2004"
            ),
            'messages': [
                # Root message
                Message(
                    message_id="msg_2001",
                    thread_id="msg_2001",
                    parent_id=None,
                    author_id="move_dev_alice",
                    content="I'm building a custom token contract in Move and need to implement transfer logic with additional validation. The standard transfer doesn't meet my requirements. How do I override the default behavior?",
                    datetime=datetime(2024, 1, 16, 14, 30, 0, tzinfo=timezone.utc),
                    depth_level=0,
                    is_root_message=True,
                    order_in_thread=1
                ),
                # Clarification question
                Message(
                    message_id="msg_2002",
                    thread_id="msg_2001",
                    parent_id="msg_2001",
                    author_id="move_expert_bob",
                    content="What kind of additional validation do you need? Is it amount limits, time restrictions, or something else?",
                    datetime=datetime(2024, 1, 16, 14, 35, 0, tzinfo=timezone.utc),
                    depth_level=1,
                    is_root_message=False,
                    order_in_thread=2
                ),
                # Detailed response
                Message(
                    message_id="msg_2003",
                    thread_id="msg_2001",
                    parent_id="msg_2002",
                    author_id="move_dev_alice", 
                    content="I need to implement daily transfer limits per address and also require approval from a secondary signer for transfers above 1000 tokens.",
                    datetime=datetime(2024, 1, 16, 14, 40, 0, tzinfo=timezone.utc),
                    depth_level=2,
                    is_root_message=False,
                    order_in_thread=3
                ),
                # Detailed solution
                Message(
                    message_id="msg_2004",
                    thread_id="msg_2001",
                    parent_id="msg_2003",
                    author_id="move_expert_bob",
                    content="You'll need to create a custom transfer function with state tracking. Here's the approach:\\n\\n```move\\nstruct TransferLimits has key {\\n    daily_limits: Table<address, u64>,\\n    last_reset: u64\\n}\\n\\npublic fun custom_transfer(\\n    from: &signer,\\n    to: address, \\n    amount: u64,\\n    approver: Option<&signer>\\n) {\\n    // Check daily limit\\n    // Require approver for large transfers\\n    // Update state\\n}\\n```\\n\\nYou'll also need to implement time-based limit resets.",
                    datetime=datetime(2024, 1, 16, 14, 50, 0, tzinfo=timezone.utc),
                    depth_level=3,
                    is_root_message=False,
                    order_in_thread=4
                ),
                # Follow-up question
                Message(
                    message_id="msg_2005",
                    thread_id="msg_2001",
                    parent_id="msg_2004",
                    author_id="move_dev_alice",
                    content="Thanks! Quick question - how do I handle the time-based reset efficiently without running into gas limits?",
                    datetime=datetime(2024, 1, 16, 15, 0, 0, tzinfo=timezone.utc),
                    depth_level=4,
                    is_root_message=False,
                    order_in_thread=5
                ),
                # Gas optimization tip
                Message(
                    message_id="msg_2006",
                    thread_id="msg_2001", 
                    parent_id="msg_2005",
                    author_id="gas_optimizer_carol",
                    content="Use lazy evaluation! Only reset limits when they're actually checked, not proactively. Store the last reset timestamp and calculate if reset is needed during the transfer call.",
                    datetime=datetime(2024, 1, 16, 15, 5, 0, tzinfo=timezone.utc),
                    depth_level=5,
                    is_root_message=False,
                    order_in_thread=6
                )
            ],
            'solution': Solution(
                thread_id="msg_2001",
                header="Custom Transfer Logic with Daily Limits and Multi-sig Approval",
                solution="Implement custom transfer function with Table-based state tracking for daily limits, require optional approver signer for large transfers, use lazy evaluation for timestamp-based limit resets to optimize gas usage.",
                label="resolved", 
                confidence_score=95
            )
        }
        return thread_data
    
    @staticmethod
    def create_sample_thread_3():
        """Creates an unresolved thread about a complex bug"""
        thread_data = {
            'thread': Thread(
                topic_id="msg_3001",
                header="Strange behavior with object ownership after dynamic field operations",
                actual_date=datetime(2024, 1, 17, 9, 15, 0, tzinfo=timezone.utc),
                status=ThreadStatus.NEW,
                is_technical=True,
                answer_id=None  # No solution yet
            ),
            'messages': [
                # Problem description
                Message(
                    message_id="msg_3001",
                    thread_id="msg_3001",
                    parent_id=None,
                    author_id="confused_dev_dave",
                    content="I'm seeing weird behavior where objects become inaccessible after adding dynamic fields. The object exists on-chain but I can't reference it in subsequent transactions. Has anyone encountered this?",
                    datetime=datetime(2024, 1, 17, 9, 15, 0, tzinfo=timezone.utc),
                    depth_level=0,
                    is_root_message=True,
                    order_in_thread=1
                ),
                # Request for more details
                Message(
                    message_id="msg_3002",
                    thread_id="msg_3001",
                    parent_id="msg_3001",
                    author_id="debugger_eve",
                    content="Can you share the transaction hash and the specific operations you're performing? Also, what's the object type?",
                    datetime=datetime(2024, 1, 17, 9, 30, 0, tzinfo=timezone.utc),
                    depth_level=1,
                    is_root_message=False,
                    order_in_thread=2
                ),
                # More details provided
                Message(
                    message_id="msg_3003",
                    thread_id="msg_3001",
                    parent_id="msg_3002",
                    author_id="confused_dev_dave",
                    content="Object type is a custom NFT struct. TX hash: 0xabc123def456. I'm adding a dynamic field for metadata, then trying to transfer the NFT in the next transaction but getting 'object not found'.",
                    datetime=datetime(2024, 1, 17, 9, 45, 0, tzinfo=timezone.utc),
                    depth_level=2,
                    is_root_message=False,
                    order_in_thread=3
                ),
                # Potential lead
                Message(
                    message_id="msg_3004",
                    thread_id="msg_3001",
                    parent_id="msg_3003",
                    author_id="investigator_frank",
                    content="I'll check the transaction. This might be related to object wrapping behavior when dynamic fields are added. Are you using the correct object ID after the modification?",
                    datetime=datetime(2024, 1, 17, 10, 0, 0, tzinfo=timezone.utc),
                    depth_level=3,
                    is_root_message=False,
                    order_in_thread=4
                ),
                # Still investigating
                Message(
                    message_id="msg_3005",
                    thread_id="msg_3001",
                    parent_id="msg_3004",
                    author_id="investigator_frank",
                    content="Looking at the TX, the object ID does change after dynamic field addition. This might be expected behavior but poorly documented. Let me dig into the Sui docs...",
                    datetime=datetime(2024, 1, 17, 11, 0, 0, tzinfo=timezone.utc),
                    depth_level=4,
                    is_root_message=False,
                    order_in_thread=5
                )
            ],
            'solution': Solution(
                thread_id="msg_3001",
                header="Object Ownership Issues with Dynamic Fields",
                solution="Investigation ongoing - appears related to object ID changes after dynamic field operations.",
                label="unresolved",
                confidence_score=30
            )
        }
        return thread_data

    @staticmethod
    def get_all_sample_threads():
        """Returns all sample threads for batch testing"""
        return [
            TestDataGenerator.create_sample_thread_1(),
            TestDataGenerator.create_sample_thread_2(), 
            TestDataGenerator.create_sample_thread_3()
        ]

if __name__ == "__main__":
    # Demo the sample data
    print("📝 Sample Thread Data Preview:")
    print("=" * 50)
    
    for i, thread_data in enumerate(TestDataGenerator.get_all_sample_threads(), 1):
        thread = thread_data['thread']
        messages = thread_data['messages']
        solution = thread_data['solution']
        
        print(f"\\n🧵 Thread {i}: {thread.header}")
        print(f"   Messages: {len(messages)}")
        print(f"   solution Status: {solution.label}")
        print(f"   Max Depth: {max(m.depth_level for m in messages)}")
        
        # Show message tree structure
        root_msg = next(m for m in messages if m.is_root_message)
        print(f"   Tree Preview:")
        print(f"   ├── {root_msg.message_id} (root)")
        
        for msg in messages:
            if not msg.is_root_message:
                indent = "   " + "│   " * msg.depth_level + "├── "
                print(f"{indent}{msg.message_id} (depth {msg.depth_level})")
    
    print("\\n✅ Sample data ready for testing!")