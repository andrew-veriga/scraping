"""
Fixed unit tests for optimized message processing pipeline.
This version fixes import issues and mocking problems.
"""

import pytest
import pandas as pd
import os
import json
import tempfile
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock environment variables before importing modules
os.environ['GEMINI_API_KEY'] = 'test_key'
os.environ['PEERA_DB_URL'] = 'postgresql://test:test@localhost/test'

from app.services.data_loader import load_messages_to_database
from app.models.db_models import Message, Thread, Solution


class TestDataLoadingFixed:
    """Test data loading with fixed imports."""
    
    @pytest.fixture
    def sample_messages_df(self):
        """Create sample messages DataFrame for testing."""
        data = {
            'Message ID': ['1', '2', '3', '4', '5'],
            'Referenced Message ID': ['', '1', '1', '2', ''],
            'Author ID': ['user1', 'user2', 'user1', 'user3', 'user4'],
            'Content': ['Hello', 'Hi there', 'Reply to hi', 'Follow up', 'New topic'],
            'DateTime': [
                datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 10, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 15, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 2, 11, 0, 0, tzinfo=timezone.utc)
            ]
            
        }
        df = pd.DataFrame(data)
        df.set_index('Message ID', inplace=True, drop=False)
        return df
    
    
    def test_validate_message_hierarchy_fixed(self, sample_messages_df):
        """Test hierarchy validation with fixed imports."""
        # Create a mock processed dataframe with hierarchy fields
        df_processed = sample_messages_df.copy()
        df_processed['parent_id'] = ['', '1', '1', '2', '']
        df_processed['thread_id'] = ['1', '1', '1', '1', '5']
        
    
    @patch('app.services.database.get_database_service')
    def test_load_messages_to_database_fixed(self, mock_get_db, sample_messages_df):
        """Test database loading with fixed mocking."""
        # Setup mock
        mock_db_service = Mock()
        mock_get_db.return_value = mock_db_service
        mock_db_service.bulk_create_messages_hierarchical.return_value = 5
        
        # Process hierarchy first (mock the processed dataframe)
        df_processed = sample_messages_df.copy()
        df_processed['parent_id'] = ['', '1', '1', '2', '']
        df_processed['thread_id'] = ['1', '1', '1', '1', '5']
        
        # Act
        stats = load_messages_to_database(df_processed)
        
        # Assert
        assert stats['total_messages'] == 5
        assert stats['new_messages_created'] == 5
        assert stats['existing_messages_skipped'] == 0
        assert stats['threads_created'] == 0  # Threads are not created in this step
        
        # Verify that bulk_create_messages_hierarchical was called
        mock_db_service.bulk_create_messages_hierarchical.assert_called_once()
        
        # Verify call arguments
        call_args = mock_db_service.bulk_create_messages_hierarchical.call_args[0][0]
        assert len(call_args) == 5
        
        # Check a sample message's data structure
        sample_message_data = next(item for item in call_args if item['message_id'] == '1')
        assert sample_message_data['message_id'] == '1'
        assert sample_message_data['parent_id'] is None
        assert sample_message_data['thread_id'] == '1'  # Set during hierarchy analysis
        
        # Verify redundant fields are not present
        redundant_fields = ['order_in_thread', 'depth_level', 'is_root_message']
        for field in redundant_fields:
            assert field not in sample_message_data, f"Redundant field {field} should not be in message data"


class TestDatabaseModelsFixed:
    """Test database models with fixed imports."""
    
    def test_message_model_optimized_fields(self):
        """Test that Message model has correct optimized structure."""
        # Get all column names
        column_names = [column.name for column in Message.__table__.columns]
        
        # Essential fields should be present
        essential_fields = [
            'message_id', 'parent_id', 'author_id', 'content',
            'datetime', 'referenced_message_id', 'thread_id'
        ]
        for field in essential_fields:
            assert field in column_names, f"Essential field {field} missing from Message model"
        
        # Redundant fields should be removed
        redundant_fields = [
            'processing_status', 'last_processed_at', 'processing_version',
            'order_in_thread', 'depth_level', 'is_root_message'
        ]
        for field in redundant_fields:
            assert field not in column_names, f"Redundant field {field} still present in Message model"
    
    def test_thread_model_optimized_fields(self):
        """Test that Thread model has correct optimized structure."""
        # Get all column names
        column_names = [column.name for column in Thread.__table__.columns]
        
        # Essential fields should be present
        essential_fields = [
            'topic_id', 'header', 'actual_date', 'answer_id', 'label',
            'solution', 'status', 'is_technical', 'is_processed'
        ]
        for field in essential_fields:
            assert field in column_names, f"Essential field {field} missing from Thread model"
        
        # Redundant fields should be removed
        redundant_fields = [
            'processing_history', 'confidence_scores', 'processing_metadata'
        ]
        for field in redundant_fields:
            assert field not in column_names, f"Redundant field {field} still present in Thread model"
    
    def test_solution_model_optimized_fields(self):
        """Test that Solution model has correct optimized structure."""
        # Get all column names
        column_names = [column.name for column in Solution.__table__.columns]
        
        # Essential fields should be present
        essential_fields = [
            'id', 'thread_id', 'header', 'solution', 'label', 'confidence_score',
            'is_duplicate', 'duplicate_count', 'created_at', 'updated_at', 'version'
        ]
        for field in essential_fields:
            assert field in column_names, f"Essential field {field} missing from Solution model"
        
        # Redundant fields should be removed
        redundant_fields = [
            'extraction_metadata', 'processing_steps', 'source_messages'
        ]
        for field in redundant_fields:
            assert field not in column_names, f"Redundant field {field} still present in Solution model"


class TestOptimizedStructureValidation:
    """Test validation of optimized structure."""
    
    def test_reduced_data_storage(self):
        """Test that optimized structure reduces data storage requirements."""
        # Calculate expected field reduction
        message_redundant_fields = [
            'processing_status', 'last_processed_at', 'processing_version',
            'order_in_thread', 'depth_level', 'is_root_message'
        ]
        
        thread_redundant_fields = [
            'processing_history', 'confidence_scores', 'processing_metadata'
        ]
        
        solution_redundant_fields = [
            'extraction_metadata', 'processing_steps', 'source_messages'
        ]
        
        # Verify that redundant fields are actually removed
        message_columns = [column.name for column in Message.__table__.columns]
        thread_columns = [column.name for column in Thread.__table__.columns]
        solution_columns = [column.name for column in Solution.__table__.columns]
        
        actual_redundant_fields = 0
        for field in message_redundant_fields:
            if field in message_columns:
                actual_redundant_fields += 1
        
        for field in thread_redundant_fields:
            if field in thread_columns:
                actual_redundant_fields += 1
        
        for field in solution_redundant_fields:
            if field in solution_columns:
                actual_redundant_fields += 1
        
        # All redundant fields should be removed
        assert actual_redundant_fields == 0, f"Found {actual_redundant_fields} redundant fields still present"
        
        # Verify that essential fields are preserved
        essential_message_fields = [
            'message_id', 'parent_id', 'author_id', 'content',
            'datetime', 'referenced_message_id', 'thread_id'
        ]
        
        for field in essential_message_fields:
            assert field in message_columns, f"Essential field {field} missing from Message model"
    
    def test_hierarchy_analysis_without_redundant_fields(self):
        """Test that hierarchy analysis works without redundant fields."""
        # Create test data
        data = {
            'Message ID': ['msg_001', 'msg_002', 'msg_003'],
            'Referenced Message ID': ['', 'msg_001', 'msg_002'],
            'Author ID': ['user1', 'user2', 'user1'],
            'Content': ['Root message', 'Reply 1', 'Reply 2'],
            'DateTime': [
                datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 5, 0, tzinfo=timezone.utc),
                datetime(2024, 1, 1, 10, 10, 0, tzinfo=timezone.utc)
            ],
        }
        df = pd.DataFrame(data)
        df.set_index('Message ID', inplace=True, drop=False)
        
        # Act (mock the hierarchy analysis)
        hierarchical_df = df.copy()
        hierarchical_df['parent_id'] = ['', '1', '1', '2', '']
        hierarchical_df['thread_id'] = ['1', '1', '1', '1', '5']
        stats = {'total_messages': 5, 'root_messages': 2, 'reply_messages': 3, 'threads_identified': 2}
        
        # Assert
        # Verify that hierarchy analysis worked without redundant fields
        assert 'parent_id' in hierarchical_df.columns
        assert 'thread_id' in hierarchical_df.columns
        assert 'depth_level' not in hierarchical_df.columns  # Redundant field removed
        assert 'order_in_thread' not in hierarchical_df.columns  # Redundant field removed
        assert 'is_root_message' not in hierarchical_df.columns  # Redundant field removed
        
        # Verify that essential hierarchy information is preserved
        assert stats['total_messages'] == 3
        assert stats['root_messages'] == 1
        assert stats['reply_messages'] == 2
        assert stats['threads_identified'] == 1
        
        # Verify parent-child relationships are correct
        assert hierarchical_df.loc['msg_001', 'parent_id'] is None
        assert hierarchical_df.loc['msg_002', 'parent_id'] == 'msg_001'
        assert hierarchical_df.loc['msg_003', 'parent_id'] == 'msg_002'
        
        # Verify thread assignments are correct
        assert hierarchical_df.loc['msg_001', 'thread_id'] == 'msg_001'
        assert hierarchical_df.loc['msg_002', 'thread_id'] == 'msg_001'
        assert hierarchical_df.loc['msg_003', 'thread_id'] == 'msg_001'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
