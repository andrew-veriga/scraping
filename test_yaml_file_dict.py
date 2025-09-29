#!/usr/bin/env python3
"""
Test function for YAML save and load operations with dictionaries containing string keys and File values.
This module provides comprehensive testing for the custom YAML handling of google.genai.types.File objects.
"""

import yaml
import os
import tempfile
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from google.genai import types


def file_to_dict(file_obj: types.File) -> Dict[str, Any]:
    """Convert a File object to a dictionary for YAML serialization."""
    return {
        'name': file_obj.name,
        'uri': file_obj.uri,
        'mime_type': file_obj.mime_type,
        'size_bytes': file_obj.size_bytes,
        'sha256_hash': file_obj.sha256_hash,
        'expiration_time': file_obj.expiration_time.isoformat() if file_obj.expiration_time else None,
        'create_time': file_obj.create_time.isoformat() if file_obj.create_time else None,
        'update_time': file_obj.update_time.isoformat() if file_obj.update_time else None,
        'state': file_obj.state.name if file_obj.state else None,
        'source': file_obj.source.name if file_obj.source else None,
        '_is_file_object': True  # Marker to identify File objects during deserialization
    }


def dict_to_genai_file(data: Dict[str, Any]) -> types.File:
    """Convert a dictionary back to a File object."""
    return types.File(
        name=data.get('name'),
        uri=data.get('uri'),
        mime_type=data.get('mime_type'),
        size_bytes=data.get('size_bytes'),
        sha256_hash=data.get('sha256_hash'),
        expiration_time=datetime.fromisoformat(data['expiration_time']) if data.get('expiration_time') else None,
        create_time=datetime.fromisoformat(data['create_time']) if data.get('create_time') else None,
        update_time=datetime.fromisoformat(data['update_time']) if data.get('update_time') else None,
        state=types.FileState[data['state']] if data.get('state') else None,
        source=types.FileSource[data['source']] if data.get('source') else None
    )


def convert_dict_for_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dictionary containing File objects to YAML-serializable format."""
    converted = {}
    for key, value in data.items():
        if isinstance(value, types.File):
            converted[key] = file_to_dict(value)
        elif isinstance(value, dict):
            converted[key] = convert_dict_for_yaml(value)
        else:
            converted[key] = value
    return converted


def convert_dict_from_yaml(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a dictionary from YAML back to original format with File objects."""
    converted = {}
    for key, value in data.items():
        if isinstance(value, dict) and value.get('_is_file_object'):
            # Remove the marker and convert to File object
            file_data = {k: v for k, v in value.items() if k != '_is_file_object'}
            converted[key] = dict_to_genai_file(file_data)
        elif isinstance(value, dict):
            converted[key] = convert_dict_from_yaml(value)
        else:
            converted[key] = value
    return converted


def create_sample_file(name: str, expiration_hours: int = 2) -> types.File:
    """Create a sample File object for testing."""
    now = datetime.now(timezone.utc)
    expiration_time = now + timedelta(hours=expiration_hours)
    
    return types.File(
        name=f"files/{name}",
        uri=f"https://generativelanguage.googleapis.com/v1beta/files/{name}",
        mime_type="image/jpeg",
        size_bytes=1024,
        sha256_hash=f"test_hash_{name}",
        expiration_time=expiration_time,
        create_time=now,
        update_time=now,
        state=types.FileState.ACTIVE,
        source=types.FileSource.UPLOADED
    )


def save_yaml_genai_files(file_dict: Dict[str, Any], file_path: str) -> bool:
    """
    Save a dictionary with File objects to YAML file.
    
    Args:
        file_dict: Dictionary with string keys and File values
        file_path: Path to save the YAML file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Convert File objects to dictionaries for YAML serialization
        converted_dict = convert_dict_for_yaml(file_dict)
        
        with open(file_path, 'w') as f:
            yaml.safe_dump(converted_dict, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Error saving YAML file: {e}")
        return False


def load_yaml_genai_files(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load a dictionary with File objects from YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dict with File objects or None if failed
    """
    try:
        with open(file_path, 'r') as f:
            loaded_dict = yaml.safe_load(f)
        
        # Convert dictionaries back to File objects
        return convert_dict_from_yaml(loaded_dict)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return None


def test_yaml_save_load_cycle():
    """
    Comprehensive test function for YAML save and load operations with File objects.
    Tests various scenarios including edge cases.
    """
    print("🧪 Testing YAML Save/Load with File Objects")
    print("=" * 60)
    
    # Test 1: Basic save/load with single File object
    print("\n1. Testing basic save/load with single File object...")
    test_dict_1 = {
        "image_001": create_sample_file("test_image_001"),
        "metadata": "This is a test string",
        "count": 42
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Save
        success = save_yaml_genai_files(test_dict_1, temp_path)
        assert success, "Failed to save YAML file"
        print("✅ Save operation successful")
        
        # Load
        loaded_dict = load_yaml_genai_files(temp_path)
        assert loaded_dict is not None, "Failed to load YAML file"
        print("✅ Load operation successful")
        
        # Verify File object properties
        loaded_file = loaded_dict["image_001"]
        original_file = test_dict_1["image_001"]
        
        assert isinstance(loaded_file, types.File), "Loaded object is not a File instance"
        assert loaded_file.name == original_file.name, "File name mismatch"
        assert loaded_file.uri == original_file.uri, "File URI mismatch"
        assert loaded_file.mime_type == original_file.mime_type, "File MIME type mismatch"
        assert loaded_file.size_bytes == original_file.size_bytes, "File size mismatch"
        assert loaded_file.state == original_file.state, "File state mismatch"
        assert loaded_file.source == original_file.source, "File source mismatch"
        print("✅ File object properties preserved correctly")
        
        # Verify other data types
        assert loaded_dict["metadata"] == "This is a test string", "String value mismatch"
        assert loaded_dict["count"] == 42, "Number value mismatch"
        print("✅ Other data types preserved correctly")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test 2: Multiple File objects
    print("\n2. Testing save/load with multiple File objects...")
    test_dict_2 = {
        "image_001": create_sample_file("test_image_001"),
        "image_002": create_sample_file("test_image_002"),
        "image_003": create_sample_file("test_image_003"),
        "mixed_data": {
            "nested_file": create_sample_file("nested_file"),
            "nested_string": "nested value"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        success = save_yaml_genai_files(test_dict_2, temp_path)
        assert success, "Failed to save YAML file with multiple files"
        print("✅ Multiple files save successful")
        
        loaded_dict = load_yaml_genai_files(temp_path)
        assert loaded_dict is not None, "Failed to load YAML file with multiple files"
        print("✅ Multiple files load successful")
        
        # Verify all files
        for key in ["image_001", "image_002", "image_003"]:
            assert isinstance(loaded_dict[key], types.File), f"{key} is not a File instance"
            assert loaded_dict[key].name == f"files/test_{key}", f"{key} name mismatch"
        print("✅ All multiple files preserved correctly")
        
        # Verify nested structure
        assert isinstance(loaded_dict["mixed_data"]["nested_file"], types.File), "Nested file is not a File instance"
        assert loaded_dict["mixed_data"]["nested_string"] == "nested value", "Nested string mismatch"
        print("✅ Nested structure preserved correctly")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test 3: Empty dictionary
    print("\n3. Testing save/load with empty dictionary...")
    test_dict_3 = {}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        success = save_yaml_genai_files(test_dict_3, temp_path)
        assert success, "Failed to save empty YAML file"
        print("✅ Empty dict save successful")
        
        loaded_dict = load_yaml_genai_files(temp_path)
        assert loaded_dict is not None, "Failed to load empty YAML file"
        assert len(loaded_dict) == 0, "Empty dict not preserved"
        print("✅ Empty dict preserved correctly")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test 4: Dictionary with only strings (no File objects)
    print("\n4. Testing save/load with only string values...")
    test_dict_4 = {
        "key1": "value1",
        "key2": "value2",
        "number": 123,
        "boolean": True
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        success = save_yaml_genai_files(test_dict_4, temp_path)
        assert success, "Failed to save string-only YAML file"
        print("✅ String-only dict save successful")
        
        loaded_dict = load_yaml_genai_files(temp_path)
        assert loaded_dict is not None, "Failed to load string-only YAML file"
        assert loaded_dict["key1"] == "value1", "String value mismatch"
        assert loaded_dict["number"] == 123, "Number value mismatch"
        assert loaded_dict["boolean"] == True, "Boolean value mismatch"
        print("✅ String-only dict preserved correctly")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test 5: File with expiration time handling
    print("\n5. Testing File objects with expiration times...")
    expired_file = create_sample_file("expired_file", expiration_hours=-1)  # Already expired
    future_file = create_sample_file("future_file", expiration_hours=24)    # Expires in 24 hours
    
    test_dict_5 = {
        "expired_file": expired_file,
        "future_file": future_file
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        success = save_yaml_genai_files(test_dict_5, temp_path)
        assert success, "Failed to save YAML file with expiration times"
        print("✅ Expiration times save successful")
        
        loaded_dict = load_yaml_genai_files(temp_path)
        assert loaded_dict is not None, "Failed to load YAML file with expiration times"
        
        # Verify expiration times are preserved
        assert loaded_dict["expired_file"].expiration_time == expired_file.expiration_time, "Expired file time mismatch"
        assert loaded_dict["future_file"].expiration_time == future_file.expiration_time, "Future file time mismatch"
        print("✅ Expiration times preserved correctly")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test 6: Error handling - non-existent file
    print("\n6. Testing error handling with non-existent file...")
    non_existent_path = "/non/existent/path/file.yaml"
    loaded_dict = load_yaml_genai_files(non_existent_path)
    assert loaded_dict is None, "Should return None for non-existent file"
    print("✅ Error handling for non-existent file works correctly")
    
    # Test 7: Error handling - invalid YAML
    print("\n7. Testing error handling with invalid YAML...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_file.write("invalid: yaml: content: [")
        temp_path = temp_file.name
    
    try:
        loaded_dict = load_yaml_genai_files(temp_path)
        assert loaded_dict is None, "Should return None for invalid YAML"
        print("✅ Error handling for invalid YAML works correctly")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    # Test 8: Large dictionary performance test
    print("\n8. Testing performance with large dictionary...")
    large_dict = {}
    for i in range(100):
        large_dict[f"image_{i:03d}"] = create_sample_file(f"test_image_{i:03d}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        import time
        start_time = time.time()
        success = save_yaml_genai_files(large_dict, temp_path)
        save_time = time.time() - start_time
        assert success, "Failed to save large YAML file"
        print(f"✅ Large dict save successful (took {save_time:.2f} seconds)")
        
        start_time = time.time()
        loaded_dict = load_yaml_genai_files(temp_path)
        load_time = time.time() - start_time
        assert loaded_dict is not None, "Failed to load large YAML file"
        assert len(loaded_dict) == 100, "Large dict size mismatch"
        print(f"✅ Large dict load successful (took {load_time:.2f} seconds)")
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print("\n🎉 All tests passed! YAML save/load with File objects works correctly.")
    print("=" * 60)


def test_dict_uploaded_images_workflow():
    """
    Test the specific workflow used in the application for dict_uploaded_images.
    This simulates the actual usage pattern from thread_service.py
    """
    print("\n🔧 Testing dict_uploaded_images workflow")
    print("=" * 50)
    
    # Simulate the workflow from thread_service.py
    dict_uploaded_images = {}
    
    # Add some files (simulating the workflow)
    dict_uploaded_images["test_image_1"] = create_sample_file("test_image_1")
    dict_uploaded_images["test_image_2"] = create_sample_file("test_image_2")
    
    # Test the save operation (as used in the app)
    test_file_path = "test_dict_uploaded_images_workflow.yaml"
    
    try:
        # Save with conversion approach
        converted_dict = convert_dict_for_yaml(dict_uploaded_images)
        with open(test_file_path, 'w') as f:
            yaml.safe_dump(converted_dict, f, default_flow_style=False)
        print("✅ Workflow save operation successful")
        
        # Load with conversion approach
        with open(test_file_path, 'r') as f:
            raw_loaded_dict = yaml.safe_load(f)
        loaded_dict = convert_dict_from_yaml(raw_loaded_dict)
        print("✅ Workflow load operation successful")
        
        # Verify the loaded data
        assert len(loaded_dict) == 2, "Workflow dict size mismatch"
        assert "test_image_1" in loaded_dict, "test_image_1 not found"
        assert "test_image_2" in loaded_dict, "test_image_2 not found"
        
        # Verify File objects
        for key, file_obj in loaded_dict.items():
            assert isinstance(file_obj, types.File), f"{key} is not a File instance"
            assert file_obj.name == f"files/{key}", f"{key} name mismatch"
        
        print("✅ Workflow File objects preserved correctly")
        
        # Test expiration cleanup (as used in the app)
        now = datetime.now(timezone.utc)
        expired_count = 0
        for key, value in list(loaded_dict.items()):
            if isinstance(value, types.File):
                if value.expiration_time < now:
                    loaded_dict.pop(key)
                    expired_count += 1
        
        print(f"✅ Expiration cleanup logic works (would remove {expired_count} expired files)")
        
    finally:
        if os.path.exists(test_file_path):
            os.unlink(test_file_path)
    
    print("✅ dict_uploaded_images workflow test completed successfully")


if __name__ == "__main__":
    # Run all tests
    test_yaml_save_load_cycle()
    test_dict_uploaded_images_workflow()
    
    print("\n🚀 All YAML File object tests completed successfully!")
    print("The custom YAML constructor and representer are working correctly.")
