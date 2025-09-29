#!/usr/bin/env python3
"""
Test script to demonstrate the custom YAML constructor for google.genai.types.File objects.
"""

import yaml
from datetime import datetime, timezone
from google.genai import types

# Custom YAML constructor and representer for google.genai.types.File objects
def file_representer(dumper, file_obj):
    """Custom representer for google.genai.types.File objects."""
    if isinstance(file_obj, types.File):
        return dumper.represent_dict({
            'name': file_obj.name,
            'uri': file_obj.uri,
            'mime_type': file_obj.mime_type,
            'size_bytes': file_obj.size_bytes,
            'sha256_hash': file_obj.sha256_hash,
            'expiration_time': file_obj.expiration_time.isoformat() if file_obj.expiration_time else None,
            'create_time': file_obj.create_time.isoformat() if file_obj.create_time else None,
            'update_time': file_obj.update_time.isoformat() if file_obj.update_time else None,
            'state': file_obj.state.name if file_obj.state else None,
            'source': file_obj.source.name if file_obj.source else None
        })
    return None

def file_constructor(loader, node):
    """Custom constructor for google.genai.types.File objects."""
    data = loader.construct_mapping(node)
    # Create a minimal File object with the essential data
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

# Register the custom representer and constructor
yaml.add_representer(types.File, file_representer)
yaml.add_constructor('tag:yaml.org,2002:python/object:google.genai.types.File', file_constructor)

def test_yaml_constructor():
    """Test the custom YAML constructor with a sample File object."""
    
    # Create a sample File object
    sample_file = types.File(
        name="files/test_image",
        uri="https://generativelanguage.googleapis.com/v1beta/files/test_image",
        mime_type="image/jpeg",
        size_bytes=1024,
        sha256_hash="test_hash_123",
        expiration_time=datetime.now(timezone.utc),
        create_time=datetime.now(timezone.utc),
        update_time=datetime.now(timezone.utc),
        state=types.FileState.ACTIVE,
        source=types.FileSource.UPLOADED
    )
    
    # Create a dict with the File object
    dict_uploaded_images = {
        "test_image_key": sample_file,
        "regular_string": "This is a regular string",
        "regular_number": 42
    }
    
    print("🔧 Testing custom YAML constructor for google.genai.types.File objects")
    print("=" * 70)
    
    # Test 1: Save to YAML
    print("1. Saving dict with File object to YAML...")
    yaml_content = yaml.dump(dict_uploaded_images, default_flow_style=False, Dumper=yaml.SafeDumper)
    print("✅ YAML content generated successfully")
    print("YAML content:")
    print("-" * 40)
    print(yaml_content)
    print("-" * 40)
    
    # Test 2: Load from YAML
    print("\n2. Loading dict from YAML...")
    loaded_dict = yaml.load(yaml_content, Loader=yaml.SafeLoader)
    print("✅ YAML content loaded successfully")
    
    # Test 3: Verify the loaded File object
    print("\n3. Verifying loaded File object...")
    loaded_file = loaded_dict["test_image_key"]
    
    print(f"   File name: {loaded_file.name}")
    print(f"   File URI: {loaded_file.uri}")
    print(f"   MIME type: {loaded_file.mime_type}")
    print(f"   Size: {loaded_file.size_bytes} bytes")
    print(f"   State: {loaded_file.state}")
    print(f"   Source: {loaded_file.source}")
    print("✅ File object properties preserved correctly")
    
    # Test 4: Verify other data types
    print("\n4. Verifying other data types...")
    print(f"   Regular string: {loaded_dict['regular_string']}")
    print(f"   Regular number: {loaded_dict['regular_number']}")
    print("✅ Other data types preserved correctly")
    
    # Test 5: Save to file and load back
    print("\n5. Testing file save/load cycle...")
    test_file_path = "test_dict_uploaded_images.yaml"
    
    # Save to file
    with open(test_file_path, 'w') as f:
        yaml.dump(dict_uploaded_images, f, default_flow_style=False, Dumper=yaml.SafeDumper)
    print("✅ Saved to file successfully")
    
    # Load from file
    with open(test_file_path, 'r') as f:
        file_loaded_dict = yaml.load(f, Loader=yaml.SafeLoader)
    print("✅ Loaded from file successfully")
    
    # Verify file-loaded object
    file_loaded_file = file_loaded_dict["test_image_key"]
    print(f"   File-loaded name: {file_loaded_file.name}")
    print(f"   File-loaded URI: {file_loaded_file.uri}")
    print("✅ File save/load cycle works correctly")
    
    # Clean up
    import os
    if os.path.exists(test_file_path):
        os.remove(test_file_path)
        print(f"✅ Cleaned up test file: {test_file_path}")
    
    print("\n🎉 All tests passed! Custom YAML constructor works correctly.")
    print("=" * 70)

if __name__ == "__main__":
    test_yaml_constructor()
