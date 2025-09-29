#!/usr/bin/env python3
"""
Example usage of YAML file utilities for handling dictionaries with File objects.
This demonstrates how to use the yaml_file_utils module in your application.
"""

from datetime import datetime, timezone, timedelta
from google.genai import types
from app.utils.yaml_file_utils import (
    save_yaml_genai_files, 
    load_yaml_genai_files, 
    cleanup_expired_files
)


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


def example_usage():
    """Demonstrate how to use the YAML file utilities."""
    print("📝 Example: YAML File Dictionary Usage")
    print("=" * 50)
    
    # Create a dictionary with File objects
    dict_uploaded_images = {
        "image_001": create_sample_file("test_image_001"),
        "image_002": create_sample_file("test_image_002"),
        "metadata": "Some metadata string",
        "count": 42
    }
    
    print(f"Created dictionary with {len(dict_uploaded_images)} items")
    print(f"File objects: {[k for k, v in dict_uploaded_images.items() if isinstance(v, types.File)]}")
    
    # Save to YAML file
    yaml_file_path = "example_dict_uploaded_images.yaml"
    success = save_yaml_genai_files(dict_uploaded_images, yaml_file_path)
    
    if success:
        print(f"✅ Successfully saved to {yaml_file_path}")
    else:
        print("❌ Failed to save YAML file")
        return
    
    # Load from YAML file
    loaded_dict = load_yaml_genai_files(yaml_file_path)
    
    if loaded_dict:
        print(f"✅ Successfully loaded from {yaml_file_path}")
        print(f"Loaded dictionary has {len(loaded_dict)} items")
        
        # Verify File objects
        file_objects = [k for k, v in loaded_dict.items() if isinstance(v, types.File)]
        print(f"File objects in loaded dict: {file_objects}")
        
        # Check properties of first file
        if file_objects:
            first_file_key = file_objects[0]
            first_file = loaded_dict[first_file_key]
            print(f"First file properties:")
            print(f"  Name: {first_file.name}")
            print(f"  URI: {first_file.uri}")
            print(f"  MIME type: {first_file.mime_type}")
            print(f"  Size: {first_file.size_bytes} bytes")
            print(f"  State: {first_file.state}")
            print(f"  Source: {first_file.source}")
        
        # Test cleanup of expired files
        print("\n🧹 Testing cleanup of expired files...")
        cleaned_dict = cleanup_expired_files(loaded_dict)
        print(f"Files before cleanup: {len([v for v in loaded_dict.values() if isinstance(v, types.File)])}")
        print(f"Files after cleanup: {len([v for v in cleaned_dict.values() if isinstance(v, types.File)])}")
        
    else:
        print("❌ Failed to load YAML file")
        return
    
    # Clean up example file
    import os
    if os.path.exists(yaml_file_path):
        os.remove(yaml_file_path)
        print(f"🧹 Cleaned up example file: {yaml_file_path}")
    
    print("\n🎉 Example completed successfully!")


if __name__ == "__main__":
    example_usage()
