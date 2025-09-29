#!/usr/bin/env python3
"""
Utility functions for YAML save/load operations with google.genai.types.File objects.
This module provides functions to handle dictionaries containing File objects in YAML format.
"""

import yaml
from datetime import datetime, timezone
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
    if data is None:
        return converted
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


def cleanup_expired_files(file_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove expired File objects from a dictionary.
    
    Args:
        file_dict: Dictionary containing File objects
        
    Returns:
        Dictionary with expired files removed
    """
    now = datetime.now(timezone.utc)
    cleaned_dict = {}
    
    for key, value in file_dict.items():
        if isinstance(value, types.File):
            if value.expiration_time and value.expiration_time > now:
                cleaned_dict[key] = value
            # Skip expired files
        else:
            cleaned_dict[key] = value
    
    return cleaned_dict
