"""
Utility functions for converting Google Cloud Storage URLs between different formats.
Uses Google Cloud Storage client library standard methods.
"""

import re
from typing import List, Union
from google.cloud import storage


def gs_to_https_url(gs_url: str, client: storage.Client = None) -> str:
    """
    Convert a gs:// URL to https://storage.googleapis.com/ URL format using Google Cloud client.
    
    Args:
        gs_url (str): URL in gs://bucket-name/path format
        client (storage.Client, optional): Google Cloud Storage client instance
        
    Returns:
        str: URL in https://storage.googleapis.com/bucket-name/path format
        
    Example:
        gs://my-bucket/folder/file.jpg -> https://storage.googleapis.com/my-bucket/folder/file.jpg
    """
    if not gs_url.startswith('gs://'):
        return gs_url  # Return as-is if not a gs:// URL
    
    # Use Google Cloud client if available for more robust parsing
    if client:
        try:
            # Parse the gs:// URL to get bucket and blob name
            path_without_prefix = gs_url[5:]  # Remove 'gs://'
            if '/' in path_without_prefix:
                bucket_name, blob_name = path_without_prefix.split('/', 1)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                return blob.public_url
            else:
                # Just bucket name, return bucket URL
                return f"https://storage.googleapis.com/{path_without_prefix}"
        except Exception:
            # Fall back to simple string replacement if client method fails
            pass
    
    # Fallback: simple string replacement
    path_without_prefix = gs_url[5:]  # Remove 'gs://'
    return f"https://storage.googleapis.com/{path_without_prefix}"


def https_to_gs_url(https_url: str) -> str:
    """
    Convert a https://storage.googleapis.com/ URL to gs:// URL format.
    
    Args:
        https_url (str): URL in https://storage.googleapis.com/bucket-name/path format
        
    Returns:
        str: URL in gs://bucket-name/path format
        
    Example:
        https://storage.googleapis.com/my-bucket/folder/file.jpg -> gs://my-bucket/folder/file.jpg
    """
    if not https_url.startswith('https://storage.googleapis.com/'):
        return https_url  # Return as-is if not a storage.googleapis.com URL
    
    # Remove https://storage.googleapis.com/ prefix and add gs://
    path_without_prefix = https_url[31:]  # Remove 'https://storage.googleapis.com/'
    return f"gs://{path_without_prefix}"


def convert_attachment_urls(attachments: str, to_format: str = 'https') -> str:
    """
    Convert attachment URLs in a semicolon-separated string between gs:// and https formats.
    
    Args:
        attachments (str): Semicolon-separated string of URLs
        to_format (str): Target format - 'https' or 'gs' (default: 'https')
        
    Returns:
        str: Converted URLs in the same semicolon-separated format
    """
    if attachments == 'No attachments':
        return attachments
    
    urls = attachments.split(';')
    converted_urls = []
    
    for url in urls:
        url = url.strip()
        if to_format.lower() == 'https':
            converted_url = gs_to_https_url(url)
        elif to_format.lower() == 'gs':
            converted_url = https_to_gs_url(url)
        else:
            raise ValueError("to_format must be 'https' or 'gs'")
        
        converted_urls.append(converted_url)
    
    return ';'.join(converted_urls)


def batch_convert_urls(urls: List[str], to_format: str = 'https') -> List[str]:
    """
    Convert a list of URLs between gs:// and https formats.
    
    Args:
        urls (List[str]): List of URLs to convert
        to_format (str): Target format - 'https' or 'gs' (default: 'https')
        
    Returns:
        List[str]: List of converted URLs
    """
    converted_urls = []
    
    for url in urls:
        if to_format.lower() == 'https':
            converted_url = gs_to_https_url(url)
        elif to_format.lower() == 'gs':
            converted_url = https_to_gs_url(url)
        else:
            raise ValueError("to_format must be 'https' or 'gs'")
        
        converted_urls.append(converted_url)
    
    return converted_urls


def is_gs_url(url: str) -> bool:
    """Check if a URL is in gs:// format."""
    return url.startswith('gs://')


def is_storage_https_url(url: str) -> bool:
    """Check if a URL is in https://storage.googleapis.com/ format."""
    return url.startswith('https://storage.googleapis.com/')


def get_blob_public_url(bucket_name: str, blob_name: str, client: storage.Client = None) -> str:
    """
    Get the public URL for a blob using Google Cloud Storage client.
    This is the recommended way to get public URLs.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        blob_name (str): Name/path of the blob in the bucket
        client (storage.Client, optional): Google Cloud Storage client instance
        
    Returns:
        str: Public URL in https://storage.googleapis.com/ format
    """
    if client:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.public_url
    else:
        # Fallback without client
        return f"https://storage.googleapis.com/{bucket_name}/{blob_name}"


def get_blob_signed_url(bucket_name: str, blob_name: str, expiration_minutes: int = 60, client: storage.Client = None) -> str:
    """
    Get a signed URL for a blob using Google Cloud Storage client.
    Useful for private objects that need temporary access.
    
    Args:
        bucket_name (str): Name of the GCS bucket
        blob_name (str): Name/path of the blob in the bucket
        expiration_minutes (int): How long the signed URL should be valid (default: 60 minutes)
        client (storage.Client, optional): Google Cloud Storage client instance
        
    Returns:
        str: Signed URL for temporary access
    """
    if not client:
        raise ValueError("Google Cloud Storage client is required for signed URLs")
    
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    from datetime import datetime, timedelta
    expiration = datetime.utcnow() + timedelta(minutes=expiration_minutes)
    
    return blob.generate_signed_url(expiration=expiration)


def revert_public_urls_to_gs(attachments: str) -> str:
    """
    REVERTING FUNCTION: Convert public https://storage.googleapis.com/ URLs back to gs:// URLs.
    This is the reverse operation of gs_to_https_url.
    
    Args:
        attachments (str): Semicolon-separated string of URLs (may contain public URLs)
        
    Returns:
        str: Converted URLs with public URLs reverted to gs:// format
        
    Example:
        "https://storage.googleapis.com/bucket/file.jpg;gs://bucket/file2.jpg" 
        -> "gs://bucket/file.jpg;gs://bucket/file2.jpg"
    """
    if attachments == 'No attachments':
        return attachments
    
    urls = attachments.split(';')
    converted_urls = []
    
    for url in urls:
        url = url.strip()
        if url.startswith('https://storage.googleapis.com/'):
            # Convert public URL back to gs:// format
            converted_url = https_to_gs_url(url)
            converted_urls.append(converted_url)
        else:
            # Keep non-public URLs as is (including gs:// URLs)
            converted_urls.append(url)
    
    return ';'.join(converted_urls)


def batch_revert_public_urls(urls: List[str]) -> List[str]:
    """
    REVERTING FUNCTION: Convert a list of public URLs back to gs:// URLs.
    
    Args:
        urls (List[str]): List of URLs (may contain public URLs)
        
    Returns:
        List[str]: List of URLs with public URLs reverted to gs:// format
    """
    converted_urls = []
    
    for url in urls:
        if url.startswith('https://storage.googleapis.com/'):
            converted_url = https_to_gs_url(url)
            converted_urls.append(converted_url)
        else:
            converted_urls.append(url)
    
    return converted_urls


def validate_gs_url(gs_url: str) -> bool:
    """
    Validate that a gs:// URL has the correct format.
    
    Args:
        gs_url (str): URL to validate
        
    Returns:
        bool: True if valid gs:// URL format, False otherwise
    """
    if not gs_url.startswith('gs://'):
        return False
    
    # Remove gs:// prefix
    path = gs_url[5:]
    
    # Should have at least bucket name
    if not path or '/' not in path:
        return False
    
    # Split bucket and blob name
    parts = path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else ""
    
    # Basic validation
    if not bucket_name or not blob_name:
        return False
    
    return True


def validate_public_url(public_url: str) -> bool:
    """
    Validate that a public URL has the correct storage.googleapis.com format.
    
    Args:
        public_url (str): URL to validate
        
    Returns:
        bool: True if valid public URL format, False otherwise
    """
    if not public_url.startswith('https://storage.googleapis.com/'):
        return False
    
    # Remove prefix
    path = public_url[31:]  # Remove 'https://storage.googleapis.com/'
    
    # Should have at least bucket name
    if not path or '/' not in path:
        return False
    
    return True
