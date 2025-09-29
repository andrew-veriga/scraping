#!/usr/bin/env python3
"""
Example script demonstrating how to use the URL reverting functions.
This script shows how to convert public URLs back to gs:// URLs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.utils.url_converter import (
    revert_public_urls_to_gs, 
    batch_revert_public_urls,
    validate_gs_url,
    validate_public_url,
    https_to_gs_url
)


def demonstrate_url_reverting():
    """Demonstrate the URL reverting functionality."""
    
    print("=== URL Reverting Function Demonstration ===\n")
    
    # Example 1: Single URL conversion
    print("1. Single URL Conversion:")
    public_url = "https://storage.googleapis.com/discord_pics/attachments/abc123_image.jpg"
    gs_url = https_to_gs_url(public_url)
    print(f"   Public URL: {public_url}")
    print(f"   Reverted to: {gs_url}")
    print(f"   Valid gs:// URL: {validate_gs_url(gs_url)}\n")
    
    # Example 2: Multiple URLs in semicolon-separated string
    print("2. Multiple URLs (semicolon-separated):")
    attachments = "https://storage.googleapis.com/discord_pics/attachments/file1.jpg;https://storage.googleapis.com/discord_pics/attachments/file2.png;gs://discord_pics/attachments/file3.gif"
    reverted = revert_public_urls_to_gs(attachments)
    print(f"   Original: {attachments}")
    print(f"   Reverted: {reverted}\n")
    
    # Example 3: Batch processing
    print("3. Batch Processing:")
    urls = [
        "https://storage.googleapis.com/discord_pics/attachments/image1.jpg",
        "gs://discord_pics/attachments/image2.jpg",  # Already gs:// format
        "https://storage.googleapis.com/discord_pics/attachments/image3.png"
    ]
    reverted_urls = batch_revert_public_urls(urls)
    print("   Original URLs:")
    for i, url in enumerate(urls, 1):
        print(f"     {i}. {url}")
    print("   Reverted URLs:")
    for i, url in enumerate(reverted_urls, 1):
        print(f"     {i}. {url}")
    print()
    
    # Example 4: Validation
    print("4. URL Validation:")
    test_urls = [
        "gs://discord_pics/attachments/valid_file.jpg",
        "gs://invalid",  # Missing blob name
        "https://storage.googleapis.com/discord_pics/attachments/valid_file.jpg",
        "https://invalid-url.com/file.jpg"  # Not a storage URL
    ]
    
    for url in test_urls:
        is_gs = validate_gs_url(url)
        is_public = validate_public_url(url)
        print(f"   {url}")
        print(f"     Valid gs:// URL: {is_gs}")
        print(f"     Valid public URL: {is_public}")
        print()


def test_edge_cases():
    """Test edge cases for URL reverting."""
    
    print("=== Edge Cases Testing ===\n")
    
    # Edge case 1: No attachments
    print("1. No attachments:")
    result = revert_public_urls_to_gs("No attachments")
    print(f"   Input: 'No attachments'")
    print(f"   Output: '{result}'\n")
    
    # Edge case 2: Empty string
    print("2. Empty string:")
    result = revert_public_urls_to_gs("")
    print(f"   Input: ''")
    print(f"   Output: '{result}'\n")
    
    # Edge case 3: Mixed formats with whitespace
    print("3. Mixed formats with whitespace:")
    attachments = "  https://storage.googleapis.com/discord_pics/attachments/file1.jpg  ;  gs://discord_pics/attachments/file2.jpg  "
    result = revert_public_urls_to_gs(attachments)
    print(f"   Input: '{attachments}'")
    print(f"   Output: '{result}'\n")
    
    # Edge case 4: Non-storage URLs
    print("4. Non-storage URLs:")
    attachments = "https://example.com/image.jpg;https://storage.googleapis.com/discord_pics/attachments/file.jpg"
    result = revert_public_urls_to_gs(attachments)
    print(f"   Input: '{attachments}'")
    print(f"   Output: '{result}'\n")


if __name__ == "__main__":
    demonstrate_url_reverting()
    test_edge_cases()
    
    print("=== Usage Instructions ===")
    print("To revert URLs in your Excel file, use:")
    print("  from app.services.extract_pics import convert_public_to_gs_urls_in_excel")
    print("  convert_public_to_gs_urls_in_excel()")
    print()
    print("Or for individual URL strings:")
    print("  from app.utils.url_converter import revert_public_urls_to_gs")
    print("  reverted = revert_public_urls_to_gs('your;urls;here')")
