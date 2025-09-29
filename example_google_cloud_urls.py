"""
Example demonstrating Google Cloud Storage standard URL conversion methods.
This shows the proper way to convert gs:// URLs to https://storage.googleapis.com/ URLs
using the official Google Cloud Storage Python client library.
"""

from google.cloud import storage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import google.auth
from google.oauth2 import service_account
GOOGLE_SERVICE_CREDENTIALS = os.environ.get("GOOGLE_SERVICE_CREDENTIALS")
credentials = service_account.Credentials.from_service_account_file(
    GOOGLE_SERVICE_CREDENTIALS)
# credentials, project = google.auth.default( )

def demonstrate_google_cloud_url_methods():
    """Demonstrate the standard Google Cloud Storage URL conversion methods."""
    
    # Initialize Google Cloud Storage client
    client = storage.Client()
    
    # Example bucket and blob names
    bucket_name = "discord_pics"  # From your config
    blob_name = "attachments/example_image.jpg"
    
    print("=== Google Cloud Storage Standard URL Methods ===\n")
    
    # Method 1: Using blob.public_url (RECOMMENDED)
    print("1. Using blob.public_url (Recommended):")
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    public_url = blob.public_url
    print(f"   Public URL: {public_url}")
    print(f"   Format: https://storage.googleapis.com/{bucket_name}/{blob_name}\n")
    
    # Method 2: Using blob.generate_signed_url for private objects
    print("2. Using blob.generate_signed_url (for private objects):")
    from datetime import datetime, timedelta
    expiration = datetime.utcnow() + timedelta(hours=1)
    signed_url = blob.generate_signed_url(expiration=expiration,credentials=credentials)
    print(f"   Signed URL: {signed_url[:100]}...")
    print(f"   Note: This URL expires in 1 hour\n")
    
    # Method 3: Manual construction (fallback)
    print("3. Manual construction (fallback method):")
    manual_url = f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
    print(f"   Manual URL: {manual_url}\n")
    
    # Method 4: Converting from gs:// format
    print("4. Converting from gs:// format:")
    gs_url = f"gs://{bucket_name}/{blob_name}"
    print(f"   Original gs:// URL: {gs_url}")
    
    # Parse and convert using Google Cloud client
    path_without_prefix = gs_url[5:]  # Remove 'gs://'
    bucket_name_parsed, blob_name_parsed = path_without_prefix.split('/', 1)
    bucket_parsed = client.bucket(bucket_name_parsed)
    blob_parsed = bucket_parsed.blob(blob_name_parsed)
    converted_url = blob_parsed.public_url
    print(f"   Converted URL: {converted_url}\n")
    
    print("=== Key Benefits of Using Google Cloud Client Methods ===")
    print("✓ Official Google Cloud Storage library methods")
    print("✓ Handles edge cases and special characters properly")
    print("✓ Supports both public and signed URLs")
    print("✓ Future-proof against API changes")
    print("✓ Better error handling and validation")

if __name__ == "__main__":
    demonstrate_google_cloud_url_methods()

