from dotenv import load_dotenv
from app.services import data_loader
import yaml
import os
import requests
import uuid
from google.cloud import storage
from datetime import timezone
with open("configs/config.yaml", 'r') as stream:
    config = yaml.safe_load(stream)

MESSAGES_FILE_PATH = 'C:\VSCode\scraping\data\discord_messages_fresh_pics.xlsx'
SAVE_PATH = config['images']['base_path']

# Initialize Google Cloud Storage client
load_dotenv()
gcs_client = storage.Client()
gcs_bucket_name = config['images']['gcs_bucket_name']
gcs_bucket = gcs_client.bucket(gcs_bucket_name)

def extract_pics():
    os.makedirs(SAVE_PATH, exist_ok=True)

    # load full list of messages and authors from file

    messages_df, _ = data_loader.load_and_preprocess_data(MESSAGES_FILE_PATH)
    for _, message in messages_df.iterrows():
        if message['Attachments'] != 'No attachments':
            attachment = message['Attachments']
            attachment_urls = attachment.split(';')
            if len(attachment_urls) > 1:
                print(f"Multiple attachments found in message {message['Message ID']}, processing each one.")
            saved_paths = []
            flag = False
            for attachment_url in attachment_urls:
                shrink_name = os.path.basename(attachment_url).split('?')[0]
                # if shrink_name == 'e1ddda42c970862f422f4b64d6c0e378.jpg':
                #     flag = True
                # if not flag:
                #     continue

                random_prefix = str(uuid.uuid4().hex[:16])  # Generate an 16-character random prefix
                shrink_name = f"{random_prefix}-{shrink_name}"
                file_name = os.path.join(SAVE_PATH, shrink_name)
                try:
                    response = requests.get(attachment_url, stream=True)
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    
                    with open(file_name, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"Downloaded {attachment_url} to {file_name}")

                    # Upload to Google Cloud Storage
                    blob = gcs_bucket.blob(f"attachments/{shrink_name}")
                    blob.upload_from_filename(file_name)
                    
                    # Get the public URL using Google Cloud Storage client's built-in method
                    public_url = blob.public_url
                    gs_url = f"gs://{gcs_bucket_name}/attachments/{shrink_name}"
                    
                    print(f"Uploaded {file_name} to {gs_url}")
                    print(f"Public URL: {public_url}")

                except requests.exceptions.RequestException as e:
                    print(f"Error downloading {attachment_url}: {e}")
                except Exception as e:
                    print(f"Error uploading {file_name} to GCS: {e}")
                
                # Store the public URL using Google Cloud Storage client's built-in method
                blob = gcs_bucket.blob(f"attachments/{shrink_name}")
                public_url = blob.public_url
                saved_paths.append(public_url)
            if len(saved_paths) > 0:
                messages_df.loc[messages_df['Message ID'] == message['Message ID'], 'Attachments'] = ';'.join(saved_paths)

    # Convert to timezone-naive for Excel compatibility, but keep original timezone-aware version
    messages_df_excel = messages_df.copy()
    messages_df_excel['DateTime'] = messages_df_excel['DateTime'].dt.tz_localize(timezone.utc)
    id_columns = ['Author ID', 'Referenced Message ID', 'Parent ID', 'Thread ID']
    for col in id_columns:
        if col in messages_df_excel.columns:
            messages_df_excel[col] = "'" + messages_df_excel[col].astype(str)
    messages_df_excel['Message ID'] = "'" + messages_df_excel['Message ID'].astype(str)
    
    messages_df_excel.to_excel('.\\data\\discord_messages_with_gcs_pics.xlsx', engine='openpyxl', index=False)

def delete_old_files(target_date_str = '2025-09-23 18:17'):
    import datetime

    files = os.listdir(SAVE_PATH)
    root_dir = SAVE_PATH
    # target_date_str = '2025-09-23 18:17'
    target_date = datetime.datetime.strptime(target_date_str, '%Y-%m-%d %H:%M')
    print(f"Searching for files older than {target_date_str} in '{root_dir}'...")

    older_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            modification_time = os.path.getmtime(file_path)
            if modification_time < target_date.timestamp():
                print(f"{filename}\t\t\t\t", datetime.datetime.fromtimestamp(modification_time))
                older_files.append(file_path)
    print(f"Total files older than {target_date_str}: {len(older_files)}")

    for f in older_files:
        os.remove(f)
        print(f"Deleted local file: {f}")

    gcs_bucket.delete_blobs([gcs_bucket.blob(f"attachments/{os.path.basename(f)}") for f in older_files])


def convert_gs_to_public_urls_in_excel():
    """
    TEMPORARY FUNCTION: Convert gs:// URLs to public https URLs in the Excel file.
    This function reads the Excel file, converts all gs:// URLs to public URLs,
    and saves the updated file. Use this AFTER making the bucket public.
    """
    import pandas as pd
    
    excel_file_path = '.\\data\\discord_messages_with_gcs_pics.xlsx'
    
    print(f"Loading Excel file: {excel_file_path}")
    
    # Load the Excel file
    try:
        messages_df = pd.read_excel(excel_file_path)
        print(f"Loaded {len(messages_df)} rows from Excel file")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Track conversions
    total_conversions = 0
    rows_updated = 0
    
    print("Converting gs:// URLs to public URLs...")
    
    # Process each row
    for index, row in messages_df.iterrows():
        if row['Attachments'] != 'No attachments':
            attachment_urls = row['Attachments'].split(';')
            converted_urls = []
            row_has_conversions = False
            
            for url in attachment_urls:
                url = url.strip()
                if url.startswith('gs://'):
                    try:
                        # Parse gs:// URL to get bucket and blob name
                        path_without_prefix = url[5:]  # Remove 'gs://'
                        if '/' in path_without_prefix:
                            bucket_name, blob_name = path_without_prefix.split('/', 1)
                            
                            # Get the public URL using Google Cloud Storage client
                            bucket = gcs_client.bucket(bucket_name)
                            blob = bucket.blob(blob_name)
                            public_url = blob.public_url
                            
                            converted_urls.append(public_url)
                            total_conversions += 1
                            row_has_conversions = True
                            
                            print(f"  Converted: {url}")
                            print(f"  To: {public_url}")
                        else:
                            # Just bucket name, use simple conversion
                            public_url = f"https://storage.googleapis.com/{path_without_prefix}"
                            converted_urls.append(public_url)
                            total_conversions += 1
                            row_has_conversions = True
                            
                    except Exception as e:
                        print(f"  Error converting {url}: {e}")
                        # Keep original URL if conversion fails
                        converted_urls.append(url)
                else:
                    # Not a gs:// URL, keep as is
                    converted_urls.append(url)
            
            # Update the row if there were conversions
            if row_has_conversions:
                messages_df.at[index, 'Attachments'] = ';'.join(converted_urls)
                rows_updated += 1
    
    # Save the updated Excel file
    if total_conversions > 0:
        print(f"\nSaving updated Excel file...")
        print(f"Total URLs converted: {total_conversions}")
        print(f"Rows updated: {rows_updated}")
        
        try:
            # Handle datetime columns if they exist - convert to timezone-naive for Excel compatibility
            messages_df_excel = messages_df.copy()
            if 'DateTime' in messages_df_excel.columns:
                messages_df_excel['DateTime'] = pd.to_datetime(messages_df_excel['DateTime']).dt.tz_localize(timezone.utc)
            
            messages_df_excel.to_excel(excel_file_path, index=False)
            print(f"Successfully saved updated file: {excel_file_path}")
            
        except Exception as e:
            print(f"Error saving Excel file: {e}")
    else:
        print("No gs:// URLs found to convert.")


# Uncomment ONE of the lines below to run the conversion function:

# Option 1: Use signed URLs (works with private buckets, URLs expire in 24 hours)
# convert_gs_to_signed_urls_in_excel()

def convert_public_to_gs_urls_in_excel():
    """
    REVERTING FUNCTION: Convert public https://storage.googleapis.com/ URLs back to gs:// URLs in the Excel file.
    This function reads the Excel file, converts all public URLs back to gs:// URLs,
    and saves the updated file. Use this to revert the public URL conversion.
    """
    import pandas as pd
    from app.utils.url_converter import https_to_gs_url
    
    excel_file_path = '.\\data\\discord_messages_with_gcs_pics.xlsx'
    
    print(f"Loading Excel file: {excel_file_path}")
    
    # Load the Excel file
    try:
        messages_df = pd.read_excel(excel_file_path)
        print(f"Loaded {len(messages_df)} rows from Excel file")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Track conversions
    total_conversions = 0
    rows_updated = 0
    
    print("Converting public URLs back to gs:// URLs...")
    
    # Process each row
    for index, row in messages_df.iterrows():
        if row['Attachments'] != 'No attachments':
            attachment_urls = row['Attachments'].split(';')
            converted_urls = []
            row_has_conversions = False
            
            for url in attachment_urls:
                url = url.strip()
                if url.startswith('https://storage.googleapis.com/'):
                    try:
                        # Use the utility function to convert back to gs://
                        gs_url = https_to_gs_url(url)
                        converted_urls.append(gs_url)
                        total_conversions += 1
                        row_has_conversions = True
                        
                        print(f"  Converted: {url}")
                        print(f"  To: {gs_url}")
                        
                    except Exception as e:
                        print(f"  Error converting {url}: {e}")
                        # Keep original URL if conversion fails
                        converted_urls.append(url)
                else:
                    # Not a public URL, keep as is
                    converted_urls.append(url)
            
            # Update the row if there were conversions
            if row_has_conversions:
                messages_df.at[index, 'Attachments'] = ';'.join(converted_urls)
                rows_updated += 1
    
    # Save the updated Excel file
    if total_conversions > 0:
        print(f"\nSaving updated Excel file...")
        print(f"Total URLs converted: {total_conversions}")
        print(f"Rows updated: {rows_updated}")
        
        try:
            # Handle datetime columns if they exist - convert to timezone-naive for Excel compatibility
            messages_df_excel = messages_df.copy()
            if 'DateTime' in messages_df_excel.columns:
                messages_df_excel['DateTime'] = pd.to_datetime(messages_df_excel['DateTime']).dt.tz_localize(None)
            
            messages_df_excel.to_excel(excel_file_path, index=False)
            print(f"Successfully saved updated file: {excel_file_path}")
            
        except Exception as e:
            print(f"Error saving Excel file: {e}")
    else:
        print("No public URLs found to convert back to gs:// format.")


def test_url_accessibility():
    """
    TEMPORARY FUNCTION: Test if the public URLs are actually accessible.
    This helps diagnose why Gemini is getting URL_RETRIEVAL_STATUS_ERROR.
    """
    import requests
    import pandas as pd
    
    excel_file_path = '.\\data\\discord_messages_with_gcs_pics.xlsx'
    
    print("Testing URL accessibility...")
    
    try:
        messages_df = pd.read_excel(excel_file_path)
        print(f"Loaded {len(messages_df)} rows from Excel file")
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    # Test a few URLs
    test_count = 0
    max_tests = 5
    
    for index, row in messages_df.iterrows():
        if row['Attachments'] != 'No attachments' and test_count < max_tests:
            attachment_urls = row['Attachments'].split(';')
            
            for url in attachment_urls:
                url = url.strip()
                if url.startswith('https://storage.googleapis.com/') and test_count < max_tests:
                    test_count += 1
                    print(f"\n--- Testing URL {test_count} ---")
                    print(f"URL: {url}")
                    
                    try:
                        # Test with HEAD request (faster than GET)
                        response = requests.head(url, timeout=10)
                        print(f"Status Code: {response.status_code}")
                        print(f"Content-Type: {response.headers.get('Content-Type', 'Not specified')}")
                        print(f"Content-Length: {response.headers.get('Content-Length', 'Not specified')}")
                        
                        if response.status_code == 200:
                            print("✅ URL is accessible!")
                        else:
                            print(f"❌ URL returned status {response.status_code}")
                            
                    except requests.exceptions.RequestException as e:
                        print(f"❌ Error accessing URL: {e}")
                    
                    # Also test with GET request to see if it's a HEAD vs GET issue
                    try:
                        response = requests.get(url, timeout=10, stream=True)
                        print(f"GET Status Code: {response.status_code}")
                        if response.status_code == 200:
                            print("✅ URL is accessible via GET!")
                        else:
                            print(f"❌ URL returned status {response.status_code} via GET")
                    except requests.exceptions.RequestException as e:
                        print(f"❌ Error accessing URL via GET: {e}")
                elif url.startswith('gs://'):
                    test_count += 1
                    print(f"\n--- Testing URL {test_count} ---")
                    print(f"URL: {url}")
                    try:
                        # Parse bucket and blob name from gs:// URL
                        path_without_prefix = url[5:]  # Remove 'gs://'
                        if '/' in path_without_prefix:
                            bucket_name, blob_name = path_without_prefix.split('/', 1)
                            
                            # Check if blob exists in GCS
                            bucket = gcs_client.bucket(bucket_name)
                            blob = bucket.blob(blob_name)
                            
                            if blob.exists():
                                print("✅ Blob exists in Google Cloud Storage!")
                                
                                # Get blob metadata
                                blob.reload()  # Fetch latest metadata
                                print(f"   Size: {blob.size} bytes")
                                print(f"   Content-Type: {blob.content_type}")
                                print(f"   Created: {blob.time_created}")
                                print(f"   Updated: {blob.updated}")
                                
                                # Test if blob is publicly accessible
                                
                            else:
                                print("❌ Blob does not exist in Google Cloud Storage!")
                                
                    except Exception as e:
                        print(f"❌ Error checking blob existence: {e}")

    print(f"\nTested {test_count} URLs")


# Uncomment ONE of the lines below to run the conversion function:

# Option 1: Use signed URLs (works with private buckets, URLs expire in 24 hours)
# convert_gs_to_signed_urls_in_excel()

# Option 2: Make bucket public first, then use public URLs (permanent public access)
# make_bucket_public()
# convert_gs_to_public_urls_in_excel()

# Option 3: REVERT - Convert public URLs back to gs:// URLs
# convert_public_to_gs_urls_in_excel()

# Option 4: Test URL accessibility to diagnose the issue
test_url_accessibility()

