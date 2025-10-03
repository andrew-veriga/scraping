from dotenv import load_dotenv
from google.cloud import storage
import yaml
import os
from app.services import data_loader
from google.genai import errors
from datetime import datetime, timezone, timedelta
from app.utils.file_utils import convert_Timestamp_to_str
import logging
from fastapi import HTTPException
from typing import Optional
import pandas as pd
load_dotenv()
from app.services.database import get_database_service

from google import genai
from google.genai import types
from google.genai.types import Part
import base64
import requests
import json
from urllib.parse import urlparse

def process_attachment_urls(attachments: str, gcs_client: storage.Client = None) -> str:
    """
    Process attachment URLs to ensure they're in the correct format for the Gemini API.
    Converts gs:// URLs to public https URLs using Google Cloud client.
    """
    if attachments == 'No attachments':
        return attachments
    # gcs_client = storage.Client()
    # gcs_bucket_name = config['images']['gcs_bucket_name']
    # gcs_bucket = gcs_client.bucket(gcs_bucket_name)

    urls = attachments.split(';')
    processed_urls = []
    
    for url in urls:
        url = url.strip()
        # Convert gs:// URLs to public https URLs using Google Cloud client
        if url.startswith('gs://'):
            if gcs_client:
                try:
                    # Parse gs:// URL to get bucket and blob name
                    path_without_prefix = url[5:]  # Remove 'gs://'
                    if '/' in path_without_prefix:
                        bucket_name, blob_name = path_without_prefix.split('/', 1)
                        bucket = gcs_client.bucket(bucket_name)
                        blob = bucket.blob(blob_name)
                        url = blob.public_url
                    else:
                        # Just bucket name, use simple conversion
                        url = f"https://storage.googleapis.com/{path_without_prefix}"
                except Exception:
                    # Fallback to simple string replacement
                    url = f"https://storage.googleapis.com/{path_without_prefix}"
            else:
                # Fallback without client
                path_without_prefix = url[5:]  # Remove 'gs://'
                url = f"https://storage.googleapis.com/{path_without_prefix}"
        
        processed_urls.append(url)
    
    return ';'.join(processed_urls)

def get_mime_type_from_extension(file_extension):
    """
    Get valid MIME type from file extension.
    
    Args:
        file_extension (str): File extension (with or without dot)
        
    Returns:
        str: Valid MIME type or 'image/jpeg' as fallback
    """
    # Remove dot if present and convert to lowercase
    ext = file_extension.lower().lstrip('.')
    
    # Comprehensive mapping of image extensions to MIME types
    mime_type_mapping = {
        # JPEG variants
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'jpe': 'image/jpeg',
        
        # PNG
        'png': 'image/png',
        
        # GIF
        'gif': 'image/gif',
        
        # WebP
        'webp': 'image/webp',
        
        # BMP
        'bmp': 'image/bmp',
        'dib': 'image/bmp',
        
        # TIFF variants
        'tiff': 'image/tiff',
        'tif': 'image/tiff',
        
        # ICO
        'ico': 'image/x-icon',
        'icon': 'image/x-icon',
        
        # SVG
        'svg': 'image/svg+xml',
        
        # Other formats
        'heic': 'image/heic',
        'heif': 'image/heif',
        'avif': 'image/avif',
        'jp2': 'image/jp2',
        'j2k': 'image/jp2',
        'jpx': 'image/jp2',
    }
    return mime_type_mapping.get(ext, 'image/jpeg')  # Default fallback

def get_uploaded_name(blob_name: str)->str:
    return blob_name.split('/')[-1].split('_')[0]

def image_part_from_gs_url(
    gs_url,
    gcs_bucket: storage.Bucket,
    gemini_client: genai.Client,
    dict_uploaded_images: dict,
    tmp_image_path: str,
    )->tuple[Part, types.File]:
    """
    upload image from gs:// URL to gemini client files, add to cache list for caching if present and make an content Part from its uri
    Make an image part from a gs:// URL
    args:
    gs_url - gs:// URL of the image
    gcs_bucket - gcs bucket of the image
    gemini_client - gemini client
    dict_uploaded_images - dictionary of images with keys of their names and values of their gemini uploaded files. If image is not in gemini client files, add it to the dictionary.
    return:
    image_part - Part of Content 
    file_id - file id of the image in gemini client for caching
    """
    _, blob_name = gs_url[5:].split('/', 1)
    blob = gcs_bucket.blob(blob_name)
    if blob.exists():
        uploaded_name = get_uploaded_name(blob_name)
        file_id = dict_uploaded_images.get(uploaded_name,None)
        try:
            if file_id is None:
                file_id = gemini_client.files.get(name='files/'+uploaded_name) #  file is in gemini files, but not in dict_uploaded_images
                # if cuccessfully got file from gemini files, and is not in dict_uploaded_images, add it to dict_uploaded_images
                dict_uploaded_images[uploaded_name] = file_id
            elif file_id.expiration_time > datetime.now(tz=timezone.utc): # file  is in dict_uploaded_images and not expired
                print(f"file: {uploaded_name} already exists in gemini files and is still valid for {int((file_id.expiration_time-datetime.now(tz=timezone.utc)).total_seconds()/3600)} hours")
                # file is in gemini files, and is in dict_uploaded_images but expired
            else:
                # file is not in gemini files, or is in dict_uploaded_images but expired
                print(f"file: {uploaded_name} already exists  in gemini files but expired, uploading to gemini files")
                file_id = gemini_client.files.get(name='files/'+uploaded_name) # robust to get file from gemini files
        except errors.ClientError:
            os.makedirs(os.path.join(tmp_image_path, blob.name.split('/')[-2]), exist_ok=True)
            file_path=os.path.join(tmp_image_path, blob_name)
            blob.download_to_filename(file_path)
            mime_type = blob.content_type
            file_id = gemini_client.files.upload(file=file_path, config=types.UploadFileConfig(name=uploaded_name, mime_type=mime_type))
        except Exception as e:
            logging.error(f"Error uploading file: {uploaded_name} to gemini files: {e}", exc_info=True)
            return None
        dict_uploaded_images[uploaded_name] = file_id
        image_part = Part.from_uri(
            # display_name =blob_name,
            file_uri=file_id.uri, #"gs://discord_pics/attachments/0151caec_image.png",
            mime_type=file_id.mime_type
        )
        # image_part.file_data.display_name=blob_name
        return image_part
    else:
        logging.error(f"Error downloading image: {gs_url}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error downloading image: {gs_url}")

def is_valid_url(url_string):
    """Check if a string contains valid URLs"""
    
    urls = url_string.split(';')
    for url in urls:
        url = url.strip()
        if url:  # Check if not empty
            try:
                result = urlparse(url)
                # Check if it has both scheme and netloc
                return all([result.scheme, result.netloc])
            except Exception:
                return False
    return False

def make_text_part(text):
    return Part(text=text)

def make_json_part(json_data):
    return Part(text=json.dumps(json_data))

def convert_attachments_to_markdown(attachments: str) -> str:
    """
    Convert attachment URLs to markdown image syntax.
    
    Args:
        attachments (str): Semicolon-separated string of attachment URLs
        
    Returns:
        str: Markdown formatted string with image syntax
    """
    if not attachments or attachments == 'No attachments' or attachments.strip() == '':
        return ''
    
    # Split by semicolon and process each URL
    urls = attachments.split(';')
    markdown_images = []
    
    for url in urls:
        url = url.strip()
        if url:  # Only process non-empty URLs
            # Extract filename for alt text
            bucket_name, blob_name = url[5:].split('/', 1)
            gcs_client = storage.Client()
            gcs_bucket = gcs_client.bucket(bucket_name)
            blob = gcs_bucket.blob(blob_name)
            if blob.exists():
                url = blob.public_url
            filename = url.split('/')[-1].split('?')[0]  # Remove query parameters
            
            # Create markdown image syntax with natural width
            # Option 1: HTML img tag with no width specified (uses natural width)
            markdown_image = f'<img src="{url}" alt="{filename}" style="width: 50%; height: auto;" />'
            
            # Option 2: If you prefer markdown syntax with natural width
            # markdown_image = f'![{filename}]({url})'
            
            # Option 3: If you need to specify natural width explicitly
            # markdown_image = f'<img src="{url}" alt="{filename}" style="width: auto; height: auto;" />'
            
            markdown_images.append(markdown_image)
    
    return '\n'.join(markdown_images)


def is_valid_url(url_string):
    """Check if a string contains valid URLs"""
    
    urls = url_string.split(';')
    for url in urls:
        url = url.strip()
        if url:  # Check if not empty
            try:
                result = urlparse(url)
                # Check if it has both scheme and netloc
                return all([result.scheme, result.netloc])
            except Exception:
                return False
    return False


def _create_genai_cache_for_images(dict_uploaded_images, global_config, gemini_client: genai.Client)-> Optional[types.CachedContent]:
    """
    Create GenAI cache for uploaded images and save cache metadata.
    
    Args:
        dict_uploaded_images: Dictionary of uploaded images with File objects
        global_config: Global configuration dictionary to store cache info
        
    Returns:
        object: The created GenAI cache object or None if creation failed
    """
    if len(dict_uploaded_images) == 0:
        return None
        
    # Delete existing cache if present ?????????????????????
    json_cache_thread_gathering = global_config.get('image_genai_cache', None)
    if json_cache_thread_gathering:
        cache_name = json_cache_thread_gathering.get('name', None)
        if cache_name:
            cache = gemini_client.caches.get(name=cache_name)
            if cache:
                gemini_client.caches.delete(name=cache_name)
                logging.info(f"Deleted cache: {cache_name}")
    
    ttl = f"{int(timedelta(hours=2).total_seconds())}s"
    model_name = global_config['llm']['model_name']
    try:
        image_parts = [types.Part.from_uri(file_uri=v.uri, mime_type=v.mime_type) for v in dict_uploaded_images.values()]
        image_genai_cache = gemini_client.caches.create( 
            model=model_name,
            config=types.CreateCachedContentConfig(
                contents=image_parts,
                display_name='thread_gathering',
                ttl=ttl
                )
            )
    except Exception as e:
        error_msg = str(e)
        image_genai_cache = None
        if "Cached content is too small" in error_msg or "min_total_token_count" in error_msg:
            logging.info(f"⚠️  Cache content too small for caching: {error_msg}")
        else:
            logging.error(f"❌ Error creating cache: {error_msg}", exc_info=True)
    
    if image_genai_cache:
        print(f"   Total content items: {len(dict_uploaded_images)}")
        print(f"   Saving dict_uploaded_images for future use when content is sufficient...")
        
        # Save dict_uploaded_images for future use
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'content_count': len(dict_uploaded_images)
        }
        
        cache_file_path = 'cache_list_backup.json'
        try:
            with open(cache_file_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
            print(f"   ✅ Cache list saved to: {cache_file_path}")
        except Exception as save_error:
            print(f"   ❌ Error saving cache list: {save_error}")
        logging.info(f"image_genai_cache: {image_genai_cache.model_dump()}")
        global_config['image_genai_cache'] = image_genai_cache.model_dump()
        json.dump(global_config, open('configs/config.yaml', 'w'), indent=2, default=str)
    return image_genai_cache

def parts_from_dataframe(df: pd.DataFrame, global_config: dict)->(list[Part], object):
    """
    Make parts from dataframe with images
    args:
    df - dataframe with messages and attachments
    global_config - global config
    return:
    parts - list of Parts
    image_genai_cache - genai cache for extracted images
    """
    from app.services.gemini_service import gcs_bucket, gemini_client
    from app.utils.yaml_file_utils import save_yaml_genai_files, load_yaml_genai_files

    # tmp_image_path - path to save images
    tmp_image_path=global_config['images']['base_path']
    # dict_uploaded_images - dictionary of images with their names and urls in gemini client files
    dict_uploaded_images = load_yaml_genai_files('configs/dict_uploaded_images.yaml')
    
    # remove expired images
    expired_images = []
    for key, value in dict_uploaded_images.items():
        if isinstance(value, types.File):
            if value.expiration_time < datetime.now(tz=timezone.utc):
                expired_images.append(key)
                logging.info(f"Expired image: {value.display_name}")
    for key in expired_images:
        dict_uploaded_images.pop(key)
    if len(expired_images) > 0:
        logging.info(f"Found {len(expired_images)} expired images")
    
    image_parts = []
    # i=0
    dict_df = df.to_dict(orient='index')
    for idx,row in dict_df.items():
        attachments = row['Attachments']# or row['attachments']
        _datetime = row['DateTime']# or row['datetime']
        if isinstance(_datetime, pd.Timestamp):
            row['DateTime'] = convert_Timestamp_to_str(_datetime)
        author_name = row['Author Name']# or row['author']
        content = row['Content']# or row['content']
        if  is_valid_url(attachments):
            if content=='nan' or content=='':
                content = author_name + 'attached:'
            else:
                content = content + ': ' + author_name + 'attached:'
            
            # parts.append(make_json_part(row))
            uploaded_urls = []
            for attachment in attachments.split(';'):
                # i += 1
                # parts.append(make_text_part('Image ' + str(i)+':'))
                image_part = image_part_from_gs_url(
                    attachment, 
                    gcs_bucket,
                    gemini_client,
                    dict_uploaded_images,
                    tmp_image_path,
                    )
                if image_part:
                    image_parts.append(image_part)
                    uploaded_urls.append(image_part.file_data.model_dump_json())
                # if file_id:
                #     dict_uploaded_images[attachment] = file_id.uri
            row['Attachments'] = ';'.join(uploaded_urls)
    parts=[make_json_part(json.dumps(dict_df))] + image_parts

    save_yaml_genai_files(dict_uploaded_images, 'configs/dict_uploaded_images.yaml')
    # Create GenAI cache for images (temporarily removed for clear)
    image_genai_cache = None # _create_genai_cache_for_images(dict_uploaded_images, global_config, gemini_client)

    return parts, image_genai_cache

def test_parts_from_dataframe():
    """Test parts_from_dataframe function using real data from data_loader.load_and_preprocess_data."""
    # Check if the file exists before trying to load it
    MESSAGES_FILE_PATH ='C:\\VSCode\\scraping\\data\\discord_messages_names_gcs_pics.xlsx'
    sample_messages_file_path = MESSAGES_FILE_PATH

    with open("configs/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    if not os.path.exists(sample_messages_file_path):
        print(f"Test file not found: {sample_messages_file_path}")
        return
    
    try:
        # Load real data using the actual data_loader function
        df, _ = data_loader.load_and_preprocess_data(sample_messages_file_path)
        df=df.head(300)
        # Verify we got some data
        assert len(df) > 0, "No data loaded from file"
        assert 'Attachments' in df.columns, "Attachments column not found in loaded data"
        
        # Call parts_from_dataframe with the real loaded data

        parts, image_genai_cache = parts_from_dataframe(df, global_config=config)

        # Save with safe_dump to avoid Python object tags
    
        # Assertions
        assert isinstance(parts, list)
        assert len(parts) > 0, "No parts generated from real data"
        assert isinstance(image_genai_cache, Optional[types.CachedContent])
        # Check that we have JSON parts for each message

        print(f"count of parts: {len(parts)}")
    except Exception as e:
        print(f"Error: {e}")
from sqlalchemy.orm import Session
def parts_from_db_messages(thread_id: str, global_config: dict, session: Session=None) -> tuple[list[Part], object]:
    """
    Extract parts from database messages for a specific thread.
    
    Args:
        thread_id (str): The thread ID to get messages from
        session (Session, optional): Database session. If None, creates a new session.
        global_config (dict): Global configuration dictionary
        
    Returns:
        tuple: (parts, image_genai_cache) where parts is a list of Part objects
               and image_genai_cache is the GenAI cache object
    """
    def _messages_to_dict(messages) -> dict:
        """Convert database messages to dictionary format for DataFrame creation."""
        return {
            msg.message_id: {
                'Message ID': msg.message_id,
                'Author ID': msg.author_id,
                'Content': msg.content,
                'DateTime': msg.datetime if msg.datetime else None,
                'Referenced Message ID': msg.referenced_message_id,
                'Attachments': msg.attachments,
                'Author Name': msg.author.author_name,
            } for msg in messages
        }
    
    db_service = get_database_service()
    if session is None:
        with db_service.get_session() as session:
            messages = db_service.get_thread_messages(session, thread_id)
            messages_dict = _messages_to_dict(messages)
            df = pd.DataFrame.from_dict(messages_dict, orient='index')
            parts, image_genai_cache = parts_from_dataframe(df, global_config=global_config)
    else:
        messages = db_service.get_thread_messages(session, thread_id)
        messages_dict = _messages_to_dict(messages)
        df = pd.DataFrame.from_dict(messages_dict, orient='index')
        parts, image_genai_cache = parts_from_dataframe(df, global_config=global_config)
    
    return parts, image_genai_cache

def test_parts_from_db_messages():
    """Test parts_from_dataframe function using data from database."""
    with open("configs/config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
    
    parts, image_genai_cache = parts_from_db_messages('1396582775515123968', config)
    print(f"count of parts: {len(parts)}")

if __name__ == "__main__":
    test_parts_from_db_messages()
    # test_parts_from_dataframe()


