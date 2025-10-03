import os
from google.oauth2 import service_account
from google.cloud import storage
from google import genai
from dotenv import load_dotenv

load_dotenv()

class GoogleCloudAuth:
    def __init__(self):
        self.credentials = None
        self._initialize_credentials()
    
    def _initialize_credentials(self):
        """Initialize Google Cloud credentials using service account key."""
        service_account_path = os.environ.get("GOOGLE_SERVICE_CREDENTIALS")
        
        if service_account_path and os.path.exists(service_account_path):
            # Use service account key file
            self.credentials = service_account.Credentials.from_service_account_file(
                service_account_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            # Fallback to application default credentials
            import google.auth
            self.credentials, _ = google.auth.default()
    
    def get_storage_client(self):
        """Get authenticated Google Cloud Storage client."""
        return storage.Client(credentials=self.credentials)
    
    def get_gemini_client(self):
        """Get authenticated Gemini client."""
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_REGION")
        
        return genai.Client(
            project=project_id,
            location=location,
            credentials=self.credentials
        )

# Global instance
auth_service = GoogleCloudAuth()
