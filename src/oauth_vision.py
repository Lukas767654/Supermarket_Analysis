"""
OAuth-based Google Vision API client using CLIENT_ID and CLIENT_SECRET.
"""
import logging
import os
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.cloud import vision
import json
from pathlib import Path

from .config import config

logger = logging.getLogger(__name__)

class OAuthVisionClient:
    """Google Vision API client using OAuth credentials."""
    
    def __init__(self):
        self.client = None
        self.credentials = None
        self._setup_oauth_client()
    
    def _setup_oauth_client(self):
        """Setup OAuth-based Vision client."""
        try:
            # OAuth credentials from .env
            client_id = os.getenv('CLIENT_ID')
            client_secret = os.getenv('CLIENT_SECRET')
            
            if not client_id or not client_secret:
                logger.error("CLIENT_ID or CLIENT_SECRET not found in environment")
                return
            
            # OAuth configuration
            client_config = {
                "web": {
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost:8080"]
                }
            }
            
            # Scopes needed for Vision API
            scopes = ['https://www.googleapis.com/auth/cloud-platform']
            
            # Try to load existing credentials
            token_path = "oauth_token.json"
            if os.path.exists(token_path):
                logger.info("Loading existing OAuth credentials...")
                try:
                    self.credentials = Credentials.from_authorized_user_file(
                        token_path, scopes
                    )
                except Exception as e:
                    logger.warning(f"Failed to load existing credentials: {e}")
                    self.credentials = None
            
            # Refresh credentials if needed
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                try:
                    logger.info("Refreshing OAuth credentials...")
                    self.credentials.refresh(Request())
                except Exception as e:
                    logger.warning(f"Failed to refresh credentials: {e}")
                    self.credentials = None
            
            # If no valid credentials, start OAuth flow
            if not self.credentials or not self.credentials.valid:
                logger.info("Starting OAuth flow for Google Vision API...")
                
                flow = Flow.from_client_config(
                    client_config,
                    scopes=scopes,
                    redirect_uri='http://localhost:8080'
                )
                
                # Get authorization URL
                auth_url, _ = flow.authorization_url(
                    access_type='offline',
                    include_granted_scopes='true'
                )
                
                logger.info(f"Please visit this URL to authorize the application:")
                logger.info(f"{auth_url}")
                logger.info("After authorization, you'll get a redirect URL. Copy the 'code' parameter from that URL.")
                
                # This would normally require user interaction
                # For now, we'll create a simpler test approach
                return self._setup_test_mode()
            
            # Save credentials
            if self.credentials:
                with open(token_path, 'w') as token_file:
                    token_file.write(self.credentials.to_json())
                
                # Create Vision client
                self.client = vision.ImageAnnotatorClient(credentials=self.credentials)
                logger.info("OAuth Vision client initialized successfully")
                
        except Exception as e:
            logger.error(f"Failed to setup OAuth Vision client: {e}")
            return self._setup_test_mode()
    
    def _setup_test_mode(self):
        """Setup test mode with mock responses for development."""
        logger.info("Setting up test mode - will use mock Vision API responses")
        self.client = None  # Will trigger mock mode in vision_api.py

# Update the main vision_api.py to use OAuth client
oauth_vision_client = OAuthVisionClient()