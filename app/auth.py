"""Authentication middleware for WebScraper App."""

from fastapi import HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
from typing import Optional

security = HTTPBearer()

def get_api_token() -> str:
    """Get API token from environment variables."""
    token = os.getenv("API_TOKEN")
    if not token:
        raise ValueError("API_TOKEN environment variable is not set")
    return token

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> str:
    """Verify the provided token against the configured API token."""
    expected_token = get_api_token()
    
    if credentials.credentials != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return credentials.credentials

def get_auth_dependency():
    """Get the authentication dependency for FastAPI endpoints."""
    return Security(verify_token)