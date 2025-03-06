# existing_routes/authentication_routes.py
# Contains authentication-related routes from the original code

from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timedelta
import jwt
import secrets
import logging
import requests
import os
from contextlib import contextmanager
from psycopg2.extras import RealDictCursor

# Import the DB connection from your utils
from database_utils import get_db_connection

router = APIRouter(tags=["Authentication"])

# Logger
logger = logging.getLogger(__name__)

# Authentication configuration
SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15

# LinkedIn Configuration
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USER_INFO_URL = "https://api.linkedin.com/v2/me"
LINKEDIN_EMAIL_URL = "https://api.linkedin.com/v2/emailAddress?q=members&projection=(elements*(handle~))"
LINKEDIN_CLIENT_ID = os.getenv("LINKEDIN_CLIENT_ID", "862y38vijwl2bo") 
LINKEDIN_CLIENT_SECRET = os.getenv("LINKEDIN_CLIENT_SECRET", "WPL_AP1.NHxhFw63PjAviOa2.xSLofw==")
LINKEDIN_REDIRECT_URI = os.getenv("LINKEDIN_REDIRECT_URI", "http://poc.thesprintfit.com/dashboard")

# Google Auth Config
GOOGLE_TOKEN_INFO_URL = "https://www.googleapis.com/oauth2/v3/tokeninfo"

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

# Pydantic models
class LinkedInSignInRequest(BaseModel):
    code: str  # LinkedIn authorization code

class GoogleSignInRequest(BaseModel):
    access_token: str
    email: str

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class TokenResponse(BaseModel):
    access: str
    refresh_token: Optional[str] = None
    token_type: str
    email: Optional[str] = None

# Token functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_refresh_token(data: dict, expires_delta: timedelta = timedelta(days=7)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_email = payload.get("sub")
        if user_email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return {"email": user_email}
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# LinkedIn authentication functions
def exchange_code_for_access_token(code: str):
    """ Exchange authorization code for an access token """
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": LINKEDIN_REDIRECT_URI,
        "client_id": LINKEDIN_CLIENT_ID,
        "client_secret": LINKEDIN_CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(LINKEDIN_TOKEN_URL, data=data, headers=headers)
    
    if response.status_code != 200:
        logger.error(f"Failed to get access token from LinkedIn: {response.text}")
        raise HTTPException(status_code=400, detail="Failed to get access token from LinkedIn")

    return response.json().get("access_token")

def get_linkedin_user_info(access_token: str):
    """ Fetch user email and LinkedIn ID using the access token """
    headers = {"Authorization": f"Bearer {access_token}"}

    # Get user info
    user_info_resp = requests.get(LINKEDIN_USER_INFO_URL, headers=headers, timeout=10)
    if user_info_resp.status_code != 200:
        logger.error(f"LinkedIn user info error: {user_info_resp.text}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid LinkedIn token")

    user_info = user_info_resp.json()
    linkedin_id = user_info.get("id")

    # Get user email
    email_resp = requests.get(LINKEDIN_EMAIL_URL, headers=headers, timeout=10)
    if email_resp.status_code != 200:
        logger.error(f"LinkedIn email fetch error: {email_resp.text}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unable to fetch email")

    email_info = email_resp.json()
    email = email_info["elements"][0]["handle~"]["emailAddress"]

    return {"email": email, "linkedin_id": linkedin_id}

# Google authentication functions
def verify_google_token(access_token: str):
    try:
        response = requests.get(
            GOOGLE_TOKEN_INFO_URL,
            params={"access_token": access_token},
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Google token verification failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google token"
        )

# Routes
@router.post("/linkedin_signin", response_model=TokenResponse)
async def linkedin_signin(request: LinkedInSignInRequest):
    logger.info("Received LinkedIn signin request")

    # Exchange code for access token
    access_token = exchange_code_for_access_token(request.code)

    # Get user email and LinkedIn ID
    user_info = get_linkedin_user_info(access_token)
    email = user_info.get("email")

    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not found in LinkedIn response")

    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("SELECT * FROM Linkedin_login WHERE email = %s", (email,))
            user = cursor.fetchone()

            if not user:
                logger.info(f"Creating new user with email: {email}")
                cursor.execute("INSERT INTO Linkedin_login (email, created_at) VALUES (%s, %s) RETURNING id", (email, datetime.utcnow()))
                conn.commit()
                user = cursor.fetchone()

            access_token = create_access_token({"sub": email})
            refresh_token = create_refresh_token({"sub": email})

            cursor.execute("UPDATE Linkedin_login SET refresh_token = %s WHERE email = %s", (refresh_token, email))
            conn.commit()

            return TokenResponse(
                access=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                email=email
            )

        except Exception as e:
            conn.rollback()
            logger.error(f"LinkedIn Sign-in error: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during sign-in")
        finally:
            cursor.close()

@router.post("/signin", response_model=TokenResponse)
async def google_signin(request: GoogleSignInRequest):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            logger.info("Received Google signin request")
            # Uncomment if you want to verify the token with Google
            # user_info = verify_google_token(request.access_token)
            
            email = request.email

            if not email:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email not found")

            cursor.execute("SELECT * FROM login WHERE email = %s", (email,))
            user = cursor.fetchone()

            if not user:
                logger.info(f"Creating new user with email: {email}")
                cursor.execute("INSERT INTO login (email, created_at) VALUES (%s, %s) RETURNING id", (email, datetime.utcnow()))
                conn.commit()
                user = cursor.fetchone()

            access_token = create_access_token({"sub": email})
            refresh_token = secrets.token_urlsafe(32)

            cursor.execute("UPDATE login SET refresh_token = %s WHERE email = %s", (refresh_token, email))
            conn.commit()

            return TokenResponse(
                access=access_token,
                refresh_token=refresh_token,
                token_type="bearer",
                email=email
            )
        except Exception as e:
            conn.rollback()
            logger.error(f"Sign in error: {str(e)}")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error during sign in")
        finally:
            cursor.close()
            
@router.post("/refresh_token", response_model=TokenResponse)
async def refresh_linkedin_token(refresh_token: str):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            # Check if refresh token exists in DB
            cursor.execute("SELECT email FROM Linkedin_login WHERE refresh_token = %s", (refresh_token,))
            user = cursor.fetchone()

            if not user:
                raise HTTPException(status_code=401, detail="Invalid refresh token")

            email = user["email"]
            new_access_token = create_access_token({"sub": email})

            return TokenResponse(
                access=new_access_token,
                token_type="bearer",
                email=email
            )

        except Exception as e:
            conn.rollback()
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            cursor.close()

@router.post("/refresh", response_model=TokenResponse)
async def refresh_google_token(request: RefreshTokenRequest):
    with get_db_connection() as conn:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            logger.info("Received refresh request")

            # Fetch user by refresh_token
            cursor.execute("SELECT * FROM login WHERE refresh_token = %s", (request.refresh_token,))
            user = cursor.fetchone()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )

            # Generate new access token
            access_token = create_access_token(
                {"sub": user["email"]},
                timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            )

            logger.info(f"Refresh successful for email: {user['email']}")
            return TokenResponse(
                access=access_token,
                token_type="bearer",
                email=user["email"]
            )

        except Exception as e:
            logger.error(f"Refresh token error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error during token refresh"
            )
        finally:
            cursor.close()