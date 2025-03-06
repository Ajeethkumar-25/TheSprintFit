# existing_routes/investor_routes.py
# Contains investor profile management routes from the original code

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, status
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import psycopg2
import boto3
import os
from datetime import datetime

# Import the DB connection and authentication from your utils
from database_utils import get_db_connection
from .authentication_routes import get_current_user

# Initialize router
router = APIRouter(tags=["Investor Profiles"])

# Logger
logger = logging.getLogger(__name__)

# AWS S3 Config
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "AKIAS6J7QNNRXHXM45GE")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "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")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "thesprintfit")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")
S3_BASE_URL = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/"

# Initialize S3 client
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=S3_REGION
)

# Pydantic models
class InvestorData(BaseModel):
    investor_type: str
    investment_experience: str
    investment_focus_areas: List[str]
    primary_impact_areas: str
    geographical_preferences: List[str]
    startup_stages: List[str]
    business_models: List[str]
    target_roi: float
    exit_horizon: int
    check_size_min: int
    check_size_max: int
    preferred_ownership: float
    revenue_milestones: str
    monthly_recurring_revenue: int
    tam: int
    som: int
    sam: int
    traction_revenue_market: str
    technology_scalability: List[str]

# Routes
@router.post("/investors/", dependencies=[Depends(get_current_user)])
def create_investor(profile: InvestorData, current_user: dict = Depends(get_current_user)):
    """
    Create an investor profile and link it with the logged-in user's email.
    """
    user_email = current_user.get("email")  # Get email from logged-in user

    if not user_email:
        raise HTTPException(status_code=401, detail="Unauthorized: Email not found")

    query = """
    INSERT INTO investor_profiles 
    (investor_type, investment_experience, investment_focus_areas, primary_impact_areas, 
     geographical_preferences, startup_stages, business_models, target_roi, exit_horizon, 
     check_size_min, check_size_max, preferred_ownership, revenue_milestones, 
     monthly_recurring_revenue, tam, som, sam, traction_revenue_market, technology_scalability, email) 
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    RETURNING id;
    """
    
    values = (
        profile.investor_type, profile.investment_experience, profile.investment_focus_areas,
        profile.primary_impact_areas, profile.geographical_preferences, profile.startup_stages,
        profile.business_models, profile.target_roi, profile.exit_horizon, profile.check_size_min,
        profile.check_size_max, profile.preferred_ownership, profile.revenue_milestones,
        profile.monthly_recurring_revenue, profile.tam, profile.som, profile.sam,
        profile.traction_revenue_market, profile.technology_scalability, user_email  # Use logged-in user's email
    )
    
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(query, values)
                investor_id = cursor.fetchone()[0]
                conn.commit()
                return {
                    "id": investor_id,
                    "email": user_email,
                    "message": "Investor created successfully"
                }
            except psycopg2.Error as e:
                conn.rollback()
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@router.post("/investors/{investor_id}/upload/", dependencies=[Depends(get_current_user)])
async def upload_pitch_deck(investor_id: int, file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_content = await file.read()
    file_size = len(file_content)

    if file_size > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be under 10MB")

    try:
        file_extension = file.filename.split(".")[-1]
        s3_filename = f"investor_{investor_id}_pitchdeck_{int(datetime.utcnow().timestamp())}.{file_extension}"

        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_filename,
            Body=file_content,
            ContentType="application/pdf"
        )

        file_url = f"{S3_BASE_URL}{s3_filename}"

        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                query = """
                UPDATE investor_profiles 
                SET pitch_deck_url = %s 
                WHERE id = %s 
                RETURNING id;
                """
                cursor.execute(query, (file_url, investor_id))
                result = cursor.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="Investor not found")
                
                conn.commit()
                return {"message": "Pitch deck uploaded successfully", "file_url": file_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get_all_investors/")
def get_all_investors():
    query = """
    SELECT id, investor_type, investment_experience, investment_focus_areas,
           geographical_preferences, startup_stages, target_roi, check_size_min,
           check_size_max, pitch_deck_url
    FROM investor_profiles
    ORDER BY id DESC;
    """
    
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(query)
                investors = cursor.fetchall()
                return {"investors": investors, "count": len(investors)}
            except psycopg2.Error as e:
                raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")