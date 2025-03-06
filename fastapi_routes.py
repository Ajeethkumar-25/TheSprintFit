# FastAPI Routes for Agent-Based Investment Analysis System

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from psycopg2.extras import RealDictCursor
import json
import logging
import boto3
import os
from datetime import datetime
from botocore.exceptions import NoCredentialsError

# Import your database utilities
from database_utils import get_db_connection, download_pdf_from_s3, extract_text_from_pdf, store_analysis_result, inspect_table_columns,fix_database_schema

# Import agent-based analysis system
from investment_analysis import run_investment_analysis

logger = logging.getLogger(__name__)

# AWS S3 Config
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID", "AKIAS6J7QNNRXHXM45GE")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY", "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")
S3_BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "thesprintfit")
S3_REGION = os.environ.get("S3_REGION", "ap-south-1")
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


# Updated request model
class CompareRequest(BaseModel):
    investor_id: int
    pitch_id: Optional[int] = None
    s3_url: Optional[str] = None
    # startup_name: Optional[str] = None
    # investor_data: Optional[InvestorData]

class ComparisonFilter(BaseModel):
    min_score: Optional[int] = None
    max_score: Optional[int] = None
    startup_stages: Optional[List[str]] = None
    investor_type: Optional[str] = None
    tech_areas: Optional[List[str]] = None

# FastAPI app
app = FastAPI()

# CORS Middleware - Use your existing configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="signin")

# Routes related to the new agent-based system

@app.post("/upload-pitch/")
async def upload_pitch(investor_id: int, file: UploadFile = File(...)):
    """Upload a pitch deck PDF to S3, store it in the database, and return the ID with the S3 URL."""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_content = await file.read()
    file_size = len(file_content)

    if file_size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size must be under 10MB")

    try:
        file_extension = file.filename.split(".")[-1]
        s3_filename = f"pitch_deck_{int(datetime.utcnow().timestamp())}.{file_extension}"

        # Upload to S3
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_filename,
            Body=file_content,
            ContentType="application/pdf"
        )

        file_url = f"{S3_BASE_URL}{s3_filename}"

        # Insert into pitch_summary table and get the ID
        query = """
        INSERT INTO pitch_summary (pitch_deck_url, investor_id, created_at)
        VALUES (%s, %s, NOW())
        RETURNING pitch_id;
        """
        
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, (file_url, investor_id))
                pitch_id = cursor.fetchone()["pitch_id"]
                conn.commit()

        return {
            "pitch_id": pitch_id,
            "s3_url": file_url,
            "message": "Pitch deck uploaded successfully"
        }

    except NoCredentialsError:
        raise HTTPException(status_code=500, detail="AWS credentials not available")
    except Exception as e:
        logger.error(f"Error uploading file to S3: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")


def get_primary_key_column(table_name):
    """Get the primary key column name for a given table."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                     AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = %s::regclass
                AND    i.indisprimary;
            """, (table_name,))
            
            result = cursor.fetchone()
            if result:
                return result[0]
            return None
            """Get all column names for a given table."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s;
            """, (table_name,))
            columns = [row[0] for row in cursor.fetchall()]
            logger.info(f"Columns in {table_name}: {columns}")
            return columns

@app.post("/compare-thesis-fit/")
def compare_thesis_fit(request: CompareRequest):
    logger.info("✅ compare-thesis-fit endpoint was hit!")  # ✅ Confirm endpoint is hit

    try:
        # ✅ Validate required fields
        if not request.s3_url:
            logger.error("❌ Missing s3_url in request!")
            raise HTTPException(status_code=400, detail="s3_url is required.")

        # ✅ Extract text from PDF
        logger.info(f"🔹 Received S3 URL: {request.s3_url}")
        pdf_bytes = download_pdf_from_s3(request.s3_url)
        extracted_text = extract_text_from_pdf(pdf_bytes)
        logger.info(f"🔹 Extracted Text (First 500 chars): {extracted_text[:500] if extracted_text else 'No text extracted!'}")

        if not extracted_text:
            logger.error("❌ No text extracted from PDF.")
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")

        # ✅ Extract startup name and generate structured analysis
        startup_name = extract_startup_name_from_pitch(extracted_text) or "Unknown Startup"
        logger.info(f"📌 Extracted Startup Name: {startup_name}")

        pitch_analysis = generate_structured_pitch_analysis(extracted_text)
        if "Error generating analysis" in pitch_analysis:
            logger.error(f"❌ Pitch Analysis Generation Failed. Full Output: {pitch_analysis}")
            raise HTTPException(status_code=500, detail="Pitch analysis generation failed.")
        logger.info(f"📌 Generated Pitch Analysis: {pitch_analysis[:500]}")  # ✅ Log first 500 chars

        # ✅ Run LangGraph Analysis
        logger.info("🚀 Calling run_investment_analysis()...")
        analysis_result = run_investment_analysis(
            pitch_analysis,
            {},
            startup_name
        ) or {}
        logger.info("✅ LangGraph analysis completed!")

        # ✅ Ensure response is dictionary format
        if not isinstance(analysis_result, dict):
            analysis_result = vars(analysis_result)
        executive_summary = format_executive_summary(analysis_result) or {}
        detailed_analysis = format_detailed_analysis(analysis_result) or {}

        # ✅ Store results in `comparison_results` table
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO comparison_results 
                (s3_url, startup_name, executive_summary, detailed_analysis, 
                raw_comparison, signal_strength, innovation_index, market_pulse, overall_fit, comparison_result, pitch_id, investor_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) 
                RETURNING comparison_id;
            """, (
                request.s3_url,
                startup_name,
                json.dumps(executive_summary),
                json.dumps(detailed_analysis),
                pitch_analysis,
                json.dumps(analysis_result.get("signal_strength", {})),
                json.dumps(analysis_result.get("innovation_index", {})),
                json.dumps(analysis_result.get("market_pulse", {})),
                json.dumps(analysis_result.get("overall_fit_score", {})),
                json.dumps(analysis_result),
                request.pitch_id,
                request.investor_id
            ))

            new_id = cursor.fetchone()[0]
            conn.commit()
            cursor.close()
            conn.close()

        # ✅ Return structured response
        return {
            "comparison_id": new_id,
            "startup_name": startup_name,
            "executive_summary": executive_summary,
            "detailed_analysis": detailed_analysis,
            "pitch_id": request.pitch_id,
            "investor_id": request.investor_id
        }

    except Exception as e:
        logger.error(f"❌ Error in compare-thesis-fit endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Helper functions to fetch data by ID
def get_investor_data_by_id(investor_id):
    """Fetch investor data from database using investor ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM investor_profiles WHERE investor_id = %s", (investor_id,))
            result = cursor.fetchone()
            print(result)
            
            if not result:
                return None
                
            # Convert row to dictionary
            investor_data = dict(result)
            
            # Parse JSON fields if needed
            for field in ['investor_type', 'investment_experience', 'investment_focus_areas','primary_impact_areas','geographical_preferences','startup_stages','business_models','target_roi','exit_horizon', 'check_size_min', 'check_size_max','preferred_ownership', 'revenue_milestones', 'monthly_recurring_revenue', 'monthly_recurring_revenue','tam','som','sam','traction_revenue_market','technology_scalability' ]:
                if investor_data.get(field) and isinstance(investor_data[field], str):
                    investor_data[field] = json.loads(investor_data[field])
                    
            return investor_data
            
    except Exception as e:
        logger.error(f"❌ Error fetching investor data: {str(e)}", exc_info=True)
        return None


def get_pitch_deck_by_id(pitch_id):
    """Fetch pitch deck data from database using pitch deck ID"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cursor.execute("SELECT * FROM pitch_summary WHERE pitch_id = %s", (pitch_id,))
            result = cursor.fetchone()
            
            if not result:
                return None
                
            # Convert row to dictionary
            pitch_data = dict(result)
            
            # Parse JSON fields if needed
            for field in ['pitch_id', 'pitch_deck_url', 'investor_id']:
                if pitch_data.get(field) and isinstance(pitch_data[field], str):
                    pitch_data[field] = json.loads(pitch_data[field])
                    
            return pitch_data
            
    except Exception as e:
        logger.error(f"❌ Error fetching pitch deck data: {str(e)}", exc_info=True)
        return None





@app.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: int):
    """Get the complete analysis for a specific comparison."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            # First, identify the primary key column name
            cursor.execute("""
                SELECT a.attname
                FROM   pg_index i
                JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                     AND a.attnum = ANY(i.indkey)
                WHERE  i.indrelid = 'comparison_results'::regclass
                AND    i.indisprimary;
            """)
            
            result = cursor.fetchone()
            if not result:
                # If we can't determine the primary key, try common options
                pk_column = None
                for possible_pk in ['id', 'comparison_id', 'comparison_results_id']:
                    cursor.execute(f"""
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_name = 'comparison_results' AND column_name = '{possible_pk}'
                        );
                    """)
                    if cursor.fetchone()[0]:
                        pk_column = possible_pk
                        break
                        
                if not pk_column:
                    raise HTTPException(status_code=500, detail="Could not identify primary key column")
            else:
                pk_column = result[0]
                
            logger.info(f"Using '{pk_column}' as primary key column for comparison_results")
            
            # Now use the identified primary key in the query
            query = f"""
                SELECT 
                    {pk_column} as id, 
                    startup_name, 
                    comparison_result,
                    executive_summary,
                    detailed_analysis,
                    thesis_fit_score
                FROM comparison_results 
                WHERE {pk_column} = %s
            """
            
            cursor.execute(query, (analysis_id,))
            
            result = cursor.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Analysis not found")
            
            # Extract the components
            id, startup_name, comparison_result, executive_summary, detailed_analysis, thesis_fit_score = result
            
            # Parse JSON fields - handle both string and JSON objects
            if comparison_result and isinstance(comparison_result, str):
                try:
                    comparison_result = json.loads(comparison_result)
                except json.JSONDecodeError:
                    logger.warning(f"?? Could not parse comparison_result as JSON for ID {analysis_id}")
                    
            if executive_summary and isinstance(executive_summary, str):
                try:
                    executive_summary = json.loads(executive_summary)
                except json.JSONDecodeError:
                    logger.warning(f"?? Could not parse executive_summary as JSON for ID {analysis_id}")
                    
            if detailed_analysis and isinstance(detailed_analysis, str):
                try:
                    detailed_analysis = json.loads(detailed_analysis)
                except json.JSONDecodeError:
                    logger.warning(f"?? Could not parse detailed_analysis as JSON for ID {analysis_id}")
            
            # Ensure startup_name is a plain string, not JSON
            if startup_name and isinstance(startup_name, str):
                # Check if it's a JSON string like "\"Unknown Startup\""
                if startup_name.startswith('"') and startup_name.endswith('"'):
                    try:
                        startup_name = json.loads(startup_name)
                    except json.JSONDecodeError:
                        # Keep it as is if it's not valid JSON
                        pass
            
            # Return the complete analysis
            return {
                "id": id,
                "startup_name": startup_name,
                "thesis_fit_score": thesis_fit_score,
                "executive_summary": executive_summary,
                "detailed_analysis": detailed_analysis,
                "full_result": comparison_result
            }
            
    except Exception as e:
        logger.error(f"? Error fetching analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching analysis: {str(e)}")

@app.get("/data-room/")
async def get_data_room(
    skip: int = 0, 
    limit: int = 20,
    min_score: Optional[int] = None,
    max_score: Optional[int] = None,
    stages: Optional[str] = None
):
    """List all analyzed startups in the data room with their thesis fit scores."""
    try:
        # Build the query with filters
        query = """
            SELECT 
                cr.id, 
                cr.startup_name, 
                cr.s3_url, 
                cr.thesis_fit_score,
                dr.one_liner,
                cr.created_at
            FROM comparison_results cr
            LEFT JOIN data_room dr ON cr.id = dr.comparison_id
            WHERE 1=1
        """
        params = []
        
        # Add filters if specified
        if min_score is not None:
            query += " AND cr.thesis_fit_score >= %s"
            params.append(min_score)
            
        if max_score is not None:
            query += " AND cr.thesis_fit_score <= %s"
            params.append(max_score)
            
        if stages:
            # Convert comma-separated string to array
            stage_list = stages.split(',')
            placeholders = ', '.join(['%s'] * len(stage_list))
            query += f" AND cr.investor_data->'startup_stages' ?| array[{placeholders}]"
            params.extend(stage_list)
            
        # Add ordering, pagination
        query += " ORDER BY cr.created_at DESC OFFSET %s LIMIT %s"
        params.extend([skip, limit])
        
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(query, params)
            startups = cursor.fetchall()
            
            # Get the total count with the same filters but without pagination
            count_query = query.split("ORDER BY")[0].replace("cr.id, cr.startup_name, cr.s3_url, cr.thesis_fit_score, dr.one_liner, cr.created_at", "COUNT(*)")
            count_params = params[:-2]  # Remove the offset and limit params
            
            cursor.execute(count_query, count_params)
            total_count = cursor.fetchone()["count"]
            
            return {
                "startups": startups,
                "total_count": total_count,
                "skip": skip,
                "limit": limit,
                "filters": {
                    "min_score": min_score,
                    "max_score": max_score,
                    "stages": stages
                }
            }
            
    except Exception as e:
        logger.error(f"? Error fetching data room: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching data room: {str(e)}")

@app.get("/thesis-fit-breakdown/{analysis_id}")
async def get_thesis_fit_breakdown(analysis_id: int):
    """Get the breakdown of thesis fit scores for visualization."""
    try:
        with get_db_connection() as conn:
            # First, identify the primary key column name - use a regular cursor for this
            with conn.cursor() as schema_cursor:
                schema_cursor.execute("""
                    SELECT a.attname
                    FROM   pg_index i
                    JOIN   pg_attribute a ON a.attrelid = i.indrelid
                                         AND a.attnum = ANY(i.indkey)
                    WHERE  i.indrelid = 'comparison_results'::regclass
                    AND    i.indisprimary;
                """)
                
                result = schema_cursor.fetchone()
                if not result:
                    # If we can't determine the primary key, try common options
                    pk_column = None
                    for possible_pk in ['id', 'comparison_id', 'comparison_results_id']:
                        schema_cursor.execute(f"""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'comparison_results' AND column_name = '{possible_pk}'
                            );
                        """)
                        if schema_cursor.fetchone()[0]:
                            pk_column = possible_pk
                            break
                            
                    if not pk_column:
                        raise HTTPException(status_code=500, detail="Could not identify primary key column")
                else:
                    pk_column = result[0]
            
            logger.info(f"Using '{pk_column}' as primary key column for comparison_results")
            
            # Now use a RealDictCursor for the actual data query
            with conn.cursor(cursor_factory=RealDictCursor) as data_cursor:
                # Use the identified primary key in the query
                query = f"""
                    SELECT 
                        thesis_fit_score,
                        market_fit_score->>'score' as market_fit_score,
                        traction_score->>'score' as traction_score,
                        tech_maturity_score->>'score' as tech_maturity_score,
                        team_strength_score->>'score' as team_strength_score,
                        financial_viability_score->>'score' as financial_viability_score,
                        startup_name
                    FROM comparison_results
                    WHERE {pk_column} = %s
                """
                
                data_cursor.execute(query, (analysis_id,))
                
                result = data_cursor.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail="Analysis not found")
                
                # Convert string scores to numbers
                result["market_fit_score"] = float(result["market_fit_score"]) if result["market_fit_score"] else 0
                result["traction_score"] = float(result["traction_score"]) if result["traction_score"] else 0
                result["tech_maturity_score"] = float(result["tech_maturity_score"]) if result["tech_maturity_score"] else 0
                result["team_strength_score"] = float(result["team_strength_score"]) if result["team_strength_score"] else 0
                result["financial_viability_score"] = float(result["financial_viability_score"]) if result["financial_viability_score"] else 0
                
                # Calculate weighted components for visualization
                result["weighted_components"] = {
                    "Market Fit (25%)": result["market_fit_score"] * 0.25,
                    "Traction (20%)": result["traction_score"] * 0.20,
                    "Tech & Product (20%)": result["tech_maturity_score"] * 0.20,
                    "Team (15%)": result["team_strength_score"] * 0.15,
                    "Financial (20%)": result["financial_viability_score"] * 0.20
                }
                
                return result
                
    except Exception as e:
        logger.error(f"? Error fetching thesis fit breakdown: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error fetching thesis fit breakdown: {str(e)}")
        
@app.on_event("startup")
async def startup_event():
    """Run on application startup to ensure database schema is correct."""
    logger.info("? Running database schema fix...")
    try:
        fix_database_schema()
        logger.info("? Database schema validated successfully")
    except Exception as e:
        logger.error(f"? Error fixing database schema: {str(e)}", exc_info=True)