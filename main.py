# main.py - Main Application Entry Point with Agent Integration

import os
import uvicorn
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)

logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import database utilities
from database_utils import create_tables, fix_database_schema, fix_database_constraints

# Import existing routes
# Keep your existing authentication, investor profile, and LinkedIn scraping routes
from existing_routes import authentication_routes, investor_routes, linkedin_routes

# Import the new agent-based routes
from fastapi_routes import (
    upload_pitch,
    compare_thesis_fit,
    get_analysis, 
    get_data_room,
    get_thesis_fit_breakdown
)

# Register authentication routes
app.include_router(authentication_routes)

# Register investor profile routes
app.include_router(investor_routes)

# Register LinkedIn scraping routes
app.include_router(linkedin_routes)

# Register new agent-based routes
app.post("/upload-pitch/")(upload_pitch)
app.post("/compare-thesis-fit/")(compare_thesis_fit)
app.get("/analysis/{analysis_id}")(get_analysis)
app.get("/data-room/")(get_data_room)
app.get("/thesis-fit-breakdown/{analysis_id}")(get_thesis_fit_breakdown)

@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    # Create database tables if they don't exist
    create_tables()
    
    # Fix database schema issues
    fix_database_schema()
    
    # Fix database constraints
    fix_database_constraints()
    
    logger.info("✅ Application started successfully with database fixes applied")

# Main entry point
if __name__ == "__main__":
    # Run the FastAPI app
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)