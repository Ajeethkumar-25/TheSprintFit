# Database and Utility Functions for Investment Analysis System

import fitz  # PyMuPDF
import pytesseract
import requests
import logging
import json
import io
from PIL import Image
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Connection pool from your original code
from psycopg2.pool import SimpleConnectionPool

# Use your existing DB configuration
DB_CONFIG = {
    "dbname": "investor_db",
    "user": "investor_user",
    "password": "investor_password",
    "host": "34.193.248.230",
    "port": "5432"
}

pool = SimpleConnectionPool(minconn=1, maxconn=10, **DB_CONFIG)

@contextmanager
def get_db_connection():
    """Database connection context manager."""
    conn = None
    try:
        conn = pool.getconn()
        yield conn
    finally:
        if conn:
            pool.putconn(conn)

def create_tables():
    """Create necessary database tables if they don't exist."""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Update the comparison_results table to support the new agent-based analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS comparison_results (
                    id SERIAL PRIMARY KEY,
                    investor_data JSONB NOT NULL,
                    s3_url TEXT NOT NULL,
                    startup_name TEXT NOT NULL DEFAULT 'Unknown Startup',
                    comparison_result JSONB,
                    executive_summary JSONB,
                    detailed_analysis JSONB,
                    market_fit_score JSONB,
                    traction_score JSONB,
                    tech_maturity_score JSONB,
                    team_strength_score JSONB,
                    financial_viability_score JSONB,
                    thesis_fit_score INTEGER,
                    raw_comparison TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create pitch_summaries table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pitch_summaries (
                    id SERIAL PRIMARY KEY,
                    s3_url TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create a new table for tracking missing data in pitch decks
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS missing_data_flags (
                    id SERIAL PRIMARY KEY,
                    comparison_id INTEGER REFERENCES comparison_results(id),
                    section_name TEXT NOT NULL,
                    is_missing BOOLEAN DEFAULT FALSE,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create a new table for the Data Room
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_room (
                    id SERIAL PRIMARY KEY,
                    comparison_id INTEGER REFERENCES comparison_results(id),
                    startup_name TEXT NOT NULL,
                    thesis_fit_score INTEGER,
                    one_liner TEXT,
                    tags TEXT[],
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            logger.info("✅ Database tables created/verified")

def download_pdf_from_s3(s3_url: str) -> bytes:
    """Downloads a PDF from S3 and returns its byte content."""
    try:
        logger.info(f"Downloading PDF from: {s3_url}")
        response = requests.get(s3_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to download PDF from S3. Status code: {response.status_code}")
            raise Exception(f"Failed to download PDF from S3. Status code: {response.status_code}")
        
        logger.info(f"Successfully downloaded PDF ({len(response.content)} bytes)")
        return response.content
    except Exception as e:
        logger.error(f"Error downloading PDF from S3: {str(e)}", exc_info=True)
        raise

def is_image_pdf(pdf_bytes):
    """Check if a PDF is an image-based PDF by detecting empty text pages."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page in doc:
            if page.get_text().strip():
                return False
        return True
    except Exception as e:
        logger.error(f"Error checking if PDF is image-based: {str(e)}", exc_info=True)
        # If we can't determine, assume it's not an image PDF
        return False

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Extract text from a PDF file, handling both text-based and image-based PDFs.
    
    Args:
        pdf_bytes (bytes): PDF content as bytes
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        logger.info(f"PDF opened successfully with {total_pages} pages")

        # Check if it's an image-based PDF
        if is_image_pdf(pdf_bytes):
            logger.info("Processing image-based PDF with OCR")
            for page_num in range(total_pages):
                try:
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    
                    # Convert Pixmap to an Image without saving to disk
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = pytesseract.image_to_string(img)
                    text += page_text + "\n"
                    
                    logger.info(f"Processed page {page_num+1}/{total_pages} with OCR")
                except Exception as e:
                    logger.error(f"Error processing page {page_num+1} with OCR: {str(e)}")
                    continue
        else:
            logger.info("Processing text-based PDF")
            for page_num, page in enumerate(doc):
                try:
                    page_text = page.get_text("text")
                    text += page_text + "\n"
                    logger.info(f"Processed page {page_num+1}/{total_pages}")
                except Exception as e:
                    logger.error(f"Error extracting text from page {page_num+1}: {str(e)}")
                    continue
        
        final_text = text.strip()
        logger.info(f"Successfully extracted {len(final_text)} characters from PDF")
        return final_text
    
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}", exc_info=True)
        return ""
    
def fix_database_schema():
    """
    Update the database schema to fix the NOT NULL constraint issue and ensure compatibility
    with the new analysis system.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # First, check if the comparison_results table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'comparison_results'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if table_exists:
                # Alter the comparison_results table to allow NULL for comparison_result temporarily
                # This is a safer approach than dropping the table
                cursor.execute("""
                    ALTER TABLE comparison_results 
                    ALTER COLUMN comparison_result DROP NOT NULL;
                """)
                
                # Add any missing columns that are needed for the new system
                # First check if columns exist
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'comparison_results';
                """)
                existing_columns = [col[0] for col in cursor.fetchall()]
                
                # Add missing columns if they don't exist
                if 'startup_name' not in existing_columns:
                    cursor.execute("""
                        ALTER TABLE comparison_results 
                        ADD COLUMN startup_name TEXT DEFAULT 'Unknown Startup';
                    """)
                
                if 'thesis_fit_score' not in existing_columns:
                    cursor.execute("""
                        ALTER TABLE comparison_results 
                        ADD COLUMN thesis_fit_score INTEGER;
                    """)
                
                if 'executive_summary' not in existing_columns:
                    cursor.execute("""
                        ALTER TABLE comparison_results 
                        ADD COLUMN executive_summary JSONB;
                    """)
                
                if 'detailed_analysis' not in existing_columns:
                    cursor.execute("""
                        ALTER TABLE comparison_results 
                        ADD COLUMN detailed_analysis JSONB;
                    """)
                
                if 'created_at' not in existing_columns:
                    cursor.execute("""
                        ALTER TABLE comparison_results 
                        ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP;
                    """)
                
                # Add component score columns
                for component in ['market_fit_score', 'traction_score', 'tech_maturity_score', 
                                'team_strength_score', 'financial_viability_score']:
                    if component not in existing_columns:
                        cursor.execute(f"""
                            ALTER TABLE comparison_results 
                            ADD COLUMN {component} JSONB;
                        """)
            else:
                # Create the table if it doesn't exist
                cursor.execute("""
                    CREATE TABLE comparison_results (
                        id SERIAL PRIMARY KEY,
                        investor_data JSONB NOT NULL,
                        s3_url TEXT NOT NULL,
                        startup_name TEXT DEFAULT 'Unknown Startup',
                        comparison_result JSONB,
                        executive_summary JSONB,
                        detailed_analysis JSONB,
                        market_fit_score JSONB,
                        traction_score JSONB,
                        tech_maturity_score JSONB,
                        team_strength_score JSONB,
                        financial_viability_score JSONB,
                        thesis_fit_score INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
            
            # Commit the changes
            conn.commit()
            print("✅ Database schema updated successfully")

def inspect_table_columns(table_name):
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

def fix_database_constraints():
    """
    Add missing constraints to database tables to support ON CONFLICT operations.
    """
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # First check if the missing_data_flags table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'missing_data_flags'
                );
            """)
            missing_data_flags_exists = cursor.fetchone()[0]
            
            if missing_data_flags_exists:
                # Check if the constraint already exists
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM pg_constraint
                    WHERE conname = 'missing_data_flags_comparison_id_section_name_key';
                """)
                constraint_exists = cursor.fetchone()[0] > 0
                
                if not constraint_exists:
                    # Add composite unique constraint for ON CONFLICT support
                    try:
                        cursor.execute("""
                            ALTER TABLE missing_data_flags
                            ADD CONSTRAINT missing_data_flags_comparison_id_section_name_key
                            UNIQUE (comparison_id, section_name);
                        """)
                        logger.info("✅ Added unique constraint to missing_data_flags table")
                    except Exception as e:
                        logger.error(f"❌ Error adding constraint to missing_data_flags: {str(e)}")
            
            # Check if the data_room table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'data_room'
                );
            """)
            data_room_exists = cursor.fetchone()[0]
            
            if data_room_exists:
                # Check if the constraint already exists
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM pg_constraint
                    WHERE conname = 'data_room_comparison_id_key';
                """)
                constraint_exists = cursor.fetchone()[0] > 0
                
                if not constraint_exists:
                    # Add unique constraint for comparison_id
                    try:
                        cursor.execute("""
                            ALTER TABLE data_room
                            ADD CONSTRAINT data_room_comparison_id_key
                            UNIQUE (comparison_id);
                        """)
                        logger.info("✅ Added unique constraint to data_room table")
                    except Exception as e:
                        logger.error(f"❌ Error adding constraint to data_room: {str(e)}")
            
            # Commit the changes
            conn.commit()
            logger.info("✅ Database constraints fixed successfully")

def store_analysis_result(comparison_id, analysis_result):
    """Store the analysis result in the database."""
    try:
        # Handle the case where startup_name might be missing
        if "startup_name" not in analysis_result:
            analysis_result["startup_name"] = "Unknown Startup"
            
        # Ensure startup_name is a simple string, not a dictionary or JSON
        startup_name = analysis_result.get("startup_name")
        if isinstance(startup_name, dict):
            # If it's a dictionary, extract a meaningful value or use default
            if "name" in startup_name:
                startup_name = startup_name["name"]
            else:
                startup_name = next(iter(startup_name.values()), "Unknown Startup")

        # Get table columns first
        columns = inspect_table_columns('comparison_results')
        
        # Create a dict of updates
        updates = {}
        
        # Add fields if they exist in the schema
        if 'comparison_result' in columns:
            updates['comparison_result'] = json.dumps(analysis_result)
        if 'executive_summary' in columns:
            updates['executive_summary'] = json.dumps(analysis_result.get("executive_summary", {}))
        if 'detailed_analysis' in columns:
            updates['detailed_analysis'] = json.dumps(analysis_result.get("detailed_analysis", {}))
        if 'market_fit_score' in columns:
            updates['market_fit_score'] = json.dumps(analysis_result.get("detailed_analysis", {}).get("Market Fit", {}))
        if 'traction_score' in columns:
            updates['traction_score'] = json.dumps(analysis_result.get("detailed_analysis", {}).get("Traction", {}))
        if 'tech_maturity_score' in columns:
            updates['tech_maturity_score'] = json.dumps(analysis_result.get("detailed_analysis", {}).get("Tech & Product Maturity", {}))
        if 'team_strength_score' in columns:
            updates['team_strength_score'] = json.dumps(analysis_result.get("detailed_analysis", {}).get("Team Strength", {}))
        if 'financial_viability_score' in columns:
            updates['financial_viability_score'] = json.dumps(analysis_result.get("detailed_analysis", {}).get("Financial Viability", {}))
        if 'thesis_fit_score' in columns:
            updates['thesis_fit_score'] = analysis_result.get("thesis_fit_score", 0)
        if 'startup_name' in columns:
            updates['startup_name'] = startup_name
            
        # Create the SQL query parts
        set_parts = [f"{col} = %s" for col in updates.keys()]
        set_sql = ", ".join(set_parts)
        values = list(updates.values())
        values.append(comparison_id)  # For the WHERE clause
        
        # Determine the primary key column
        pk_column = None
        for possible_pk in ['id', 'comparison_id', 'comparison_results_id']:
            if possible_pk in columns:
                pk_column = possible_pk
                break
                
        if not pk_column:
            logger.error("❌ Could not identify primary key column in comparison_results table")
            return False
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Update the comparison result
                update_query = f"""
                    UPDATE comparison_results
                    SET {set_sql}
                    WHERE {pk_column} = %s
                """
                
                logger.info(f"Executing UPDATE: {update_query}")
                cursor.execute(update_query, values)
                
                # Only proceed with missing_data_flags if that table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'missing_data_flags'
                    );
                """)
                missing_data_flags_exists = cursor.fetchone()[0]
                
                if missing_data_flags_exists:
                    # Store missing data flags
                    missing_flags = analysis_result.get("missing_data_flags", {})
                    for section, is_missing in missing_flags.items():
                        cursor.execute("""
                            INSERT INTO missing_data_flags
                            (comparison_id, section_name, is_missing)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (comparison_id, section_name) 
                            DO UPDATE SET is_missing = EXCLUDED.is_missing
                        """, (
                            comparison_id,
                            section,
                            is_missing
                        ))
                
                # Only proceed with data_room if that table exists
                cursor.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'data_room'
                    );
                """)
                data_room_exists = cursor.fetchone()[0]
                
                if data_room_exists:
                    # Get data_room columns
                    data_room_cols = inspect_table_columns('data_room')
                    
                    # Only proceed if we have the necessary columns
                    if 'startup_name' in data_room_cols and 'comparison_id' in data_room_cols:
                        # Check if a record already exists
                        cursor.execute("""
                            SELECT id FROM data_room WHERE comparison_id = %s
                        """, (comparison_id,))
                        
                        existing_record = cursor.fetchone()
                        
                        if existing_record:
                            # Update existing record
                            cursor.execute("""
                                UPDATE data_room
                                SET startup_name = %s,
                                    thesis_fit_score = %s,
                                    one_liner = %s,
                                    last_updated = CURRENT_TIMESTAMP
                                WHERE comparison_id = %s
                            """, (
                                startup_name,
                                analysis_result.get("thesis_fit_score", 0),
                                analysis_result.get("executive_summary", {}).get("Executive Summary", "")[:100] + "...",
                                comparison_id
                            ))
                        else:
                            # Insert new record
                            cursor.execute("""
                                INSERT INTO data_room
                                (comparison_id, startup_name, thesis_fit_score, one_liner)
                                VALUES (%s, %s, %s, %s)
                            """, (
                                comparison_id,
                                startup_name,
                                analysis_result.get("thesis_fit_score", 0),
                                analysis_result.get("executive_summary", {}).get("Executive Summary", "")[:100] + "..."
                            ))
                
                conn.commit()
                logger.info(f"✅ Analysis result stored for comparison ID: {comparison_id}")
                return True
    except Exception as e:
        logger.error(f"❌ Error storing analysis result: {str(e)}", exc_info=True)
        return False