# existing_routes/linkedin_routes.py
# Contains LinkedIn scraping functionality from the original code

from fastapi import FastAPI, HTTPException, UploadFile, File, Form , Depends, Body, status, APIRouter
from pydantic import BaseModel
import os
import json
import re
import time
import logging
import platform
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import DB connection
from database_utils import get_db_connection
from psycopg2.extras import RealDictCursor


# Setup webdriver and scraping tools
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service


# OpenAI import
from openai import OpenAI

# Initialize router
router = APIRouter(tags=["LinkedIn Scraping"])

# Configure logging
log = logging.getLogger("linkedin_scraper")
log.setLevel(logging.INFO)

# Create a file handler for LinkedIn scraper logs
linkedin_handler = logging.FileHandler("linkedin_scraper.log")
linkedin_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))

# Create a stream handler for console output
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))

# Attach handlers to LinkedIn logger
log.addHandler(linkedin_handler)
log.addHandler(stream_handler)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic model
class LinkedInRequest(BaseModel):
    linkedin_url: str
    email: str

def scrape_linkedin_profile(url):
    log.info(f"Starting to scrape LinkedIn profile: {url}")
    
    # Validate URL format
    if not url.startswith("https://www.linkedin.com/") or len(url) < 30:
        log.error(f"Invalid LinkedIn URL: {url}")
        raise HTTPException(status_code=400, detail="Invalid LinkedIn URL format")
    
    # Create a unique directory for screenshots and debugging
    unique_dir = f"/tmp/chrome_data_{int(time.time())}"
    if not os.path.exists(unique_dir):
        os.makedirs(unique_dir)
    
    # Set up enhanced Chrome options for better security bypass
    options = webdriver.ChromeOptions()
    options.add_argument(f"--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument(f"--user-data-dir={unique_dir}/chrome_profile")
    
    # Initialize Chrome driver with logging
    service = Service("/usr/local/bin/chromedriver", log_path=f"{unique_dir}/chromedriver.log")
    driver = None
    
    # Use multiple accounts to prevent lockouts
    linkedin_accounts = [
        {"email": "praveenchellapandian395@gmail.com", "password": "iZCsBT2tj*rVx2R"},
        # Add backup accounts if available
        # {"email": BACKUP_EMAIL, "password": BACKUP_PASSWORD},
    ]
    
    # Track which account is being used
    current_account_index = 0
    max_retries = 2  # Limit retries to avoid wasting resources
    
    for attempt in range(max_retries):
        try:
            # Select an account - rotate through available accounts to avoid lockouts
            account = linkedin_accounts[current_account_index % len(linkedin_accounts)]
            current_email = account["email"]
            current_password = account["password"]
            
            log.info(f"Attempt {attempt + 1}/{max_retries} using account {current_email[:3]}***")
            
            # Ensure credentials are available
            if not current_email or not current_password:
                log.error("LinkedIn credentials not configured")
                raise HTTPException(status_code=500, detail="LinkedIn credentials not properly configured")
            
            # Initialize new browser instance
            if driver:
                driver.quit()
                
            driver = webdriver.Chrome(service=service, options=options)
            
            # Add custom JS to modify navigator properties
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                Object.defineProperty(navigator, 'webdriver', { get: () => false });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
                """
            })
            
            # Set a longer timeout for page loads
            driver.set_page_load_timeout(45)
            
            # Add randomized delays between operations to appear human-like
            def random_sleep(min_seconds=1, max_seconds=3):
                time.sleep(random.uniform(min_seconds, max_seconds))
            
            # Add human-like typing function
            def human_type(element, text):
                for char in text:
                    element.send_keys(char)
                    time.sleep(random.uniform(0.05, 0.2))  # Random delay between keypresses

            # Add random mouse movements to appear more human-like
            def move_mouse_randomly(driver):
                try:
                    action = ActionChains(driver)
                    for _ in range(random.randint(2, 5)):
                        x = random.randint(100, 700)
                        y = random.randint(100, 500)
                        action.move_by_offset(x, y)
                        action.perform()
                        time.sleep(random.uniform(0.1, 0.3))
                    action.move_to_element(driver.find_element(By.TAG_NAME, 'body'))
                    action.perform()
                except Exception as e:
                    log.warning(f"Mouse movement simulation failed: {str(e)}")
            
            # Try an alternative login approach to bypass security
            log.info("Using alternative login approach")
            
            # First, visit the LinkedIn homepage to set cookies
            driver.get('https://www.linkedin.com/')
            random_sleep(2, 4)
            
            # Move mouse randomly
            move_mouse_randomly(driver)
            
            # Sometimes LinkedIn redirects immediately to login
            if "login" not in driver.current_url:
                log.info("Navigating to LinkedIn login page")
                try:
                    # Find and click login button if on homepage
                    login_btns = driver.find_elements(By.CSS_SELECTOR, "a[data-tracking-control-name='guest_homepage-basic_nav-header-signin']")
                    if login_btns:
                        action = ActionChains(driver)
                        action.move_to_element(login_btns[0])
                        action.perform()
                        random_sleep(0.5, 1.5)
                        login_btns[0].click()
                    else:
                        # Direct navigation if button not found
                        driver.get('https://www.linkedin.com/login')
                except Exception as e:
                    log.warning(f"Error clicking login button: {str(e)}")
                    driver.get('https://www.linkedin.com/login')
            
            random_sleep(3, 5)
            driver.save_screenshot(f"{unique_dir}/login_page.png")
            
            # Wait for login fields
            try:
                username_field = WebDriverWait(driver, 15).until(
                    EC.presence_of_element_located((By.ID, "username"))
                )
                log.info("LinkedIn login page loaded successfully")
            except TimeoutException:
                log.error("Could not load LinkedIn login page properly")
                driver.save_screenshot(f"{unique_dir}/login_error.png")
                raise HTTPException(status_code=500, detail="Failed to load LinkedIn login page")
                
            log.info("Entering login credentials")
            
            # Clear and enter email with human-like typing
            email_field = driver.find_element(By.ID, 'username')
            email_field.clear()
            human_type(email_field, current_email)
            
            random_sleep(1, 2)
            
            # Clear and enter password with human-like typing
            password_field = driver.find_element(By.ID, 'password')
            password_field.clear()
            human_type(password_field, current_password)
            
            random_sleep(1, 3)
            
            # Find and click sign-in button
            sign_in_button = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            
            # Move mouse to login button with natural movement
            action = ActionChains(driver)
            action.move_to_element(sign_in_button)
            action.perform()
            random_sleep(0.5, 1.5)
            
            sign_in_button.click()
            log.info("Login credentials submitted")
            
            # Make longer wait after login to allow full page load
            random_sleep(3, 6)
            driver.save_screenshot(f"{unique_dir}/after_login_attempt.png")
            
            # Complex checkpoint handling - attempt to bypass if possible
            if "checkpoint" in driver.current_url:
                log.warning("LinkedIn security checkpoint detected - attempting bypass")
                driver.save_screenshot(f"{unique_dir}/security_checkpoint.png")
                
                # Save the HTML for analysis
                with open(f"{unique_dir}/checkpoint_page.html", "w", encoding="utf-8") as f:
                    f.write(driver.page_source)
                
                # Try different bypass strategies based on checkpoint type
                try:
                    remember_checkboxes = driver.find_elements(By.CSS_SELECTOR, "input[type='checkbox']")
                    if remember_checkboxes:
                        for checkbox in remember_checkboxes:
                            if checkbox.is_displayed():
                                checkbox.click()
                                random_sleep(1, 2)
                except Exception as e:
                    log.warning(f"Checkbox interaction failed: {str(e)}")
                
                try:
                    buttons = driver.find_elements(By.TAG_NAME, "button")
                    for button in buttons:
                        if button.is_displayed() and any(text in button.text.lower() for text in ["continue", "verify", "confirm"]):
                            log.info(f"Attempting to click checkpoint button: {button.text}")
                            button.click()
                            random_sleep(3, 5)
                            break
                except Exception as e:
                    log.warning(f"Button interaction failed: {str(e)}")
                
                # Wait to see if bypass worked
                random_sleep(3, 5)
                driver.save_screenshot(f"{unique_dir}/after_checkpoint_attempt.png")
                
                # If still on checkpoint page, try the next account
                if "checkpoint" in driver.current_url or "authwall" in driver.current_url:
                    log.warning("Checkpoint bypass failed, will try next account")
                    current_account_index += 1
                    continue  # Try next account
            
            # Check for CAPTCHA (attempt to handle simple ones)
            if "captcha" in driver.page_source.lower() or driver.find_elements(By.ID, "captcha-internal"):
                log.warning("CAPTCHA challenge detected - attempting simple bypass")
                driver.save_screenshot(f"{unique_dir}/captcha.png")
                
                # Try to find and interact with CAPTCHA
                try:
                    captcha_checkbox = driver.find_elements(By.CSS_SELECTOR, ".recaptcha-checkbox-checkmark")
                    if captcha_checkbox:
                        captcha_checkbox[0].click()
                        random_sleep(2, 4)
                    
                    verify_buttons = driver.find_elements(By.XPATH, "//button[contains(text(), 'Verify')]")
                    if verify_buttons:
                        verify_buttons[0].click()
                        random_sleep(2, 4)
                except Exception as e:
                    log.warning(f"CAPTCHA interaction failed: {str(e)}")
                
                # Check if we're still on CAPTCHA
                if "captcha" in driver.page_source.lower():
                    log.warning("CAPTCHA bypass failed, will try next account")
                    current_account_index += 1
                    continue  # Try next account
            
            # Wait for login completion with better checking
            try:
                WebDriverWait(driver, 30).until(lambda d: (
                    "login" not in d.current_url and 
                    "authwall" not in d.current_url and
                    "checkpoint" not in d.current_url and
                    d.execute_script("return document.readyState") == "complete"
                ))
                log.info(f"Login successful, redirected to: {driver.current_url}")
                driver.save_screenshot(f"{unique_dir}/after_login.png")
            except TimeoutException:
                log.error("Login process timed out")
                driver.save_screenshot(f"{unique_dir}/login_timeout.png")
                current_url = driver.current_url
                log.error(f"Current URL at timeout: {current_url}")
                
                # If still on problem page, try the next account
                if "login" in current_url or "checkpoint" in current_url or "authwall" in current_url:
                    current_account_index += 1
                    continue  # Try next account
                
                raise HTTPException(status_code=500, detail=f"Timeout during LinkedIn login process. Current URL: {current_url}")
            
            # Check for LinkedIn feed to verify successful login
            if "feed" in driver.current_url or "mynetwork" in driver.current_url:
                log.info("Successfully logged in to LinkedIn")
            else:
                log.warning(f"Login may not be complete - current URL: {driver.current_url}")
            
            # Add cookies and local storage manipulation to further evade detection
            driver.execute_script("""
                localStorage.setItem('li:lms:analytics', JSON.stringify({version: '1.0.0'}));
                localStorage.setItem('li:user:settings', JSON.stringify({version: '1.0.0'}));
            """)
            
            # Add a slight delay before navigating to target profile
            random_sleep(2, 5)
            
            # Navigate to profile with better error handling
            log.info(f"Navigating to target profile: {url}")
            try:
                # Use referrer approach to look more natural
                driver.get('https://www.linkedin.com/feed/')
                random_sleep(2, 4)
                
                # Simulate search bar usage sometimes
                if random.random() > 0.5:
                    try:
                        search_box = driver.find_element(By.CSS_SELECTOR, "input[role='combobox']")
                        human_type(search_box, "LinkedIn")
                        random_sleep(1, 2)
                        search_box.send_keys(Keys.ENTER)
                        random_sleep(2, 4)
                    except Exception as e:
                        log.warning(f"Search bar simulation failed: {str(e)}")
                
                # Navigate to the target profile
                driver.get(url)
                random_sleep(3, 6)
                
                # Wait for profile page to load
                try:
                    WebDriverWait(driver, 30).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.profile-detail"))
                    )
                    log.info("Profile page loaded successfully")
                    driver.save_screenshot(f"{unique_dir}/profile_page.png")
                except TimeoutException:
                    log.error("Failed to load profile page")
                    driver.save_screenshot(f"{unique_dir}/profile_load_error.png")
                    raise HTTPException(status_code=500, detail="Failed to load LinkedIn profile page")
                
                # Extract profile data using BeautifulSoup
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                
                # Extract profile name
                try:
                    profile_name = soup.find("h1", class_="text-heading-xlarge").get_text(strip=True)
                except AttributeError:
                    profile_name = "Not Found"
                
                # Extract headline
                try:
                    headline = soup.find("div", class_="text-body-medium").get_text(strip=True)
                except AttributeError:
                    headline = "Not Found"
                
                # Extract about section
                try:
                    about_section = soup.find("div", class_="display-flex ph5 pv3").get_text(strip=True)
                except AttributeError:
                    about_section = "Not Found"
                
                # Extract experience
                experience = []
                try:
                    experience_sections = soup.find_all("li", class_="experience-item")
                    for exp in experience_sections:
                        title = exp.find("h3", class_="t-16 t-bold").get_text(strip=True)
                        company = exp.find("p", class_="t-14 t-normal").get_text(strip=True)
                        duration = exp.find("span", class_="t-14 t-normal t-black--light").get_text(strip=True)
                        experience.append({
                            "title": title,
                            "company": company,
                            "duration": duration
                        })
                except AttributeError:
                    pass
                
                # Extract education
                education = []
                try:
                    education_sections = soup.find_all("li", class_="education-item")
                    for edu in education_sections:
                        school = edu.find("h3", class_="t-16 t-bold").get_text(strip=True)
                        degree = edu.find("p", class_="t-14 t-normal").get_text(strip=True)
                        duration = edu.find("span", class_="t-14 t-normal t-black--light").get_text(strip=True)
                        education.append({
                            "school": school,
                            "degree": degree,
                            "duration": duration
                        })
                except AttributeError:
                    pass
                
                # Compile extracted data
                profile_data = {
                    "name": profile_name,
                    "headline": headline,
                    "about": about_section,
                    "experience": experience,
                    "education": education
                }
                
                log.info(f"Successfully extracted profile data: {profile_data}")
                
                # Save the extracted data to a JSON file
                with open(f"{unique_dir}/profile_data.json", "w") as f:
                    json.dump(profile_data, f, indent=4)
                
                return profile_data
            
            except Exception as e:
                log.error(f"Error navigating to or extracting data from profile: {str(e)}")
                driver.save_screenshot(f"{unique_dir}/profile_error.png")
                raise HTTPException(status_code=500, detail=f"Error extracting profile data: {str(e)}")
        
        except Exception as e:
            log.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                log.error("All attempts failed")
                raise HTTPException(status_code=500, detail="All attempts to scrape LinkedIn profile failed")
    
        finally:
            # Clean up: close the browser and remove temporary files
            if driver:
                driver.quit()
            if os.path.exists(unique_dir):
                for root, dirs, files in os.walk(unique_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(unique_dir)
                log.info(f"Cleaned up temporary directory: {unique_dir}")

def process_with_openai(profile_data):
    log.info("Starting OpenAI processing of profile data")
    prompt = f"""
    Based on the following LinkedIn profile data, extract structured investment-related details. Make logical inferences where information is not explicitly stated but can be reasonably deduced from their experience, skills, and background.

    ### LinkedIn Profile Data:
    {json.dumps(profile_data, indent=2)}

    ### Extraction Guidelines:
    - Investor Type: Infer from experience (VC, Angel, PE, etc.)
    - Investment Experience: Extract years or use company tenures
    - Investment Focus Areas: Infer from work history, skills, interests
    - Geographical Preferences: Look for regions mentioned in experience
    - Startup Stages: Analyze language used in experience and about section
    - Business Models: Identify patterns in companies they've worked with
    - For monetary values (check sizes, MRR, etc.): Extract if specified, otherwise use "Not specified"
    - Technology areas: Infer from skills, experience, and interests

    ### Extract and categorize into the following format:
    {{
        "investor_type": ["e.g., Venture Capitalist, Angel Investor, Private Equity, etc."],
        "investment_experience": "years of experience or relevant background",
        "investment_focus_areas": ["list of industries they are interested in"],
        "primary_impact_areas": ["main social or environmental impact areas if mentioned"],
        "geographical_preferences": ["regions or countries they prefer to invest in"],
        "startup_stages": ["e.g., Seed, Series A, Series B"],
        "business_models": ["e.g., B2B, B2C, SaaS"],
        "target_roi": "target return on investment (numeric value if mentioned)",
        "exit_horizon": "preferred exit timeline in years",
        "check_size_min": "minimum check size in USD",
        "check_size_max": "maximum check size in USD",
        "preferred_ownership": "preferred ownership percentage",
        "revenue_milestones": "key revenue milestones they look for",
        "monthly_recurring_revenue": "minimum MRR they expect (in USD)",
        "tam": "total addressable market size they prefer (in USD)",
        "sam": "serviceable addressable market size they prefer (in USD)",
        "som": "serviceable obtainable market size they prefer (in USD)",
        "traction_revenue_market": "description of traction, revenue, or market metrics they value",
        "technology_scalability": ["list of tech scalability factors they consider"]
    }}

    Important:
    - If information is not explicitly available, make logical inferences based on their background
    - For fields where no inference can be made with confidence, use "Not specified"
    - If you see terms like "pre-seed", "seed", "Series A", "early-stage", etc. in their profile, include them in startup_stages
    - Look for industries mentioned in their experience and include them in investment_focus_areas
    - Look for technology areas mentioned in skills or experience and include in technology_scalability
    """

    log.info("Sending request to OpenAI API")
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # Consider using a more capable model if available
            messages=[
                {"role": "system", "content": "You are an expert at extracting investment patterns from professional profiles. Make reasonable inferences from profile data when explicit statements aren't available."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Slightly higher temperature for reasonable inferences
        )
        
        log.info("Received response from OpenAI API")
        # Process the response
        content = response.choices[0].message.content
        log.info(f"Raw response content (first 100 chars): {content[:100]}...")
        
        # Save the raw response for inspection
        with open(f"openai_response_{int(time.time())}.txt", "w") as f:
            f.write(content)
            log.info("Saved raw OpenAI response to file")
        
        # Extract JSON from response
        json_pattern = r'```(?:json)?(.*?)```'
        json_match = re.search(json_pattern, content, re.DOTALL)
        
        if json_match:
            log.info("Found JSON in code block format")
            content = json_match.group(1).strip()
        
        try:
            parsed_data = json.loads(content)
            log.info("Successfully parsed JSON response")
            # Save the parsed data for inspection
            with open(f"parsed_data_{int(time.time())}.json", "w") as f:
                json.dump(parsed_data, f, indent=2)
                log.info("Saved parsed data to file")
            return parsed_data
        except json.JSONDecodeError as e:
            log.warning(f"JSON parse error: {str(e)}, trying alternative extraction")
            # If we still can't parse the JSON, try to extract it from the text
            json_pattern = r'{.*}'
            json_match = re.search(json_pattern, content, re.DOTALL)
            
            if json_match:
                log.info("Found JSON using pattern matching")
                try:
                    parsed_data = json.loads(json_match.group(0))
                    log.info("Successfully parsed JSON from pattern match")
                    # Save the parsed data for inspection
                    with open(f"parsed_data_alternative_{int(time.time())}.json", "w") as f:
                        json.dump(parsed_data, f, indent=2)
                        log.info("Saved alternatively parsed data to file")
                    return parsed_data
                except json.JSONDecodeError as e2:
                    log.error(f"Alternative JSON parsing failed: {str(e2)}")
                    raise HTTPException(status_code=500, detail=f"Could not parse response as JSON: {e2}")
            else:
                log.error("No JSON pattern found in response")
                raise HTTPException(status_code=500, detail=f"Could not parse response as JSON: {content}")
    
    except Exception as e:
        log.error(f"OpenAI API error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

def clean_numeric(value, is_int=False):
    """Ensures numbers are correctly parsed, and replaces None values with 0.0 (float) or 0 (int)."""
    if value is None:  
        return 0.0 if not is_int else 0  # ✅ Ensure None is replaced with 0.0 (float) or 0 (int)
    
    if isinstance(value, (int, float)):
        return float(value) if not is_int else int(value)
    
    if isinstance(value, str):
        value = value.replace("$", "").replace(",", "").strip()
        if value.replace(".", "", 1).isdigit():  # Allow decimal numbers
            return float(value) if not is_int else int(float(value))

    return 0.0 if not is_int else 0  # ✅ Default to 0.0 for float fields

def clean_investor_data(investor_data):
    """Converts lists to strings and ensures proper numeric values."""
    return {
        "investor_type": ", ".join(investor_data.get("investor_type", [])) if isinstance(investor_data.get("investor_type"), list) else investor_data.get("investor_type", ""),
        "investment_experience": investor_data.get("investment_experience", ""),
        "investment_focus_areas": investor_data.get("investment_focus_areas", []),
        "primary_impact_areas": ", ".join(investor_data.get("primary_impact_areas", [])) if isinstance(investor_data.get("primary_impact_areas"), list) else investor_data.get("primary_impact_areas", ""),
        "geographical_preferences": investor_data.get("geographical_preferences", []),
        "startup_stages": investor_data.get("startup_stages", []),
        "business_models": investor_data.get("business_models", []),
        "target_roi": clean_numeric(investor_data.get("target_roi")),  # ✅ Now always a valid float (0.0 if None)
        "exit_horizon": clean_numeric(investor_data.get("exit_horizon"), is_int=True),
        "check_size_min": clean_numeric(investor_data.get("check_size_min"), is_int=True),
        "check_size_max": clean_numeric(investor_data.get("check_size_max"), is_int=True),
        "preferred_ownership": clean_numeric(investor_data.get("preferred_ownership")),  # ✅ Now always a valid float (0.0 if None)
        "revenue_milestones": investor_data.get("revenue_milestones", ""),
        "monthly_recurring_revenue": clean_numeric(investor_data.get("monthly_recurring_revenue"), is_int=True),
        "tam": clean_numeric(investor_data.get("tam"), is_int=True),
        "sam": clean_numeric(investor_data.get("sam"), is_int=True),
        "som": clean_numeric(investor_data.get("som"), is_int=True),
        "traction_revenue_market": investor_data.get("traction_revenue_market", ""),
        "technology_scalability": investor_data.get("technology_scalability", []),
    }

@router.post("/scrape-investor")
def scrape_investor(data: LinkedInRequest):
    log.info(f"Received scrape request for URL: {data.linkedin_url}")

    try:
        log.info("Starting LinkedIn profile scraping")
        scraped_data = scrape_linkedin_profile(data.linkedin_url)
        log.info("LinkedIn scraping completed successfully")

        log.info("Starting OpenAI processing")
        raw_investor_data = process_with_openai(scraped_data)
        log.info("OpenAI processing completed successfully")

        investor_data = clean_investor_data(raw_investor_data)

        log.info("Connecting to database")
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # First, check for the table's primary key column name
                log.info("Checking primary key column name")
                cursor.execute("""
                    SELECT c.column_name 
                    FROM information_schema.table_constraints tc 
                    JOIN information_schema.constraint_column_usage AS ccu USING (constraint_schema, constraint_name) 
                    JOIN information_schema.columns AS c ON c.table_schema = tc.constraint_schema 
                      AND tc.table_name = c.table_name AND ccu.column_name = c.column_name 
                    WHERE constraint_type = 'PRIMARY KEY' AND tc.table_name = 'investor_profiles';
                """)
                
                result = cursor.fetchone()
                primary_key_column = result['column_name'] if result else 'investor_id'  # Default fallback
                log.info(f"Primary key column name: {primary_key_column}")
                
                log.info("Preparing SQL insert")
                insert_query = f"""
                INSERT INTO investor_profiles (
                    investor_type, investment_experience, investment_focus_areas, primary_impact_areas,
                    geographical_preferences, startup_stages, business_models, target_roi, exit_horizon, 
                    check_size_min, check_size_max, preferred_ownership, revenue_milestones, 
                    monthly_recurring_revenue, tam, som, sam, traction_revenue_market, technology_scalability
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {primary_key_column};
                """

                log.info("Processing values for database insertion")
                values = (
                    investor_data.get["email"],
                    investor_data.get["investor_type"],
                    investor_data.get["investment_experience"],
                    investor_data.get["investment_focus_areas"],
                    investor_data.get["primary_impact_areas"],
                    investor_data.get["geographical_preferences"],
                    investor_data.get["startup_stages"],
                    investor_data.get["business_models"],
                    investor_data.get["target_roi"],
                    investor_data.get["exit_horizon"],
                    investor_data.get["check_size_min"],
                    investor_data.get["check_size_max"],
                    investor_data.get["preferred_ownership"],
                    investor_data.get["revenue_milestones"],
                    investor_data.get["monthly_recurring_revenue"],
                    investor_data.get["tam"],
                    investor_data.get["som"],
                    investor_data.get["sam"],
                    investor_data.get["traction_revenue_market"],
                    investor_data.get["technology_scalability"],
                )

                log.info("Executing database insert")
                cursor.execute(insert_query, values)
                result = cursor.fetchone()
                investor_id = result[primary_key_column]
                conn.commit()
                log.info(f"Database insert successful, got investor_id: {investor_id}")

        log.info("Scrape-investor endpoint completed successfully")
        return {"investor_id": investor_id, "investor_data": investor_data, "email": investor_data.get["email"]}

    except Exception as e:
        log.error(f"Error in scrape-investor endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# =====linkedin_profile_url====
@router.post("/linkedin_profile_url/")
async def update_linkedin_profile_url(
    profile_url: Optional[str] = Body(None, embed=True)
):
    """Store or update the LinkedIn profile URL (No Email Needed)."""

    if not profile_url or profile_url.strip() == "":
        raise HTTPException(status_code=400, detail="LinkedIn profile URL is required.")

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # ? Ensure table exists
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS Linkedin_login_url (
                    id SERIAL PRIMARY KEY,
                    linkedin_profile_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

                # ? Insert new LinkedIn profile URL and get the inserted ID
                cursor.execute(
                    "INSERT INTO Linkedin_login_url (linkedin_profile_url, created_at) VALUES (%s, %s) RETURNING id",
                    (profile_url, datetime.utcnow())
                )
                inserted_id = cursor.fetchone()["id"]

                conn.commit()
                return {
                    "message": "LinkedIn profile URL stored successfully.",
                    "id": inserted_id,
                    "profile_url": profile_url
                }

    except psycopg2.Error as e:
        logger.error(f"? Database Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Database Error: Unable to store LinkedIn profile URL.")

#GET Comparision_result

@router.get("/investor-profile/{id}")
def get_investor_profile(id: int):
    """Retrieve investor profile using id mapping from comparison_results"""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Ensure the given id exists in comparison_results
                cursor.execute("SELECT id FROM comparison_results WHERE id = %s", (id,))
                mapping = cursor.fetchone()

                if not mapping:
                    raise HTTPException(status_code=404, detail="No matching investor profile found for this id")

                # Fetch investor profile using the same id
                cursor.execute("SELECT * FROM investor_profiles WHERE id = %s", (id,))
                investor_profile = cursor.fetchone()

                if not investor_profile:
                    raise HTTPException(status_code=404, detail="Investor profile not found")

        return investor_profile

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
