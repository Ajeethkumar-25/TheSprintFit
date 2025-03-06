# Enhanced Investment Analysis System with Agent Architecture
# Implements the 5-agent model and new scoring mechanism

import logging
import openai
import json
import re
import os
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, Field
# LangGraph and LangChain imports
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
import operator


# Configure logging
logger = logging.getLogger(__name__)

# State Model for LangGraph
class InvestmentAnalysisState(BaseModel):
    # ID and basic info
    startup_name: str = "Unknown Startup"
    investor_data: Dict[str, Any] = {}
    s3_url: str = ""
    
    # Raw data
    extracted_text: str = ""
    structured_pitch_analysis: Dict[str, Any] = {}
    
    # Evaluation scores
    market_fit_score: Dict[str, Any] = {}  # 25%
    traction_score: Dict[str, Any] = {}  # 20%
    tech_maturity_score: Dict[str, Any] = {}  # 20%
    team_strength_score: Dict[str, Any] = {}  # 15%
    financial_viability_score: Dict[str, Any] = {}  # 20%

    completed_agents:  Dict[str, bool] = Field(default_factory=dict)
    
    # Missing data flags
    missing_data_flags: Dict[str, bool] = Field(default_factory=dict)

    # Final results
    thesis_fit_score: float = 0.0
    executive_summary: str = ""
    strengths_and_weaknesses: Dict[str, Any] = {}
    comparison_table: Dict[str, Any] = {}

# Initialize LangChain model
def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", openai_api_key=os.getenv("OPENAI_API_KEY"))

#############################
# Agent 1: Pitch Deck Processing Agent
#############################

def safe_update_dict(original_dict, updates):
    """Safely update a dictionary by creating a copy first."""
    if original_dict is None:
        return updates
    result = original_dict.copy()
    result.update(updates)
    return result

def should_run_thesis_matching(state):
    """
    Conditional routing function to determine if all component evaluations are complete.
    This enables the parallel workflow to converge back to a single path.
    """
    completed = [
        "market_fit" in state.completed_agents,
        "traction" in state.completed_agents,
        "tech_maturity" in state.completed_agents,
        "team_strength" in state.completed_agents, 
        "financial_viability" in state.completed_agents
    ]
    
    return all(completed)


def mark_completion(state, agent_name):
    """Add completion marker for an agent."""
    updated_completed = state.completed_agents.copy()
    updated_completed[agent_name] = True
    state.completed_agents = updated_completed
    return state


def process_pitch_deck(state):
    """
    Agent responsible for extracting and structuring data from the pitch deck.
    """
    logger.info("🚀 Pitch Deck Processing Agent started")

    try:
        # Start with a copy of the original state
        new_state = state.model_copy(deep=True)
        
        # Extract startup name
        extracted_name = extract_startup_name(state.extracted_text)
        
        # Update startup name if we found a valid one
        if extracted_name and extracted_name != "Unnamed Startup":
            new_state.startup_name = extracted_name
            
            # Also update it in the structured analysis
            if "Company Overview" not in new_state.structured_pitch_analysis:
                new_state.structured_pitch_analysis["Company Overview"] = {}
            
            new_state.structured_pitch_analysis["Company Overview"]["Name"] = extracted_name
            logger.info(f"✅ Extracted Startup Name: {extracted_name}")
        
        # Generate structured analysis from the extracted text
        structured_analysis = generate_structured_pitch_analysis(state.extracted_text)
        
        # Handle structured analysis errors
        if isinstance(structured_analysis, str) and "Error" in structured_analysis:
            logger.error(f"❌ Error in pitch analysis: {structured_analysis}")
            
            # Use a default structured analysis
            new_state.structured_pitch_analysis = {
                "Company Overview": {"Name": new_state.startup_name, "Description": "Information not available"},
                "Problem & Market Opportunity": {},
                "Solution & Product Offering": {},
                "Business Model & Monetization": {},
                "Traction & Validation": {},
                "Competitive Landscape": {},
                "Team & Expertise": {},
                "Funding & Growth Strategy": {},
                "Strengths & Investment Highlights": {},
                "Risks & Challenges": {},
                "Investment-Specific Fields": {}
            }
            
            # Update missing data flags
            if not new_state.missing_data_flags:
                new_state.missing_data_flags = {}
            new_state.missing_data_flags["structured_analysis"] = True
        else:
            # Parse structured analysis if needed
            if isinstance(structured_analysis, str):
                parsed_json = extract_json_from_text(structured_analysis)
                structured_analysis = parsed_json if parsed_json else {}
            
            # Update state with the parsed structured analysis
            new_state.structured_pitch_analysis = structured_analysis
            logger.info(f"✅ Structured analysis generated.")
        
        # Check for missing data and update flags
        missing_flags = check_missing_data(new_state.structured_pitch_analysis)
        
        # Initialize the missing_data_flags if it's None
        if not new_state.missing_data_flags:
            new_state.missing_data_flags = {}
            
        # Update with the missing flags
        for key, value in missing_flags.items():
            new_state.missing_data_flags[key] = value
        
        # Mark this agent as completed
        if not new_state.completed_agents:
            new_state.completed_agents = {}
        new_state.completed_agents["pitch_processing"] = True
        
        return new_state
    
    except Exception as e:
        logger.error(f"❌ Error in pitch deck processing: {str(e)}", exc_info=True)
        
        # Create a minimal updated state in case of error
        new_state = state.model_copy(deep=True)
        
        # Initialize missing_data_flags if needed
        if not new_state.missing_data_flags:
            new_state.missing_data_flags = {}
        
        # Set the error flag
        new_state.missing_data_flags["pitch_processing"] = True
        
        return new_state


def check_missing_data(structured_pitch_analysis):
    """
    Check for missing data in the structured analysis and set flags accordingly.
    """
    # Define key sections to check
    key_sections = [
        "Company Overview", 
        "Problem & Market Opportunity", 
        "Solution & Product Offering",
        "Business Model & Monetization", 
        "Traction & Validation", 
        "Team & Expertise",
        "Funding & Growth Strategy"
    ]

    missing_flags = {}
    
    if not structured_pitch_analysis:
        # Mark all sections as missing if no analysis was generated
        for section in key_sections:
            missing_flags[section] = True
        return missing_flags
    
    # Check each section for missing data
    for section in key_sections:
        section_data = structured_pitch_analysis.get(section, {})
        if not section_data or len(section_data) < 2:  # If section is empty or has very little data
            missing_flags[section] = True
            logger.warning(f"⚠️ Missing data in section: {section}")
    
    # Specific checks for important fields
    if "TAM" not in str(structured_pitch_analysis):
        missing_flags["market_size"] = True
    
    if "revenue" not in str(structured_pitch_analysis).lower() and "traction" not in str(structured_pitch_analysis).lower():
        missing_flags["traction_revenue"] = True
    
    if "team" not in str(structured_pitch_analysis).lower() and "founder" not in str(structured_pitch_analysis).lower():
        missing_flags["team_info"] = True
        
    return missing_flags

def merge_evaluation_results(state):
    """
    Junction node that receives results from all evaluation agents.
    This node doesn't modify the state, just verifies all evaluations are complete.
    """
    logger.info("🔄 Merging evaluation results before thesis matching")
    
    # Check if all evaluations are complete
    required_evaluations = [
        "market_fit", "traction", "tech_maturity", "team_strength", "financial_viability"
    ]
    
    completed = [eval_name in state.completed_agents for eval_name in required_evaluations]
    
    if not all(completed):
        missing = [required_evaluations[i] for i, is_complete in enumerate(completed) if not is_complete]
        logger.warning(f"⚠️ Some evaluations not complete: {missing}")
    else:
        logger.info("✅ All evaluations complete, proceeding to thesis matching")
    
    # Return the state unchanged - the individual evaluation agents have already updated it
    return state


def extract_startup_name(text):
    """Extract the startup name from the pitch deck text."""
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": 
                 "You are an expert at identifying company names in pitch decks. Extract only the startup/company name from the following pitch deck text. Return ONLY the name, nothing else. If you cannot confidently identify the name, return 'Unnamed Startup'."
                },
                {"role": "user", "content": text[:4000]}  # Using first 4000 chars where name likely appears
            ],
            temperature=0.3,
        )
        extracted_name = response.choices[0].message.content.strip()
        return extracted_name if extracted_name and extracted_name != "Unnamed Startup" else "Unnamed Startup"
    except Exception as e:
        logger.error(f"Error extracting startup name: {str(e)}")
        return "Unnamed Startup"

def generate_structured_pitch_analysis(text):
    """Generates a structured analysis of the pitch deck using OpenAI API."""
    try:
        logger.info(f"🔹 Running generate_structured_pitch_analysis() with input (First 500 chars): {text[:500]}")

        if not text or text.strip() == "":
            logger.error("❌ No valid text provided for pitch analysis.")
            return "Error generating analysis. No valid text found."

        client = openai.OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """You are an expert in analyzing startup pitch decks for investment evaluation. Extract and structure key insights from this pitch deck in JSON format with the following fields:

1. Company Overview: 
   - Name, Description, Core mission, Unique value proposition

2. Problem & Market Opportunity:
   - Problem definition
   - Market size (TAM, SAM, SOM)
   - Industry trends & regulatory landscape

3. Solution & Product Offering:
   - Key features and differentiators
   - Technology scalability

4. Business Model & Monetization:
   - Revenue streams
   - Startup stage
   - Traction & revenue milestones

5. Traction & Validation:
   - Key milestones achieved
   - User/customer metrics
   - Awards & recognition

6. Competitive Landscape:
   - Key competitors
   - Competitive advantage

7. Team & Expertise:
   - Founders & leadership background
   - Domain expertise and relevant experience

8. Funding & Growth Strategy:
   - Capital sought
   - Use of funds
   - Exit strategy & investor ROI expectations

9. Strengths & Investment Highlights:
   - Key strengths that make this a compelling investment

10. Risks & Challenges:
    - Major risks investors should consider
    - Weaknesses that could impact investment viability

11. Investment-Specific Fields:
    - Investor Type, Investment Focus Areas, Geographical Preferences, etc.
    - Financial metrics like MRR, ROI targets, etc."""},
                {"role": "user", "content": f"Here is the startup pitch deck text:\n\n{text}\n\nExtract and provide structured JSON output with the requested insights."}
            ],
            temperature=0.5,
        )

        structured_analysis = response.choices[0].message.content.strip()
        logger.info(f"✅ Pitch Analysis Generated Successfully. Sample Output: {structured_analysis[:500]}")
        return structured_analysis

    except Exception as e:
        logger.error(f"❌ Error in generate_structured_pitch_analysis: {str(e)}", exc_info=True)
        return f"Error generating analysis: {str(e)}"

def extract_json_from_text(text):
    """Extract JSON object from a text response and handle special characters."""
    try:
        # First try to extract JSON from code blocks
        json_match = re.search(r"```(?:json)?\n([\s\S]*?)\n```", text)
        
        if not json_match:
            # Fallback to looking for JSON objects with balanced braces
            pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})'
            json_match = re.search(pattern, text)
            
        if json_match:
            raw_json = json_match.group(1)
            
            # Clean common formatting issues - using safer regex patterns
            raw_json = raw_json.replace('"', '"').replace('"', '"')  # Replace smart quotes
            raw_json = raw_json.replace(''', "'").replace(''', "'")  # Replace smart apostrophes
            raw_json = raw_json.replace('isn"t', "isn't")
            # New: Fix apostrophes before replacing all single quotes
            # Fix "word"s" pattern that causes most issues
            raw_json = re.sub(r'(\w+)"s', r'\1\'s', raw_json)
            raw_json = re.sub(r'([^"])(\w+)"s([^"])', r'\1\2\'s\3', raw_json)
            
            # New: Fix possessive forms in key content
            raw_json = raw_json.replace('investor"s', 'investor\'s')
            raw_json = raw_json.replace('company"s', 'company\'s')
            raw_json = raw_json.replace('product"s', 'product\'s')
            raw_json = raw_json.replace('startup"s', 'startup\'s')
            
            # Continue with your existing replacements
            raw_json = raw_json.replace("'", '"')  # Replace single quotes with double quotes
            
            # Fix problematic characters one by one instead of complex regex
            # This avoids unterminated character set errors
            raw_json = re.sub(r'(\w+)\s*:', r'"\1":', raw_json)  # Quote keys
            raw_json = re.sub(r',\s*}', '}', raw_json)  # Fix trailing commas
            
            # New: Final apostrophe cleanup for special cases
            raw_json = raw_json.replace('"s ', "\'s ")
            raw_json = raw_json.replace('"s,', "\'s,")
            raw_json = raw_json.replace('"s"', "\'s\"")
            
            logger.debug(f"Cleaned JSON: {raw_json}")
            
            try:
                return json.loads(raw_json)
            except json.JSONDecodeError:
                # If still having issues, try a more lenient approach with a JSON library
                try:
                    import demjson3  # If available in your environment
                    return demjson3.decode(raw_json)
                except ImportError:
                    # If demjson3 is not available, try one more approach
                    # Replace any remaining problematic apostrophes
                    escaped_json = re.sub(r'(\w+)\'s', r'\1\\\'s', raw_json)
                    return json.loads(escaped_json)
            
        logger.error("❌ No valid JSON found in response text")
        return None
    except re.error as e:
        logger.error(f"❌ Regex Error: {str(e)}")
        
        # Fallback to a much simpler approach if regex is failing
        try:
            # Look for content between the outermost braces
            start = text.find('{')
            end = text.rfind('}')
            
            if start >= 0 and end > start:
                simple_json = text[start:end+1]
                # Basic replacements without regex
                simple_json = simple_json.replace('"', '"').replace('"', '"')
                simple_json = simple_json.replace(''', "'").replace(''', "'")
                
                # New: Fix apostrophes in the simple approach too
                simple_json = simple_json.replace('investor"s', 'investor\'s')
                simple_json = simple_json.replace('"s ', "\'s ")
                simple_json = simple_json.replace('"s,', "\'s,")
                
                simple_json = simple_json.replace("'", '"')
                
                return json.loads(simple_json)
        except:
            pass
        
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected extraction error: {str(e)}")
        return None

#############################
# Agent 2: Market Fit Evaluator (25%)
#############################

def evaluate_market_fit(state):
    """
    Evaluates the market fit of the startup (25% of overall score).
    Focuses on market size, growth potential, and addressable opportunity.
    """
    logger.info("🔥 Evaluating Market Fit (25% weight)")

    startup_name_local = state.startup_name
    
    
    llm = get_llm()
    
    # Get structured pitch analysis data
    pitch_analysis = state.structured_pitch_analysis
    
    # Extract relevant market data for the prompt
    market_data = {
        "market_size": pitch_analysis.get("Problem & Market Opportunity", {}).get("Market size (TAM, SAM, SOM)", "Not specified"),
        "industry_trends": pitch_analysis.get("Problem & Market Opportunity", {}).get("Industry trends & regulatory landscape", "Not specified"),
        "problem_statement": pitch_analysis.get("Problem & Market Opportunity", {}).get("Problem", "Not specified")
    }
    
    # Extract investor preferences related to market
    investor_prefs = {
        "geographical_preferences": state.investor_data.get("geographical_preferences", []),
        "investment_focus_areas": state.investor_data.get("investment_focus_areas", []),
        "target_roi": state.investor_data.get("target_roi", "Not specified"),
        "exit_horizon": state.investor_data.get("exit_horizon", "Not specified")
    }

    missing_data_flags_local = state.missing_data_flags.copy() if state.missing_data_flags else {}
    
    # Create prompt for market fit evaluation
    prompt = f"""
    You are a market research analyst assessing a startup's market fit for potential investment.
    Your task is to evaluate the Market Fit score (0-100) based on the following factors:

    1. **Market Size & Growth Potential**: Evaluate TAM, SAM, SOM and market growth rate
    2. **Market Alignment with Investor**: How well the startup's target market matches investor preferences
    3. **Problem-Solution Fit**: How well the startup's solution addresses a significant market need
    4. **Market Timing**: Is this the right time for this solution in this market

    ### **Startup Data**:
    Startup Name: {startup_name_local}
    Market Size Information: {market_data['market_size']}
    Industry Trends: {market_data['industry_trends']}
    Problem Statement: {market_data['problem_statement']}

    ### **Investor Preferences**:
    Geographical preferences: {investor_prefs['geographical_preferences']}
    Investment focus areas: {investor_prefs['investment_focus_areas']}
    Target ROI: {investor_prefs['target_roi']}
    Exit horizon: {investor_prefs['exit_horizon']} years

    ### **Missing Data Flags**:
    {missing_data_flags_local.get('market_size', False) and "⚠️ Warning: Market size information appears to be missing" or ""}
    {missing_data_flags_local.get('Problem & Market Opportunity', False) and "⚠️ Warning: Problem & Market Opportunity section is incomplete" or ""}

    ### **Scoring Guidelines**:
    - Apply a 10-point penalty for each significant piece of missing market information
    - Score 80-100 for large, growing markets with excellent alignment to investor thesis
    - Score 50-79 for moderate market size/growth or partial alignment to investor preferences
    - Score 0-49 for small/shrinking markets or poor alignment with investor focus

    ### **Output Format (JSON)**:
    {{
      "score": <number>,
      "analysis": "<3-4 sentences explaining your evaluation>",
      "strengths": ["<key market strength 1>", "<key market strength 2>"],
      "weaknesses": ["<key market weakness or risk 1>", "<key market weakness or risk 2>"],
      "alignment_with_investor": "<how well the market aligns with investor preferences>",
      "label": "<Strong/Moderate/Weak based on score>"
    }}
    """
    
    try:
        # Invoke the LLM for evaluation
        response = llm.invoke([HumanMessage(content=prompt)])
        
        response_text = response.content.strip() if response and response.content else ""
        if not response_text:
            logger.error("❌ Empty response from LLM for market fit evaluation")
            raise ValueError("Empty LLM response")
            
        logger.info(f"📌 Market Fit Raw Response:\n{response_text[:500]}...")
        
        # Parse the response
        market_fit_json = extract_json_from_text(response_text)
        
        if market_fit_json:
            # Apply missing data penalty if needed
            if any(flag for key, flag in state.missing_data_flags.items() 
                  if 'market' in key.lower() or 'problem' in key.lower()):
                original_score = float(market_fit_json.get("score", 50))
                penalized_score = max(0, original_score - 10)  # Apply 10-point penalty
                market_fit_json["score"] = penalized_score
                market_fit_json["analysis"] += " (Score adjusted due to missing market data.)"
                logger.info(f"⚠️ Applied missing data penalty to market fit score: {original_score} → {penalized_score}")
            
            # Ensure score is a number
            if isinstance(market_fit_json.get("score"), str):
                market_fit_json["score"] = float(market_fit_json["score"])
                
            # Store the result
            state_dict = state.dict()
            state_dict["market_fit_score"] = market_fit_json

            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["market_fit"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            # Create a new state object from the updated dictionary
            result_state = InvestmentAnalysisState(**state_dict)
        else:
            logger.error("❌ Failed to parse market fit evaluation")
            
            # ===== Create a NEW state object with default values ====
            state_dict = state.dict()
            
            state_dict["market_fit_score"] = {
                "score": 50,
                "analysis": "Could not evaluate market fit due to parsing error.",
                "strengths": ["Not determined"],
                "weaknesses": ["Not determined"],
                "alignment_with_investor": "Unknown",
                "label": "Moderate"
            }
            
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["market_fit"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            result_state = InvestmentAnalysisState(**state_dict)
    except Exception as e:
        logger.error(f"❌ Error in market fit evaluation: {str(e)}", exc_info=True)
        
        # ===== Create a NEW state object in case of error ====
        state_dict = state.dict()
        
        state_dict["market_fit_score"] = {
            "score": 50,
            "analysis": f"Error evaluating market fit: {str(e)}",
            "strengths": ["Not determined"],
            "weaknesses": ["Not determined"],
            "alignment_with_investor": "Unknown",
            "label": "Moderate"
        }
        
        completed_agents_local = state.completed_agents.copy()
        completed_agents_local["market_fit"] = True
        state_dict["completed_agents"] = completed_agents_local
        
        result_state = InvestmentAnalysisState(**state_dict)
    
    logger.info(f"✅ Market Fit Score: {result_state.market_fit_score.get('score', 'Unknown')}")
    return result_state

#############################
# Agent 3: Traction Evaluator (20%)
#############################

def evaluate_traction(state):
    """
    Evaluates the traction of the startup (20% of overall score).
    Focuses on user growth, revenue, partnerships, and validation.
    """
    logger.info("🔥 Evaluating Traction Score (20% weight)")

    startup_name_local = state.startup_name

    llm = get_llm()
    
    # Extract traction data from pitch analysis
    traction_data = state.structured_pitch_analysis.get("Traction & Validation", {})
    business_data = state.structured_pitch_analysis.get("Business Model & Monetization", {})
    
    # Extract investor preferences
    investor_prefs = {
        "monthly_recurring_revenue": state.investor_data.get("monthly_recurring_revenue", "Not specified"),
        "revenue_milestones": state.investor_data.get("revenue_milestones", "Not specified"),
        "traction_revenue_market": state.investor_data.get("traction_revenue_market", "Not specified"),
        "startup_stages": state.investor_data.get("startup_stages", [])
    }

    missing_data_flags_local = state.missing_data_flags.copy() if state.missing_data_flags else {}

    # Create prompt for traction evaluation
    prompt = f"""
    You are a startup metrics analyst evaluating the traction of {startup_name_local} for potential investment.
    Your task is to score their traction (0-100) based on:

    1. **User/Customer Growth**: Acquisition rate and customer retention
    2. **Revenue Traction**: MRR, revenue growth, and financial projections
    3. **Partnerships & Validation**: Strategic partnerships, pilots, or client testimonials
    4. **Stage-Appropriate Metrics**: Evaluate metrics appropriate to their growth stage
    5. **Alignment with Investor Expectations**: How the startup's traction matches investor requirements

    ### **Startup Traction Data**:
    ```
    {json.dumps(traction_data, indent=2)}
    ```

    ### **Business Model & Revenue Information**:
    ```
    {json.dumps(business_data, indent=2)}
    ```

    ### **Investor Expectations**:
    Minimum Monthly Recurring Revenue: {investor_prefs['monthly_recurring_revenue']}
    Revenue Milestones Expected: {investor_prefs['revenue_milestones']}
    Traction Metrics Valued: {investor_prefs['traction_revenue_market']}
    Preferred Startup Stages: {investor_prefs['startup_stages']}

    ### **Missing Data Flags**:
    {missing_data_flags_local.get('traction_revenue', False) and "⚠️ Warning: Traction and revenue data appears to be missing" or ""}
    {missing_data_flags_local.get('Traction & Validation', False) and "⚠️ Warning: Traction & Validation section is incomplete" or ""}

    ### **Scoring Guidelines**:
    - Apply a 10-point penalty for each significant piece of missing traction information
    - Score 80-100 for exceptional traction relative to stage (strong growth, revenue, validation)
    - Score 50-79 for average traction (some growth and validation, but not standout)
    - Score 0-49 for weak traction (minimal growth, revenue, or validation)

    ### **Output Format (JSON)**:
    {{
      "score": <number>,
      "analysis": "<3-4 sentences explaining the traction evaluation>",
      "key_metrics": {{
        "user_growth": "<assessment of user/customer acquisition>",
        "revenue_progress": "<assessment of revenue metrics>",
        "validation": "<assessment of external validation>"
      }},
      "strengths": ["<traction strength 1>", "<traction strength 2>"],
      "weaknesses": ["<traction weakness 1>", "<traction weakness 2>"],
      "stage_appropriate": "<yes/no/partial>",
      "investor_alignment": "<strong/moderate/weak>",
      "label": "<Strong/Moderate/Weak based on score>"
    }}
    """
    
    try:
        # Invoke the LLM for evaluation
        response = llm.invoke([HumanMessage(content=prompt)])
        
        response_text = response.content.strip() if response and response.content else ""
        if not response_text:
            logger.error("❌ Empty response from LLM for traction evaluation")
            raise ValueError("Empty LLM response")
            
        logger.info(f"📌 Traction Raw Response:\n{response_text[:500]}...")
        
        # Parse the response
        traction_json = extract_json_from_text(response_text)
        
        if traction_json:
            # Apply missing data penalty if needed
            if any(flag for key, flag in state.missing_data_flags.items() 
                   if 'traction' in key.lower() or 'revenue' in key.lower()):
                original_score = float(traction_json.get("score", 50))
                penalized_score = max(0, original_score - 10)  # Apply 10-point penalty
                traction_json["score"] = penalized_score
                traction_json["analysis"] += " (Score adjusted due to missing traction data.)"
                logger.info(f"⚠️ Applied missing data penalty to traction score: {original_score} → {penalized_score}")
            
            # Ensure score is a number
            if isinstance(traction_json.get("score"), str):
                traction_json["score"] = float(traction_json["score"])
                
            # Store the result
            state_dict = state.dict()
            
            # Update only our specific field in the dictionary
            state_dict["traction_score"] = traction_json
            
            # Update the completed_agents in the dictionary
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["traction"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            # Create a new state object from the updated dictionary
            result_state = InvestmentAnalysisState(**state_dict)
            
        else:
            logger.error("❌ Failed to parse traction evaluation")
            
            # ===== Create a NEW state object with default values ====
            state_dict = state.dict()
            
            state_dict["traction_score"] = {
                "score": 50,
                "analysis": "Could not evaluate traction due to parsing error.",
                "key_metrics": {
                    "user_growth": "Unknown",
                    "revenue_progress": "Unknown",
                    "validation": "Unknown"
                },
                "strengths": ["Not determined"],
                "weaknesses": ["Not determined"],
                "stage_appropriate": "Unknown",
                "investor_alignment": "Unknown",
                "label": "Moderate"
            }
            
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["traction"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            result_state = InvestmentAnalysisState(**state_dict)
    except Exception as e:
        logger.error(f"❌ Error in traction evaluation: {str(e)}", exc_info=True)
        
        # ===== Create a NEW state object in case of error ====
        state_dict = state.dict()
        
        state_dict["traction_score"] = {
            "score": 50,
            "analysis": f"Error evaluating traction: {str(e)}",
            "key_metrics": {
                "user_growth": "Unknown",
                "revenue_progress": "Unknown",
                "validation": "Unknown"
            },
            "strengths": ["Not determined"],
            "weaknesses": ["Not determined"],
            "stage_appropriate": "Unknown",
            "investor_alignment": "Unknown",
            "label": "Moderate"
        }
        
        completed_agents_local = state.completed_agents.copy()
        completed_agents_local["traction"] = True
        state_dict["completed_agents"] = completed_agents_local
        
        result_state = InvestmentAnalysisState(**state_dict)
    
    logger.info(f"✅ Traction Score: {result_state.traction_score.get('score', 'Unknown')}")
    return result_state

#############################
# Agent 4: Tech & Product Maturity Evaluator (20%)
#############################

def evaluate_tech_maturity(state):
    """
    Evaluates the technology and product maturity of the startup (20% of overall score).
    Focuses on innovation, technology scalability, product-market fit and development stage.
    """
    logger.info("🔥 Evaluating Tech & Product Maturity Score (20% weight)")

    startup_name_local = state.startup_name
    
    llm = get_llm()
    
    # Extract relevant tech and product data
    product_data = state.structured_pitch_analysis.get("Solution & Product Offering", {})
    market_data = state.structured_pitch_analysis.get("Problem & Market Opportunity", {})
    business_data = state.structured_pitch_analysis.get("Business Model & Monetization", {})
    
    # Extract investor preferences related to technology
    investor_prefs = {
        "technology_scalability": state.investor_data.get("technology_scalability", []),
        "business_models": state.investor_data.get("business_models", []),
        "investment_focus_areas": state.investor_data.get("investment_focus_areas", [])
    }
    
    # Create prompt for tech maturity evaluation

    missing_data_flags_local = state.missing_data_flags.copy() if state.missing_data_flags else {}

    prompt = f"""
    You are a technology and product analyst evaluating {startup_name_local} for potential investment.
    Your task is to score their technology & product maturity (0-100) based on:

    1. **Innovation Level**: How novel and differentiated is the technology or product
    2. **Technical Scalability**: Architecture, infrastructure, and ability to scale with growth
    3. **Product Development Stage**: MVP, beta, or market-ready product
    4. **Product-Market Fit**: Evidence that the product meets market needs
    5. **Technical Moat**: IP, patents, or technical barriers to entry
    6. **Tech Alignment with Investor**: How well the technology matches investor's tech preferences

    ### **Product & Technology Data**:
    ```
    {json.dumps(product_data, indent=2)}
    ```

    ### **Market Fit Context**:
    ```
    {json.dumps(market_data, indent=2)}
    ```

    ### **Business Model Context**:
    ```
    {json.dumps(business_data, indent=2)}
    ```

    ### **Investor Technology Preferences**:
    Technology Scalability Preferences: {investor_prefs['technology_scalability']}
    Business Models Preferences: {investor_prefs['business_models']}
    Investment Focus Areas: {investor_prefs['investment_focus_areas']}

    ### **Missing Data Flags**:
    {missing_data_flags_local.get('Solution & Product Offering', False) and "⚠️ Warning: Product and solution data appears to be missing" or ""}

    ### **Scoring Guidelines**:
    - Apply a 10-point penalty for each significant piece of missing product/tech information
    - Score 80-100 for highly innovative, scalable technology with strong product-market fit
    - Score 50-79 for solid technology with some innovation and reasonable scalability
    - Score 0-49 for weak technology, limited innovation, or poor scalability

    ### **Output Format (JSON)**:
    {{
      "score": <number>,
      "analysis": "<3-4 sentences explaining the technology evaluation>",
      "key_aspects": {{
        "innovation_level": "<assessment of how innovative the tech is>",
        "scalability": "<assessment of technical scalability>",
        "product_stage": "<current development stage>",
        "product_market_fit": "<assessment of product-market fit evidence>"
      }},
      "strengths": ["<tech/product strength 1>", "<tech/product strength 2>"],
      "weaknesses": ["<tech/product weakness 1>", "<tech/product weakness 2>"],
      "investor_alignment": "<strong/moderate/weak>",
      "label": "<Strong/Moderate/Weak based on score>"
    }}
    """
    
    try:
        # Invoke the LLM for evaluation
        response = llm.invoke([HumanMessage(content=prompt)])
        
        response_text = response.content.strip() if response and response.content else ""
        if not response_text:
            logger.error("❌ Empty response from LLM for tech maturity evaluation")
            raise ValueError("Empty LLM response")
            
        logger.info(f"📌 Tech Maturity Raw Response:\n{response_text[:500]}...")
        
        # Parse the response
        tech_json = extract_json_from_text(response_text)
        
        if tech_json:
            # Apply missing data penalty if needed
            if any(flag for key, flag in missing_data_flags_local.items() 
                   if 'product' in key.lower() or 'solution' in key.lower() or 'tech' in key.lower()):
                original_score = float(tech_json.get("score", 50))
                penalized_score = max(0, original_score - 10)  # Apply 10-point penalty
                tech_json["score"] = penalized_score
                tech_json["analysis"] += " (Score adjusted due to missing product/tech data.)"
                logger.info(f"⚠️ Applied missing data penalty to tech score: {original_score} → {penalized_score}")
            
            # Ensure score is a number
            if isinstance(tech_json.get("score"), str):
                tech_json["score"] = float(tech_json["score"])
                
            # Store the result
            state_dict = state.dict()
            
            # Update only our specific field in the dictionary
            state_dict["tech_maturity_score"] = tech_json
            
            # Update the completed_agents in the dictionary
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["tech_maturity"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            # Create a new state object from the updated dictionary
            result_state = InvestmentAnalysisState(**state_dict)
            
        else:
            logger.error("❌ Failed to parse tech maturity evaluation")
            
            # ===== Create a NEW state object with default values ====
            state_dict = state.dict()
            
            state_dict["tech_maturity_score"] = {
                "score": 50,
                "analysis": "Could not evaluate technology maturity due to parsing error.",
                "key_aspects": {
                    "innovation_level": "Unknown",
                    "scalability": "Unknown",
                    "product_stage": "Unknown",
                    "product_market_fit": "Unknown"
                },
                "strengths": ["Not determined"],
                "weaknesses": ["Not determined"],
                "investor_alignment": "Unknown",
                "label": "Moderate"
            }
            
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["tech_maturity"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            result_state = InvestmentAnalysisState(**state_dict)
    except Exception as e:
        logger.error(f"❌ Error in tech maturity evaluation: {str(e)}", exc_info=True)
        
        # ===== Create a NEW state object in case of error ====
        state_dict = state.dict()
        
        state_dict["tech_maturity_score"] = {
            "score": 50,
            "analysis": f"Error evaluating technology maturity: {str(e)}",
            "key_aspects": {
                "innovation_level": "Unknown",
                "scalability": "Unknown",
                "product_stage": "Unknown",
                "product_market_fit": "Unknown"
            },
            "strengths": ["Not determined"],
            "weaknesses": ["Not determined"],
            "investor_alignment": "Unknown",
            "label": "Moderate"
        }
        
        completed_agents_local = state.completed_agents.copy()
        completed_agents_local["tech_maturity"] = True
        state_dict["completed_agents"] = completed_agents_local
        
        result_state = InvestmentAnalysisState(**state_dict)
    
    logger.info(f"✅ Tech & Product Maturity Score: {result_state.tech_maturity_score.get('score', 'Unknown')}")
    return result_state

#############################
# Agent 5: Team Strength Evaluator (15%)
#############################

def evaluate_team_strength(state):
    """
    Evaluates the team strength of the startup (15% of overall score).
    Focuses on founders, leadership, domain expertise and track record.
    """
    logger.info("🔥 Evaluating Team Strength Score (15% weight)")
    
    startup_name_local =  state.startup_name

    llm = get_llm()
    
    # Extract team data
    team_data = state.structured_pitch_analysis.get("Team & Expertise", {})

    missing_data_flags_local = state.missing_data_flags.copy() if state.missing_data_flags else {}
    
    
    # Create prompt for team evaluation
    prompt = f"""
    You are an expert in evaluating startup founding teams for investment potential.
    Your task is to score the team of {startup_name_local} on a scale of 0-100 based on:

    1. **Founder/Leadership Quality**: Experience, track record, and leadership skills
    2. **Domain Expertise**: Relevant industry and technical knowledge
    3. **Team Completeness**: Whether key roles are filled with qualified individuals
    4. **Execution Track Record**: Previous accomplishments and evidence of execution ability
    5. **Team Dynamics**: Complementary skills and working relationships

    ### **Team Data**:
    ```
    {json.dumps(team_data, indent=2)}
    ```

    ### **Business Context**:
    Industry: {state.structured_pitch_analysis.get("Company Overview", {}).get("Description", "Not specified")}
    Problem Solving: {state.structured_pitch_analysis.get("Problem & Market Opportunity", {}).get("Problem", "Not specified")}

    ### **Missing Data Flags**:
    {missing_data_flags_local.get('team_info', False) and "⚠️ Warning: Team information appears to be missing or incomplete" or ""}
    {missing_data_flags_local.get('Team & Expertise', False) and "⚠️ Warning: Team & Expertise section is incomplete" or ""}

    ### **Scoring Guidelines**:
    - Apply a 15-point penalty for significantly missing team information
    - Score 80-100 for exceptional teams (serial entrepreneurs, domain experts, complementary skills)
    - Score 50-79 for solid teams with some experience but potential gaps
    - Score 0-49 for inexperienced teams or teams with major gaps

    ### **Output Format (JSON)**:
    {{
      "score": <number>,
      "analysis": "<3-4 sentences explaining the team evaluation>",
      "key_aspects": {{
        "founder_quality": "<assessment of founder/leadership caliber>",
        "domain_expertise": "<assessment of relevant industry knowledge>",
        "team_completeness": "<assessment of whether critical roles are filled>",
        "track_record": "<assessment of previous accomplishments>"
      }},
      "strengths": ["<team strength 1>", "<team strength 2>"],
      "weaknesses": ["<team weakness 1>", "<team weakness 2>"],
      "label": "<Strong/Moderate/Weak based on score>"
    }}
    """
    
    try:
        # Invoke the LLM for evaluation
        response = llm.invoke([HumanMessage(content=prompt)])
        
        response_text = response.content.strip() if response and response.content else ""
        if not response_text:
            logger.error("❌ Empty response from LLM for team evaluation")
            raise ValueError("Empty LLM response")
            
        logger.info(f"📌 Team Strength Raw Response:\n{response_text[:500]}...")
        
        # Parse the response
        team_json = extract_json_from_text(response_text)
        
        if team_json:
            # Apply missing data penalty if needed
            if any(flag for key, flag in missing_data_flags_local.items() 
                   if 'team' in key.lower() or 'expertise' in key.lower()):
                original_score = float(team_json.get("score", 50))
                penalized_score = max(0, original_score - 15)  # Apply 15-point penalty
                team_json["score"] = penalized_score
                team_json["analysis"] += " (Score significantly adjusted due to missing team data.)"
                logger.info(f"⚠️ Applied missing data penalty to team score: {original_score} → {penalized_score}")
            
            # Ensure score is a number
            if isinstance(team_json.get("score"), str):
                team_json["score"] = float(team_json["score"])
                
            # Store the result
            state_dict = state.dict()
            
            # Update only our specific field in the dictionary
            state_dict["team_strength_score"] = team_json
            
            # Update the completed_agents in the dictionary
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["team_strength"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            # Create a new state object from the updated dictionary
            result_state = InvestmentAnalysisState(**state_dict)
            
        else:
            logger.error("❌ Failed to parse team strength evaluation")
            
            # ===== Create a NEW state object with default values ====
            state_dict = state.dict()
            
            state_dict["team_strength_score"] = {
                "score": 50,
                "analysis": "Could not evaluate team strength due to parsing error.",
                "key_aspects": {
                    "founder_quality": "Unknown",
                    "domain_expertise": "Unknown",
                    "team_completeness": "Unknown",
                    "track_record": "Unknown"
                },
                "strengths": ["Not determined"],
                "weaknesses": ["Not determined"],
                "label": "Moderate"
            }
            
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["team_strength"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            result_state = InvestmentAnalysisState(**state_dict)
    except Exception as e:
        logger.error(f"❌ Error in team strength evaluation: {str(e)}", exc_info=True)
        
        # ===== Create a NEW state object in case of error ====
        state_dict = state.dict()
        
        state_dict["team_strength_score"] = {
            "score": 50,
            "analysis": f"Error evaluating team strength: {str(e)}",
            "key_aspects": {
                "founder_quality": "Unknown",
                "domain_expertise": "Unknown",
                "team_completeness": "Unknown",
                "track_record": "Unknown"
            },
            "strengths": ["Not determined"],
            "weaknesses": ["Not determined"],
            "label": "Moderate"
        }
        
        completed_agents_local = state.completed_agents.copy()
        completed_agents_local["team_strength"] = True
        state_dict["completed_agents"] = completed_agents_local
        
        result_state = InvestmentAnalysisState(**state_dict)
    
    logger.info(f"✅ Team Strength Score: {result_state.team_strength_score.get('score', 'Unknown')}")
    return result_state

#############################
# Agent 6: Financial Viability Evaluator (20%)
#############################

def evaluate_financial_viability(state):
    """
    Evaluates the financial viability of the startup (20% of overall score).
    Focuses on business model, unit economics, funding strategy and financial projections.
    """
    logger.info("🔥 Evaluating Financial Viability Score (20% weight)")

    startup_name_local = state.startup_name
    
    llm = get_llm()
    
    # Extract financial data
    business_data = state.structured_pitch_analysis.get("Business Model & Monetization", {})
    funding_data = state.structured_pitch_analysis.get("Funding & Growth Strategy", {})
    traction_data = state.structured_pitch_analysis.get("Traction & Validation", {})
    
    # Extract investor preferences related to financials
    investor_prefs = {
        "check_size_min": state.investor_data.get("check_size_min", "Not specified"),
        "check_size_max": state.investor_data.get("check_size_max", "Not specified"),
        "target_roi": state.investor_data.get("target_roi", "Not specified"),
        "exit_horizon": state.investor_data.get("exit_horizon", "Not specified"),
        "preferred_ownership": state.investor_data.get("preferred_ownership", "Not specified")
    }
    
    missing_data_flags_local = state.missing_data_flags.copy() if state.missing_data_flags else {}
    

    # Create prompt for financial viability evaluation
    prompt = f"""
    You are a financial analyst evaluating the financial viability of {startup_name_local} for potential investment.
    Your task is to score their financial viability (0-100) based on:

    1. **Business Model Soundness**: Clarity and viability of revenue generation
    2. **Unit Economics**: Margins, CAC, LTV, and other unit metrics
    3. **Capital Efficiency**: How efficiently the startup uses capital
    4. **Funding Strategy**: Appropriateness of current funding round and use of funds
    5. **Financial Projections**: Realism and achievability of financial forecasts
    6. **Alignment with Investor Financial Requirements**: How well the financials match investor expectations

    ### **Business Model & Monetization Data**:
    ```
    {json.dumps(business_data, indent=2)}
    ```

    ### **Funding Strategy Data**:
    ```
    {json.dumps(funding_data, indent=2)}
    ```

    ### **Traction & Financial Evidence**:
    ```
    {json.dumps(traction_data, indent=2)}
    ```

    ### **Investor Financial Requirements**:
    Check Size Min: ${investor_prefs['check_size_min']}
    Check Size Max: ${investor_prefs['check_size_max']}
    Target ROI: {investor_prefs['target_roi']}x
    Exit Horizon: {investor_prefs['exit_horizon']} years
    Preferred Ownership: {investor_prefs['preferred_ownership']}%

    ### **Missing Data Flags**:
    {missing_data_flags_local.get('Business Model & Monetization', False) and "⚠️ Warning: Business model data appears to be missing" or ""}
    {missing_data_flags_local.get('Funding & Growth Strategy', False) and "⚠️ Warning: Funding strategy data appears to be missing" or ""}

    ### **Scoring Guidelines**:
    - Apply a 10-point penalty for each significant piece of missing financial information
    - Score 80-100 for highly viable business models with strong unit economics and realistic projections
    - Score 50-79 for reasonable business models with some financial concerns or uncertainties
    - Score 0-49 for weak business models, poor unit economics, or unrealistic projections

    ### **Output Format (JSON)**:
    {{
      "score": <number>,
      "analysis": "<3-4 sentences explaining the financial viability evaluation>",
      "key_aspects": {{
        "business_model_viability": "<assessment of business model>",
        "unit_economics": "<assessment of margins and efficiency>",
        "funding_appropriateness": "<assessment of funding strategy>",
        "projection_realism": "<assessment of financial projections>"
      }},
      "strengths": ["<financial strength 1>", "<financial strength 2>"],
      "weaknesses": ["<financial weakness 1>", "<financial weakness 2>"],
      "investor_alignment": "<strong/moderate/weak>",
      "label": "<Strong/Moderate/Weak based on score>"
    }}
    """
    
    try:
        # Invoke the LLM for evaluation
        response = llm.invoke([HumanMessage(content=prompt)])
        
        response_text = response.content.strip() if response and response.content else ""
        if not response_text:
            logger.error("❌ Empty response from LLM for financial viability evaluation")
            raise ValueError("Empty LLM response")
            
        logger.info(f"📌 Financial Viability Raw Response:\n{response_text[:500]}...")
        
        # Parse the response
        financial_json = extract_json_from_text(response_text)
        
        if financial_json:
            # Apply missing data penalty if needed
            if any(flag for key, flag in missing_data_flags_local.items() 
                   if 'business' in key.lower() or 'fund' in key.lower() or 'financ' in key.lower()):
                original_score = float(financial_json.get("score", 50))
                penalized_score = max(0, original_score - 10)  # Apply 10-point penalty
                financial_json["score"] = penalized_score
                financial_json["analysis"] += " (Score adjusted due to missing financial data.)"
                logger.info(f"⚠️ Applied missing data penalty to financial score: {original_score} → {penalized_score}")
            
            # Ensure score is a number
            if isinstance(financial_json.get("score"), str):
                financial_json["score"] = float(financial_json["score"])
                
            # Store the result
            state_dict = state.dict()
            
            # Update only our specific field in the dictionary
            state_dict["financial_viability_score"] = financial_json
            
            # Update the completed_agents in the dictionary
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["financial_viability"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            # Create a new state object from the updated dictionary
            result_state = InvestmentAnalysisState(**state_dict)
            
        else:
            logger.error("❌ Failed to parse financial viability evaluation")
            
            # ===== Create a NEW state object with default values ====
            state_dict = state.dict()
            
            state_dict["financial_viability_score"] = {
                "score": 50,
                "analysis": "Could not evaluate financial viability due to parsing error.",
                "key_aspects": {
                    "business_model_viability": "Unknown",
                    "unit_economics": "Unknown",
                    "funding_appropriateness": "Unknown",
                    "projection_realism": "Unknown"
                },
                "strengths": ["Not determined"],
                "weaknesses": ["Not determined"],
                "investor_alignment": "Unknown",
                "label": "Moderate"
            }
            
            completed_agents_local = state.completed_agents.copy()
            completed_agents_local["financial_viability"] = True
            state_dict["completed_agents"] = completed_agents_local
            
            result_state = InvestmentAnalysisState(**state_dict)
    except Exception as e:
        logger.error(f"❌ Error in financial viability evaluation: {str(e)}", exc_info=True)
        
        # ===== Create a NEW state object in case of error ====
        state_dict = state.dict()
        
        state_dict["financial_viability_score"] = {
            "score": 50,
            "analysis": f"Error evaluating financial viability: {str(e)}",
            "key_aspects": {
                "business_model_viability": "Unknown",
                "unit_economics": "Unknown",
                "funding_appropriateness": "Unknown",
                "projection_realism": "Unknown"
            },
            "strengths": ["Not determined"],
            "weaknesses": ["Not determined"],
            "investor_alignment": "Unknown",
            "label": "Moderate"
        }
        
        completed_agents_local = state.completed_agents.copy()
        completed_agents_local["financial_viability"] = True
        state_dict["completed_agents"] = completed_agents_local
        
        result_state = InvestmentAnalysisState(**state_dict)
    
    logger.info(f"✅ Financial Viability Score: {result_state.financial_viability_score.get('score', 'Unknown')}")
    return result_state

#############################
# Agent 7: Thesis Matching Agent
#############################

def calculate_thesis_fit(state):
    """
    Calculates the overall thesis fit score based on the weighted average of individual scores.
    Generates an executive summary and detailed analysis.
    """
    logger.info("🚀 Calculating Final Thesis Fit Score")
    
    try:
        # Extract scores, defaulting to 50 if missing
        market_score = float(state.market_fit_score.get("score", 50))
        traction_score = float(state.traction_score.get("score", 50))
        tech_score = float(state.tech_maturity_score.get("score", 50))
        team_score = float(state.team_strength_score.get("score", 50))
        financial_score = float(state.financial_viability_score.get("score", 50))
        
        # Apply weights as specified in the requirements
        weighted_score = (
            market_score * 0.25 +      # Market Fit (25%)
            traction_score * 0.20 +    # Traction (20%)
            tech_score * 0.20 +        # Tech & Product Maturity (20%)
            team_score * 0.15 +        # Team Strength (15%)
            financial_score * 0.20     # Financial Viability (20%)
        )
        
        # Round to nearest integer
        state.thesis_fit_score = round(weighted_score)
        
        logger.info(f"✅ Calculated Thesis Fit Score: {state.thesis_fit_score}")
        logger.info(f"📊 Component Scores: Market: {market_score}, Traction: {traction_score}, Tech: {tech_score}, Team: {team_score}, Financial: {financial_score}")
        
        # Determine label based on score
        if state.thesis_fit_score >= 80:
            fit_label = "High Fit"
        elif state.thesis_fit_score >= 60:
            fit_label = "Moderate Fit"
        elif state.thesis_fit_score >= 40:
            fit_label = "Low Fit"
        else:
            fit_label = "Poor Fit"
            
        # Compile strengths and weaknesses
        strengths = []
        weaknesses = []
        
        # Add top strengths from each category
        if state.market_fit_score.get("strengths"):
            strengths.extend(state.market_fit_score.get("strengths", [])[:1])
        if state.traction_score.get("strengths"):
            strengths.extend(state.traction_score.get("strengths", [])[:1])
        if state.tech_maturity_score.get("strengths"):
            strengths.extend(state.tech_maturity_score.get("strengths", [])[:1])
        if state.team_strength_score.get("strengths"):
            strengths.extend(state.team_strength_score.get("strengths", [])[:1])
        if state.financial_viability_score.get("strengths"):
            strengths.extend(state.financial_viability_score.get("strengths", [])[:1])
        
        # Add top weaknesses from each category
        if state.market_fit_score.get("weaknesses"):
            weaknesses.extend(state.market_fit_score.get("weaknesses", [])[:1])
        if state.traction_score.get("weaknesses"):
            weaknesses.extend(state.traction_score.get("weaknesses", [])[:1])
        if state.tech_maturity_score.get("weaknesses"):
            weaknesses.extend(state.tech_maturity_score.get("weaknesses", [])[:1])
        if state.team_strength_score.get("weaknesses"):
            weaknesses.extend(state.team_strength_score.get("weaknesses", [])[:1])
        if state.financial_viability_score.get("weaknesses"):
            weaknesses.extend(state.financial_viability_score.get("weaknesses", [])[:1])
        
        state.strengths_and_weaknesses = {
            "strengths": strengths,
            "weaknesses": weaknesses
        }
        
        # Generate comparison table with investor criteria
        state.comparison_table = {
            "market_fit": {
                "score": market_score,
                "label": state.market_fit_score.get("label", "Moderate"),
                "investor_alignment": state.market_fit_score.get("alignment_with_investor", "Moderate")
            },
            "traction": {
                "score": traction_score,
                "label": state.traction_score.get("label", "Moderate"),
                "investor_alignment": state.traction_score.get("investor_alignment", "Moderate")
            },
            "tech_maturity": {
                "score": tech_score,
                "label": state.tech_maturity_score.get("label", "Moderate"),
                "investor_alignment": state.tech_maturity_score.get("investor_alignment", "Moderate")
            },
            "team_strength": {
                "score": team_score,
                "label": state.team_strength_score.get("label", "Moderate"),
            },
            "financial_viability": {
                "score": financial_score,
                "label": state.financial_viability_score.get("label", "Moderate"),
                "investor_alignment": state.financial_viability_score.get("investor_alignment", "Moderate")
            },
            "thesis_fit": {
                "score": state.thesis_fit_score,
                "label": fit_label
            }
        }
        
        # Generate executive summary with LLM
        llm = get_llm()
        
        summary_prompt = f"""
        Generate a concise executive summary for an investment analysis report on {state.startup_name}.
        Include the following information in a professional, investment-focused tone:

        1. Startup Name: {state.startup_name}
        2. Overall Thesis Fit Score: {state.thesis_fit_score}/100 ({fit_label})
        3. Key Component Scores:
           - Market Fit: {market_score}/100 ({state.market_fit_score.get("label", "Moderate")})
           - Traction: {traction_score}/100 ({state.traction_score.get("label", "Moderate")})
           - Tech & Product Maturity: {tech_score}/100 ({state.tech_maturity_score.get("label", "Moderate")})
           - Team Strength: {team_score}/100 ({state.team_strength_score.get("label", "Moderate")})
           - Financial Viability: {financial_score}/100 ({state.financial_viability_score.get("label", "Moderate")})
        
        4. Key Strengths:
        {json.dumps(strengths, indent=2)}
        
        5. Key Weaknesses:
        {json.dumps(weaknesses, indent=2)}
        
        6. Investor Profile Type: {state.investor_data.get("investor_type", "Not specified")}
        
        Keep the summary under 200 words and focus on actionable insights.
        """
        
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        
        response_text = response.content.strip() if response and response.content else ""
        if response_text:
            state.executive_summary = response_text
        else:
            # Create a basic summary if LLM fails
            state.executive_summary = f"""
            Investment Analysis Summary: {state.startup_name}
            
            Overall Thesis Fit Score: {state.thesis_fit_score}/100 ({fit_label})
            
            Key Component Scores:
            - Market Fit: {market_score}/100 ({state.market_fit_score.get("label", "Moderate")})
            - Traction: {traction_score}/100 ({state.traction_score.get("label", "Moderate")})
            - Tech & Product Maturity: {tech_score}/100 ({state.tech_maturity_score.get("label", "Moderate")})
            - Team Strength: {team_score}/100 ({state.team_strength_score.get("label", "Moderate")})
            - Financial Viability: {financial_score}/100 ({state.financial_viability_score.get("label", "Moderate")})
            
            This startup has been evaluated against the investor's thesis criteria.
            """
        
        logger.info("✅ Executive summary generated")
        
    except Exception as e:
        logger.error(f"❌ Error calculating thesis fit: {str(e)}", exc_info=True)
        # Set fallback values
        state.thesis_fit_score = 50
        state.executive_summary = f"Error calculating thesis fit score: {str(e)}"
        state.strengths_and_weaknesses = {
            "strengths": ["Analysis error - results may be incomplete"],
            "weaknesses": ["Analysis error - results may be incomplete"]
        }
        state.comparison_table = {
            "thesis_fit": {"score": 50, "label": "Moderate Fit"}
        }
    
    return state

#############################
# Main LangGraph Flow
#############################
def mark_completion(state, agent_name):
    """Add completion marker for an agent."""
    state.completed_agents = safe_update_dict(
        state.completed_agents,
        {agent_name: True}
    )
    return state

def run_investment_analysis(s3_url, investor_data, extracted_text, startup_name=None):
    """
    Main function to run the complete investment analysis workflow.
    
    Args:
        s3_url (str): URL to the pitch deck on S3
        investor_data (dict): Investor profile and thesis data
        extracted_text (str): Text extracted from the pitch deck
        startup_name (str, optional): Name of the startup if already known
    
    Returns:
        dict: Complete analysis results
    """
    logger.info("🚀 Starting Enhanced Investment Analysis Workflow")

    # startup_name_dict = {"initial": str(startup_name) if startup_name else "Unknown"}
    
    # Initialize the state
    initial_state = InvestmentAnalysisState(
        s3_url=s3_url,
        investor_data=investor_data,
        extracted_text=extracted_text,
        startup_name=startup_name if startup_name else "Unknown Startup"
    )
    
    # Create the workflow graph
    workflow = StateGraph(InvestmentAnalysisState)
    
    # Add all the agent nodes
    workflow.add_node("pitch_deck_processing", process_pitch_deck)
    workflow.add_node("market_fit_evaluation", evaluate_market_fit)
    workflow.add_node("traction_evaluation", evaluate_traction)
    workflow.add_node("tech_maturity_evaluation", evaluate_tech_maturity)
    workflow.add_node("team_strength_evaluation", evaluate_team_strength)
    workflow.add_node("financial_viability_evaluation", evaluate_financial_viability)
    workflow.add_node("merge_evaluations", merge_evaluation_results)
    workflow.add_node("thesis_matching", calculate_thesis_fit)
    
    # Define the workflow execution order
    workflow.set_entry_point("pitch_deck_processing")

    workflow.add_edge("pitch_deck_processing", "market_fit_evaluation")
    workflow.add_edge("market_fit_evaluation", "traction_evaluation")
    workflow.add_edge("traction_evaluation", "tech_maturity_evaluation")
    workflow.add_edge("tech_maturity_evaluation", "team_strength_evaluation")
    workflow.add_edge("team_strength_evaluation", "financial_viability_evaluation")
    workflow.add_edge("financial_viability_evaluation", "thesis_matching")
    workflow.add_edge("thesis_matching", END)
    
    # After pitch processing, evaluate all criteria in parallel
    # workflow.add_edge("pitch_deck_processing", "market_fit_evaluation")
    # workflow.add_edge("pitch_deck_processing", "traction_evaluation")
    # workflow.add_edge("pitch_deck_processing", "tech_maturity_evaluation")
    # workflow.add_edge("pitch_deck_processing", "team_strength_evaluation")
    # workflow.add_edge("pitch_deck_processing", "financial_viability_evaluation")

    # workflow.add_edge("market_fit_evaluation", "merge_evaluations")
    # workflow.add_edge("traction_evaluation", "merge_evaluations")
    # workflow.add_edge("tech_maturity_evaluation", "merge_evaluations")
    # workflow.add_edge("team_strength_evaluation", "merge_evaluations")
    # workflow.add_edge("financial_viability_evaluation", "merge_evaluations")
    # workflow.add_edge("merge_evaluations", "thesis_matching")

    # workflow.add_conditional_edges(
    #     "market_fit_evaluation",
    #     should_run_thesis_matching,
    #     {
    #         True: "thesis_matching",
    #         False: END  # If not ready, end this branch but other branches continue
    #     }
    # )
    
    # workflow.add_conditional_edges(
    #     "traction_evaluation",
    #     should_run_thesis_matching,
    #     {
    #         True: "thesis_matching",
    #         False: END
    #     }
    # )
    
    # workflow.add_conditional_edges(
    #     "tech_maturity_evaluation",
    #     should_run_thesis_matching,
    #     {
    #         True: "thesis_matching",
    #         False: END
    #     }
    # )
    
    # workflow.add_conditional_edges(
    #     "team_strength_evaluation",
    #     should_run_thesis_matching,
    #     {
    #         True: "thesis_matching",
    #         False: END
    #     }
    # )
    
    # workflow.add_conditional_edges(
    #     "financial_viability_evaluation",
    #     should_run_thesis_matching,
    #     {
    #         True: "thesis_matching",
    #         False: END
    #     }
    # )    
    
    # Set the end of the workflow
    
    # Compile and run the workflow
    try:
        app = workflow.compile()
        result = app.invoke(initial_state)
        
        # Format the result for API response
        formatted_result = format_analysis_result(result)
        logger.info("✅ Investment analysis workflow completed successfully")
        return formatted_result
        
    except Exception as e:
        logger.error(f"❌ LangGraph workflow failed: {str(e)}", exc_info=True)
        return {
            "error": f"Analysis failed: {str(e)}",
            "startup_name": initial_state.startup_name or "Unknown Startup",
            "thesis_fit_score": 0,
            "executive_summary": f"Error during analysis: {str(e)}",
            "detailed_analysis": {}
        }

def format_analysis_result(result):
    """Format the analysis result for API response."""
    try:
        # If result is already a dict, use it directly
        if isinstance(result, dict):
            state_dict = result
        else:
            # Otherwise convert to dict
            state_dict = result.dict()
        
        # Format the executive summary
        executive_summary = {
            "Startup Name": state_dict.get("startup_name", "Unknown"),
            "Investor Profile Type": state_dict.get("investor_data", {}).get("investor_type", "Unknown"),
            "Thesis Fit Score": state_dict.get("thesis_fit_score", 0),
            "Market Fit": f"{state_dict.get('market_fit_score', {}).get('score', 0)}/100 - {state_dict.get('market_fit_score', {}).get('analysis', 'No analysis available')}",
            "Traction": f"{state_dict.get('traction_score', {}).get('score', 0)}/100 - {state_dict.get('traction_score', {}).get('analysis', 'No analysis available')}",
            "Tech & Product Maturity": f"{state_dict.get('tech_maturity_score', {}).get('score', 0)}/100 - {state_dict.get('tech_maturity_score', {}).get('analysis', 'No analysis available')}",
            "Team Strength": f"{state_dict.get('team_strength_score', {}).get('score', 0)}/100 - {state_dict.get('team_strength_score', {}).get('analysis', 'No analysis available')}",
            "Financial Viability": f"{state_dict.get('financial_viability_score', {}).get('score', 0)}/100 - {state_dict.get('financial_viability_score', {}).get('analysis', 'No analysis available')}",
            "Executive Summary": state_dict.get("executive_summary", "No summary available")
        }
        
        # Format the detailed analysis
        detailed_analysis = {
            "Market Fit": state_dict.get("market_fit_score", {}),
            "Traction": state_dict.get("traction_score", {}),
            "Tech & Product Maturity": state_dict.get("tech_maturity_score", {}),
            "Team Strength": state_dict.get("team_strength_score", {}),
            "Financial Viability": state_dict.get("financial_viability_score", {}),
            "Strengths and Weaknesses": state_dict.get("strengths_and_weaknesses", {}),
            "Comparison with Investor Criteria": state_dict.get("comparison_table", {})
        }
        
        return {
            "id": str(id(result)),  # Using object ID as a placeholder - replace with DB ID in production
            "startup_name": state_dict.get("startup_name", "Unknown"),
            "investor_data": state_dict.get("investor_data", {}),
            "thesis_fit_score": state_dict.get("thesis_fit_score", 0),
            "executive_summary": executive_summary,
            "detailed_analysis": detailed_analysis,
            "missing_data_flags": state_dict.get("missing_data_flags", {})
        }
    except Exception as e:
        logger.error(f"❌ Error formatting analysis result: {str(e)}", exc_info=True)
        return {
            "error": f"Error formatting analysis result: {str(e)}",
            "startup_name": "Unknown",
            "thesis_fit_score": 0
        }