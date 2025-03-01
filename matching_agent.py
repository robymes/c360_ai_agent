"""
CrewAI agent definitions for customer matching.
"""

import logging
import json
from typing import Dict, List, Any
from crewai import Agent, Task, Crew
from langchain_openai import AzureChatOpenAI

from config import AGENT_MODEL, AZURE_OPENAI_API_VERSION

# Set up logger
logger = logging.getLogger(__name__)

class CustomerMatchingAgent:
    """
    CrewAI agent for determining if customer records match.
    """
    
    def __init__(self):
        """Initialize the agent."""
        self._create_agents()
    
    def _create_agents(self):
        """Create the CrewAI agents."""
        # Expert agent for detailed analysis
        self.customer_matching_expert = Agent(
            role="Customer Data Matching Expert",
            goal="Determine with high accuracy if two customer records represent the same real person",
            backstory="""You are a data scientist with 20 years of experience in entity resolution and record 
            matching. You excel at analyzing customer data to determine if different records 
            represent the same real person despite variations, typos, and missing fields.
            You understand common patterns in customer data entry errors and how personal 
            information can be represented differently across systems.""",
            verbose=True,
            allow_delegation=False,
            llm=AzureChatOpenAI(
                azure_deployment=AGENT_MODEL,
                api_version=AZURE_OPENAI_API_VERSION
            )
        )
    
    def _analyze_name_similarity(self, name1: str, name2: str) -> Dict[str, Any]:
        """
        Analyze the similarity between two names.
        
        Args:
            name1: First name
            name2: Second name
            
        Returns:
            Dictionary with analysis results
        """
        if not name1 or not name2:
            return {
                "exact_match": False,
                "possible_typo": False,
                "similarity": 0.0,
                "notes": "One or both names are missing"
            }
        
        name1 = str(name1).lower().strip()
        name2 = str(name2).lower().strip()
        
        # Check for exact match
        if name1 == name2:
            return {
                "exact_match": True,
                "possible_typo": False,
                "similarity": 1.0,
                "notes": "Exact match"
            }
        
        # Calculate basic similarity score
        # Note: This is a simplified implementation
        # Levenshtein distance would be better but we'll leave that to the LLM
        if len(name1) > 0 and len(name2) > 0:
            # Check for prefix match (e.g., "Rob" vs "Robert")
            min_len = min(len(name1), len(name2))
            if name1[:min_len] == name2[:min_len] and min_len >= 3:
                return {
                    "exact_match": False,
                    "possible_typo": False,
                    "similarity": 0.7,
                    "notes": f"Possible nickname or abbreviated form ({min_len} character prefix match)"
                }
                
            # Check for possible typo (simple check - not comprehensive)
            common_chars = set(name1).intersection(set(name2))
            similarity = len(common_chars) / max(len(set(name1)), len(set(name2)))
            
            if similarity > 0.7:
                return {
                    "exact_match": False,
                    "possible_typo": True,
                    "similarity": similarity,
                    "notes": "Possible typo or variant spelling"
                }
        
        return {
            "exact_match": False,
            "possible_typo": False,
            "similarity": 0.0,
            "notes": "Different names"
        }
    
    def _prepare_customer_comparison(
        self, 
        source_customer: Dict[str, Any], 
        candidate_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Prepare a detailed comparison of customers for agent analysis.
        
        Args:
            source_customer: Source customer record to compare
            candidate_matches: List of potential matches with their data
            
        Returns:
            Dictionary with structured comparison data
        """
        comparison = {
            "source_customer": self._format_customer_for_agent(source_customer),
            "candidate_matches": [],
            "field_analyses": []
        }
        
        for candidate in candidate_matches:
            candidate_data = candidate.get("customer_data", {})
            if not candidate_data:
                continue
                
            # Format candidate for display
            formatted_candidate = self._format_customer_for_agent(candidate_data)
            formatted_candidate["match_type"] = candidate.get("match_type", "unknown")
            formatted_candidate["similarity_score"] = candidate.get("similarity", 0.0)
            
            # Add to candidate matches
            comparison["candidate_matches"].append(formatted_candidate)
            
            # Detailed field analysis
            field_analysis = {
                "customer_id": candidate_data.get("customer_id", "unknown"),
                "fields": {}
            }
            
            # Analyze name
            source_name = source_customer.get("name", "")
            candidate_name = candidate_data.get("name", "")
            field_analysis["fields"]["name"] = self._analyze_name_similarity(source_name, candidate_name)
            
            # Analyze surname
            source_surname = source_customer.get("surname", "")
            candidate_surname = candidate_data.get("surname", "")
            field_analysis["fields"]["surname"] = self._analyze_name_similarity(source_surname, candidate_surname)
            
            # Analyze email
            source_email = source_customer.get("email", "")
            candidate_email = candidate_data.get("email", "")
            if source_email and candidate_email:
                source_email = str(source_email).lower().strip()
                candidate_email = str(candidate_email).lower().strip()
                field_analysis["fields"]["email"] = {
                    "exact_match": source_email == candidate_email,
                    "similarity": 1.0 if source_email == candidate_email else 0.0,
                    "notes": "Exact match" if source_email == candidate_email else "Different emails"
                }
            else:
                field_analysis["fields"]["email"] = {
                    "exact_match": False,
                    "similarity": 0.0,
                    "notes": "One or both emails are missing"
                }
            
            # Analyze phone
            source_phone = source_customer.get("mobile_phone_number", "")
            candidate_phone = candidate_data.get("mobile_phone_number", "")
            if source_phone and candidate_phone:
                # Normalize phone numbers (remove non-digits)
                source_phone_norm = ''.join(filter(str.isdigit, str(source_phone)))
                candidate_phone_norm = ''.join(filter(str.isdigit, str(candidate_phone)))
                
                field_analysis["fields"]["mobile_phone_number"] = {
                    "exact_match": source_phone_norm == candidate_phone_norm,
                    "similarity": 1.0 if source_phone_norm == candidate_phone_norm else 0.0,
                    "notes": "Exact match" if source_phone_norm == candidate_phone_norm else "Different phone numbers"
                }
            else:
                field_analysis["fields"]["mobile_phone_number"] = {
                    "exact_match": False,
                    "similarity": 0.0,
                    "notes": "One or both phone numbers are missing"
                }
            
            # Analyze country
            source_country = source_customer.get("country", "")
            candidate_country = candidate_data.get("country", "")
            if source_country and candidate_country:
                source_country = str(source_country).lower().strip()
                candidate_country = str(candidate_country).lower().strip()
                field_analysis["fields"]["country"] = {
                    "exact_match": source_country == candidate_country,
                    "similarity": 1.0 if source_country == candidate_country else 0.0,
                    "notes": "Exact match" if source_country == candidate_country else "Different countries"
                }
            else:
                field_analysis["fields"]["country"] = {
                    "exact_match": False,
                    "similarity": 0.0,
                    "notes": "One or both countries are missing"
                }
            
            # Analyze date of birth
            source_dob = source_customer.get("date_of_birth", "")
            candidate_dob = candidate_data.get("date_of_birth", "")
            if source_dob and candidate_dob:
                source_dob = str(source_dob).strip()
                candidate_dob = str(candidate_dob).strip()
                field_analysis["fields"]["date_of_birth"] = {
                    "exact_match": source_dob == candidate_dob,
                    "similarity": 1.0 if source_dob == candidate_dob else 0.0,
                    "notes": "Exact match" if source_dob == candidate_dob else "Different dates of birth"
                }
            else:
                field_analysis["fields"]["date_of_birth"] = {
                    "exact_match": False,
                    "similarity": 0.0,
                    "notes": "One or both dates of birth are missing"
                }
            
            # Add the field analysis to the comparison
            comparison["field_analyses"].append(field_analysis)
        
        return comparison
    
    def _format_customer_for_agent(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format customer data for agent presentation.
        
        Args:
            customer: Customer data dictionary
            
        Returns:
            Formatted customer data
        """
        # Create a clean copy with only relevant fields
        formatted = {
            "customer_id": customer.get("customer_id", "unknown"),
            "name": customer.get("name", ""),
            "surname": customer.get("surname", ""),
            "email": customer.get("email", ""),
            "mobile_phone_number": customer.get("mobile_phone_number", ""),
            "country": customer.get("country", ""),
            "date_of_birth": customer.get("date_of_birth", ""),
            "source_id": customer.get("source_id", "")
        }
        
        # Convert all values to strings and handle None values
        for key, value in formatted.items():
            if value is None:
                formatted[key] = ""
            else:
                formatted[key] = str(value)
        
        return formatted
    
    def analyze_customer_matches(
        self, 
        source_customer: Dict[str, Any], 
        candidate_matches: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze if the source customer matches any of the candidates.
        
        Args:
            source_customer: Source customer record to compare
            candidate_matches: List of potential matches with their data
            
        Returns:
            Dictionary with matching results
        """
        if not candidate_matches:
            logger.info(f"No candidate matches for customer {source_customer.get('customer_id', 'unknown')}")
            return {
                "is_match": False,
                "matched_customer_id": None,
                "matched_family_id": None,
                "confidence": 0.0,
                "reasoning": "No candidate matches found"
            }
        
        try:
            # Prepare detailed comparison data
            comparison_data = self._prepare_customer_comparison(source_customer, candidate_matches)
            
            # Create the task for the expert agent
            task = Task(
                description=f"""Analyze if the source customer record matches any of the candidate records.
                
                Here is the source customer data:
                {json.dumps(comparison_data['source_customer'], indent=2)}
                
                Here are the candidate matches and their similarity scores:
                {json.dumps(comparison_data['candidate_matches'], indent=2)}
                
                Here are detailed field analyses for each candidate:
                {json.dumps(comparison_data['field_analyses'], indent=2)}
                
                Analyze each field carefully and determine if any of the candidates represents the same real person as the source customer.
                Consider the following:
                1. Exact matches on email or phone number strongly suggest the same person
                2. Similar names with matching dates of birth suggest the same person
                3. Consider possible typos, nickname variations, and common data entry patterns
                4. Missing fields should not count against a match if other key fields match
                5. Multiple partial matches across different fields may indicate a match even without exact matches
                
                For your response, provide a JSON object with the following fields:
                - is_match: Boolean indicating if any candidate is a match
                - matched_customer_id: The customer_id of the matching record, or null if no match
                - matched_family_id: If available, the family_id of the matching record, or null if no match
                - confidence: A number from 0.0 to 1.0 indicating your confidence in the match
                - reasoning: A detailed explanation of your decision
                
                Return ONLY valid JSON without markdown formatting or extra text.""",
                agent=self.customer_matching_expert,
                expected_output="""A JSON object with:
                {
                    "is_match": true or false,
                    "matched_customer_id": "ID or null",
                    "matched_family_id": "ID or null", 
                    "confidence": 0.0-1.0,
                    "reasoning": "Explanation"
                }"""
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[self.customer_matching_expert],
                tasks=[task],
                verbose=True
            )
            
            result = crew.kickoff()
            
            # Parse the response
            try:
                # Try to parse the raw JSON response
                response_json = json.loads(result.raw())
                logger.info(f"Successfully parsed match analysis for customer {source_customer.get('customer_id', 'unknown')}")
                return response_json
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract JSON from the text
                raw_text = result.raw()
                json_start = raw_text.find('{')
                json_end = raw_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = raw_text[json_start:json_end]
                    try:
                        response_json = json.loads(json_str)
                        logger.warning(f"Extracted JSON from response text for customer {source_customer.get('customer_id', 'unknown')}")
                        return response_json
                    except json.JSONDecodeError:
                        logger.error(f"Failed to extract valid JSON from response for customer {source_customer.get('customer_id', 'unknown')}")
                
                # Return a default response if JSON parsing fails
                logger.error(f"Could not parse agent response as JSON for customer {source_customer.get('customer_id', 'unknown')}")
                return {
                    "is_match": False,
                    "matched_customer_id": None,
                    "matched_family_id": None,
                    "confidence": 0.0,
                    "reasoning": "Failed to parse agent response as JSON"
                }
        
        except Exception as e:
            logger.error(f"Error in analyze_customer_matches: {e}")
            return {
                "is_match": False,
                "matched_customer_id": None,
                "matched_family_id": None,
                "confidence": 0.0,
                "reasoning": f"Error in analysis: {str(e)}"
            }

# Test the agent
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    agent = CustomerMatchingAgent()
    
    # Test with sample data
    source_customer = {
        "customer_id": "TEST123",
        "name": "John",
        "surname": "Doe",
        "email": "john.doe@example.com",
        "mobile_phone_number": "+1 555-123-4567",
        "country": "United States",
        "date_of_birth": "1980-01-01"
    }
    
    candidate_matches = [
        {
            "customer_id": "CAND456",
            "match_type": "email_exact",
            "similarity": 1.0,
            "customer_data": {
                "customer_id": "CAND456",
                "name": "Johnny",
                "surname": "Doe",
                "email": "john.doe@example.com",
                "mobile_phone_number": "+1 555-123-7890",
                "country": "USA",
                "date_of_birth": "1980-01-01"
            }
        }
    ]
    
    result = agent.analyze_customer_matches(source_customer, candidate_matches)
    print(f"Match analysis result: {json.dumps(result, indent=2)}")