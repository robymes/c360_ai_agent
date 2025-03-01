"""
Utility functions for the Customer MDM system.
"""

import logging
import uuid
import time
from typing import Dict, Any
from datetime import datetime

# Set up logger
logger = logging.getLogger(__name__)

def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    handlers = []
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    handlers.append(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        handlers.append(file_handler)
    
    # Configure logging format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Apply formatter to handlers
    for handler in handlers:
        handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers and add new ones
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    for handler in handlers:
        root_logger.addHandler(handler)
    
    logger.info(f"Logging configured with level {level}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")

def generate_family_id() -> str:
    """
    Generate a new family ID (UUID).
    
    Returns:
        New family ID as string
    """
    return str(uuid.uuid4())

def customer_to_text(customer: Dict[str, Any]) -> str:
    """
    Convert customer record to a text representation.
    
    Args:
        customer: Customer data dictionary
        
    Returns:
        Text representation of customer
    """
    parts = []
    
    # Add customer ID
    parts.append(f"ID: {customer.get('customer_id', 'unknown')}")
    
    # Add name and surname
    name = customer.get('name', '')
    surname = customer.get('surname', '')
    if name or surname:
        parts.append(f"Name: {name} {surname}")
    
    # Add email if available
    email = customer.get('email', '')
    if email:
        parts.append(f"Email: {email}")
    
    # Add phone if available
    phone = customer.get('mobile_phone_number', '')
    if phone:
        parts.append(f"Phone: {phone}")
    
    # Add country if available
    country = customer.get('country', '')
    if country:
        parts.append(f"Country: {country}")
    
    # Add date of birth if available
    dob = customer.get('date_of_birth', '')
    if dob:
        parts.append(f"Date of Birth: {dob}")
    
    # Add source
    source_id = customer.get('source_id', '')
    source_name = "E-commerce" if source_id == 1 else "POS" if source_id == 2 else f"Source {source_id}"
    parts.append(f"Source: {source_name}")
    
    return " | ".join(parts)

def format_processing_stats(stats: Dict[str, Any]) -> str:
    """
    Format processing statistics for display.
    
    Args:
        stats: Dictionary of processing statistics
        
    Returns:
        Formatted statistics string
    """
    lines = ["Processing Statistics:"]
    lines.append("-" * 40)
    
    # Format counts
    lines.append(f"Total records processed: {stats.get('total_processed', 0)}")
    lines.append(f"Records matched: {stats.get('matched', 0)}")
    lines.append(f"New records (no match): {stats.get('new_records', 0)}")
    
    # Format percentages
    total = stats.get('total_processed', 0)
    if total > 0:
        match_pct = stats.get('matched', 0) / total * 100
        new_pct = stats.get('new_records', 0) / total * 100
        lines.append(f"Match rate: {match_pct:.2f}%")
        lines.append(f"New record rate: {new_pct:.2f}%")
    
    # Format timing
    elapsed = stats.get('elapsed_time', 0)
    records_per_sec = total / elapsed if elapsed > 0 else 0
    lines.append(f"Total processing time: {elapsed:.2f} seconds")
    lines.append(f"Processing rate: {records_per_sec:.2f} records/second")
    
    # Format error rate
    errors = stats.get('errors', 0)
    error_rate = errors / total * 100 if total > 0 else 0
    lines.append(f"Errors: {errors} ({error_rate:.2f}%)")
    
    # Format confidence stats
    confidence_sum = stats.get('confidence_sum', 0)
    confidence_count = stats.get('confidence_count', 0)
    avg_confidence = confidence_sum / confidence_count if confidence_count > 0 else 0
    lines.append(f"Average match confidence: {avg_confidence:.4f}")
    
    return "\n".join(lines)

def create_timer():
    """
    Create a simple timer for performance measurement.
    
    Returns:
        Dictionary with timer functions
    """
    start_time = time.time()
    
    def elapsed():
        """Get elapsed time in seconds."""
        return time.time() - start_time
    
    def reset():
        """Reset the timer."""
        nonlocal start_time
        start_time = time.time()
    
    return {"elapsed": elapsed, "reset": reset}

def format_datetime(dt=None):
    """
    Format a datetime object or current time as a string.
    
    Args:
        dt: Optional datetime object, uses current time if None
        
    Returns:
        Formatted datetime string
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def compare_customer_records(
    record1: Dict[str, Any], 
    record2: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two customer records and provide detailed comparison.
    
    Args:
        record1: First customer record
        record2: Second customer record
        
    Returns:
        Dictionary with comparison results
    """
    comparison = {}
    
    # Helper function to compare field values
    def compare_field(field, fuzzy=False):
        val1 = str(record1.get(field, "")).lower().strip() if record1.get(field) is not None else ""
        val2 = str(record2.get(field, "")).lower().strip() if record2.get(field) is not None else ""
        
        # Handle empty values
        if not val1 and not val2:
            return {
                "match": True,
                "type": "both_empty",
                "similarity": 1.0,
                "value1": val1,
                "value2": val2
            }
        elif not val1 or not val2:
            return {
                "match": False,
                "type": "one_empty",
                "similarity": 0.0,
                "value1": val1,
                "value2": val2
            }
        
        # Exact match
        if val1 == val2:
            return {
                "match": True,
                "type": "exact",
                "similarity": 1.0,
                "value1": val1,
                "value2": val2
            }
        
        # Special normalization for phone numbers
        if field == "mobile_phone_number":
            norm1 = ''.join(filter(str.isdigit, val1))
            norm2 = ''.join(filter(str.isdigit, val2))
            if norm1 == norm2 and norm1:
                return {
                    "match": True,
                    "type": "normalized_exact",
                    "similarity": 1.0,
                    "value1": val1,
                    "value2": val2,
                    "normalized1": norm1,
                    "normalized2": norm2
                }
        
        # For fuzzy fields, calculate basic similarity
        if fuzzy:
            # Very basic similarity calculation based on character overlap
            # This is just a placeholder - CrewAI will do more sophisticated analysis
            common_chars = set(val1).intersection(set(val2))
            similarity = len(common_chars) / max(len(set(val1)), len(set(val2)))
            
            return {
                "match": similarity > 0.7,
                "type": "fuzzy",
                "similarity": similarity,
                "value1": val1,
                "value2": val2
            }
        
        # Default: not a match
        return {
            "match": False,
            "type": "different",
            "similarity": 0.0,
            "value1": val1,
            "value2": val2
        }
    
    # Compare core fields
    comparison["name"] = compare_field("name", fuzzy=True)
    comparison["surname"] = compare_field("surname", fuzzy=True)
    comparison["email"] = compare_field("email")
    comparison["mobile_phone_number"] = compare_field("mobile_phone_number")
    comparison["country"] = compare_field("country")
    comparison["date_of_birth"] = compare_field("date_of_birth")
    
    # Calculate overall similarity score
    field_weights = {
        "name": 0.15,
        "surname": 0.15,
        "email": 0.25,
        "mobile_phone_number": 0.25,
        "country": 0.05,
        "date_of_birth": 0.15
    }
    
    weighted_sum = 0
    weight_sum = 0
    
    for field, result in comparison.items():
        weight = field_weights.get(field, 0)
        similarity = result.get("similarity", 0)
        
        weighted_sum += weight * similarity
        weight_sum += weight
    
    overall_similarity = weighted_sum / weight_sum if weight_sum > 0 else 0
    
    # Determine overall match result
    # An exact match on email or phone is often a strong indicator
    email_match = comparison["email"]["match"] and comparison["email"]["type"] == "exact"
    phone_match = comparison["mobile_phone_number"]["match"] and comparison["mobile_phone_number"]["type"] in ["exact", "normalized_exact"]
    
    # Names match with matching DOB is also strong
    name_match = comparison["name"]["match"] and comparison["surname"]["match"]
    dob_match = comparison["date_of_birth"]["match"] and comparison["date_of_birth"]["type"] != "both_empty"
    
    is_match = (
        (email_match or phone_match) or
        (name_match and dob_match) or
        overall_similarity > 0.8
    )
    
    # Add overall results
    comparison["overall"] = {
        "match": is_match,
        "similarity": overall_similarity,
        "email_match": email_match,
        "phone_match": phone_match,
        "name_match": name_match,
        "dob_match": dob_match
    }
    
    return comparison

# Test utilities
if __name__ == "__main__":
    # Setup logging
    setup_logging(level=logging.INFO)
    
    # Test family ID generation
    family_id = generate_family_id()
    print(f"Generated family ID: {family_id}")
    
    # Test customer formatting
    test_customer = {
        "customer_id": "TEST123",
        "name": "John",
        "surname": "Doe",
        "email": "john.doe@example.com",
        "mobile_phone_number": "+1 555-123-4567",
        "country": "United States",
        "date_of_birth": "1980-01-01",
        "source_id": 1
    }
    
    formatted = customer_to_text(test_customer)
    print(f"Formatted customer: {formatted}")
    
    # Test record comparison
    test_customer2 = {
        "customer_id": "TEST456",
        "name": "Jon",  # Slight variation
        "surname": "Doe",
        "email": "jon.doe@example.com",  # Different email
        "mobile_phone_number": "555-123-4567",  # Same number but different format
        "country": "USA",  # Different format
        "date_of_birth": "1980-01-01",
        "source_id": 2
    }
    
    comparison = compare_customer_records(test_customer, test_customer2)
    print("\nRecord comparison:")
    for field, result in comparison.items():
        if field == "overall":
            print(f"\nOverall: {result['match']} (similarity: {result['similarity']:.4f})")
        else:
            print(f"{field}: {result['match']} (similarity: {result['similarity']:.4f}, type: {result['type']})")
    
    # Test stats formatting
    stats = {
        "total_processed": 1000,
        "matched": 820,
        "new_records": 180,
        "elapsed_time": 152.5,
        "errors": 12,
        "confidence_sum": 738.5,
        "confidence_count": 820
    }
    
    formatted_stats = format_processing_stats(stats)
    print(f"\n{formatted_stats}")
    
    # Test timer
    timer = create_timer()
    time.sleep(1.5)
    print(f"\nElapsed time: {timer['elapsed']():.2f} seconds")
    timer["reset"]()
    time.sleep(0.5)
    print(f"After reset: {timer['elapsed']():.2f} seconds")