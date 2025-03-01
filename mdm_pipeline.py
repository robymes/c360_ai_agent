"""
Main processing pipeline for the Customer MDM system.
"""

import logging
import argparse
from typing import Dict, Any
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    LOG_LEVEL,
    LOG_FILE
)
from db_connector import DuckDBConnector
from vector_store import CustomerVectorStore
from matching_agent import CustomerMatchingAgent
from dotenv import load_dotenv

from utils import setup_logging, create_timer, generate_family_id, format_processing_stats

# Set up logger
logger = logging.getLogger(__name__)

class MDMPipeline:
    """
    Main processing pipeline for customer matching and MDM.
    """
    
    def __init__(self, batch_size: int = BATCH_SIZE):
        """
        Initialize the MDM pipeline.
        
        Args:
            batch_size: Number of records to process in each batch
        """
        self.batch_size = batch_size
        self.db = None
        self.vector_store = None
        self.matching_agent = None
        self.stats = {
            "total_processed": 0,
            "matched": 0,
            "new_records": 0,
            "errors": 0,
            "confidence_sum": 0,
            "confidence_count": 0,
            "elapsed_time": 0
        }
        self.timer = create_timer()
    
    def initialize(self):
        """Initialize connections and components."""
        try:
            # Initialize database connector
            self.db = DuckDBConnector()
            
            # Initialize vector store
            self.vector_store = CustomerVectorStore()
            
            # Initialize matching agent
            self.matching_agent = CustomerMatchingAgent()
            
            logger.info("MDM pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MDM pipeline: {e}")
            return False
    
    def close(self):
        """Close connections and clean up resources."""
        if self.db:
            self.db.close()
        
        if self.vector_store:
            self.vector_store.close()
        
        logger.info("MDM pipeline resources closed")
    
    def process_customer(self, customer: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single customer record.
        
        Args:
            customer: Customer data dictionary
            
        Returns:
            Dictionary with processing results
        """
        result = {
            "customer_id": customer.get("customer_id", "unknown"),
            "is_match": False,
            "family_id": None,
            "matched_to": None,
            "confidence": 0.0
        }
        
        try:
            # Search for similar customers in the vector store
            similar_customers = self.vector_store.find_similar_customers(customer)
            
            # If we found potential matches, use the agent to analyze
            if similar_customers:
                logger.info(f"Found {len(similar_customers)} potential matches for customer {customer['customer_id']}")
                
                # Use the matching agent to determine if any of the candidates match
                agent_result = self.matching_agent.analyze_customer_matches(customer, similar_customers)
                
                # Check if we have a match
                if agent_result.get("is_match", False):
                    matched_customer_id = agent_result.get("matched_customer_id")
                    matched_family_id = agent_result.get("matched_family_id")
                    confidence = agent_result.get("confidence", 0.0)
                    
                    # Ensure we have a valid family_id
                    if matched_family_id:
                        family_id = matched_family_id
                    else:
                        # If no family_id, try to get it from the matched customer
                        matched_customer = self.vector_store.get_customer_by_id(matched_customer_id)
                        if matched_customer and matched_customer.get("family_id"):
                            family_id = matched_customer["family_id"]
                        else:
                            # If still no family_id, generate a new one
                            family_id = generate_family_id()
                    
                    # Update result with match info
                    result["is_match"] = True
                    result["family_id"] = family_id
                    result["matched_to"] = matched_customer_id
                    result["confidence"] = confidence
                    
                    # Update stats
                    self.stats["matched"] += 1
                    self.stats["confidence_sum"] += confidence
                    self.stats["confidence_count"] += 1
                    
                    logger.info(f"Customer {customer['customer_id']} matched to {matched_customer_id} with confidence {confidence:.4f}")
                    logger.info(f"Assigned family ID: {family_id}")
                else:
                    # No match found - generate new family ID
                    family_id = generate_family_id()
                    result["family_id"] = family_id
                    self.stats["new_records"] += 1
                    
                    logger.info(f"No match found for customer {customer['customer_id']}, assigned new family ID: {family_id}")
            else:
                # No potential matches found - generate new family ID
                family_id = generate_family_id()
                result["family_id"] = family_id
                self.stats["new_records"] += 1
                
                logger.info(f"No similar customers found for {customer['customer_id']}, assigned new family ID: {family_id}")
            
            # Add the customer to the vector store with the assigned family ID
            self.vector_store.add_customer(customer, result["family_id"])
            
            # Update the customer in DuckDB
            self.db.update_customer_family_id(customer["customer_id"], result["family_id"])
            
            # Update total processed count
            self.stats["total_processed"] += 1
            
        except Exception as e:
            logger.error(f"Error processing customer {customer.get('customer_id', 'unknown')}: {e}")
            self.stats["errors"] += 1
            result["error"] = str(e)
        
        return result
    
    def process_all_customers(self):
        """Process all customers in the database."""
        try:
            # Count total customers for progress tracking
            total_customers = self.db.count_customers()
            logger.info(f"Starting processing for {total_customers} customers")
            
            # Reset timer and stats
            self.timer["reset"]()
            self.stats = {
                "total_processed": 0,
                "matched": 0,
                "new_records": 0,
                "errors": 0,
                "confidence_sum": 0,
                "confidence_count": 0,
                "elapsed_time": 0
            }
            
            # Process customers in batches
            with tqdm(total=total_customers, desc="Processing customers") as progress_bar:
                for batch in self.db.get_customers_iterator(self.batch_size):
                    for _, customer in batch.iterrows():
                        # Convert Series to dict for processing
                        customer_dict = customer.to_dict()
                        
                        # Process the customer
                        result = self.process_customer(customer_dict)  # noqa: F841
                        
                        # Update progress
                        progress_bar.update(1)
                        
                        # Periodically log progress
                        if self.stats["total_processed"] % 100 == 0 and self.stats["total_processed"] > 0:
                            elapsed = self.timer["elapsed"]()
                            rate = self.stats["total_processed"] / elapsed if elapsed > 0 else 0
                            logger.info(f"Processed {self.stats['total_processed']} customers. "
                                      f"Rate: {rate:.2f} customers/sec. "
                                      f"Matches: {self.stats['matched']} ({self.stats['matched']/self.stats['total_processed']*100:.2f}%)")
            
            # Update final elapsed time
            self.stats["elapsed_time"] = self.timer["elapsed"]()
            
            # Log final stats
            logger.info(f"Processing complete. Total: {self.stats['total_processed']}, "
                      f"Matched: {self.stats['matched']}, New: {self.stats['new_records']}, "
                      f"Errors: {self.stats['errors']}, Time: {self.stats['elapsed_time']:.2f}s")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error in process_all_customers: {e}")
            self.stats["elapsed_time"] = self.timer["elapsed"]()
            return self.stats

def main():
    """Main entry point for the MDM processing pipeline."""
    load_dotenv()
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Customer MDM Processing Pipeline")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for processing")
    parser.add_argument("--log-level", type=str, default=LOG_LEVEL, help="Logging level")
    parser.add_argument("--log-file", type=str, default=LOG_FILE, help="Log file path")
    args = parser.parse_args()
    
    # Setup logging
    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    setup_logging(level=numeric_level, log_file=args.log_file)
    
    # Initialize and run the pipeline
    pipeline = MDMPipeline(batch_size=args.batch_size)
    try:
        if pipeline.initialize():
            logger.info("Starting MDM processing pipeline...")
            stats = pipeline.process_all_customers()
            
            # Print final stats
            formatted_stats = format_processing_stats(stats)
            print("\n" + formatted_stats)
            
            logger.info("MDM processing pipeline completed successfully")
        else:
            logger.error("Failed to initialize MDM pipeline")
    finally:
        pipeline.close()

if __name__ == "__main__":
    main()