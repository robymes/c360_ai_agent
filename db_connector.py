"""
DuckDB connection handler for the Customer MDM system.
"""

import logging
import duckdb
import pandas as pd
from typing import Dict, Any, Generator, Optional

from config import DUCKDB_PATH, BATCH_SIZE

# Set up logger
logger = logging.getLogger(__name__)

class DuckDBConnector:
    """
    Handles connections and operations with DuckDB database.
    """
    
    def __init__(self, db_path: str = DUCKDB_PATH):
        """
        Initialize DuckDB connection.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        """Establish connection to DuckDB."""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"Connected to DuckDB at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to DuckDB: {e}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")
    
    def get_all_customers(self) -> pd.DataFrame:
        """
        Retrieve all customer records from DuckDB.
        
        Returns:
            DataFrame containing all customer records
        """
        try:
            query = "SELECT * FROM customers"
            result = self.conn.execute(query).fetchdf()
            logger.info(f"Retrieved {len(result)} customer records")
            return result
        except Exception as e:
            logger.error(f"Error retrieving customers: {e}")
            raise
    
    def get_customers_batch(self, offset: int = 0, limit: int = BATCH_SIZE) -> pd.DataFrame:
        """
        Retrieve a batch of customer records.
        
        Args:
            offset: Number of records to skip
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame containing the batch of customer records
        """
        try:
            query = f"SELECT * FROM customers LIMIT {limit} OFFSET {offset}"
            result = self.conn.execute(query).fetchdf()
            logger.info(f"Retrieved batch of {len(result)} customer records (offset={offset})")
            return result
        except Exception as e:
            logger.error(f"Error retrieving customer batch: {e}")
            raise
    
    def get_customers_iterator(self, batch_size: int = BATCH_SIZE) -> Generator[pd.DataFrame, None, None]:
        """
        Returns an iterator that yields batches of customer records.
        
        Args:
            batch_size: Size of each batch
            
        Yields:
            DataFrame containing a batch of customer records
        """
        offset = 0
        while True:
            batch = self.get_customers_batch(offset, batch_size)
            if batch.empty:
                break
            yield batch
            offset += batch_size
    
    def update_customer_family_id(self, customer_id: str, family_id: str) -> bool:
        """
        Update the family_id for a specific customer.
        
        Args:
            customer_id: The ID of the customer to update
            family_id: The new family_id to assign
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First check if family_id column exists
            columns = self.conn.execute("PRAGMA table_info(customers)").fetchdf()
            if "family_id" not in columns["name"].values:
                self.conn.execute("ALTER TABLE customers ADD COLUMN family_id VARCHAR")
                logger.info("Added family_id column to customers table")
            
            # Update the record
            query = f"UPDATE customers SET family_id = '{family_id}' WHERE customer_id = '{customer_id}'"
            self.conn.execute(query)
            logger.info(f"Updated customer {customer_id} with family_id {family_id}")
            return True
        except Exception as e:
            logger.error(f"Error updating customer {customer_id}: {e}")
            return False
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific customer by ID.
        
        Args:
            customer_id: The ID of the customer to retrieve
            
        Returns:
            Dictionary containing customer data or None if not found
        """
        try:
            query = f"SELECT * FROM customers WHERE customer_id = '{customer_id}'"
            result = self.conn.execute(query).fetchdf()
            if result.empty:
                logger.warning(f"Customer {customer_id} not found")
                return None
            return result.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"Error retrieving customer {customer_id}: {e}")
            return None
    
    def count_customers(self) -> int:
        """
        Count the total number of customer records.
        
        Returns:
            Total number of customer records
        """
        try:
            query = "SELECT COUNT(*) FROM customers"
            result = self.conn.execute(query).fetchone()[0]
            logger.info(f"Total customer count: {result}")
            return result
        except Exception as e:
            logger.error(f"Error counting customers: {e}")
            raise

# Simple test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = DuckDBConnector()
    try:
        total = db.count_customers()
        print(f"Total customers: {total}")
        
        # Test batch retrieval
        first_batch = db.get_customers_batch(0, 5)
        print(f"First 5 customers:\n{first_batch}")
    finally:
        db.close()