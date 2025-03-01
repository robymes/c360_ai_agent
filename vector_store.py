"""
Chroma vector database setup and operations.
"""

import logging
import uuid
import json
import chromadb
import numpy as np
from typing import Dict, List, Any, Optional
from sentence_transformers import SentenceTransformer

from config import (
    CHROMA_PERSIST_DIR, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION,
    SIMILARITY_THRESHOLD,
    MAX_RECORDS_TO_COMPARE,
    SEARCH_WEIGHT_NAME,
    SEARCH_WEIGHT_SURNAME,
    SEARCH_WEIGHT_EMAIL,
    SEARCH_WEIGHT_PHONE,
    SEARCH_WEIGHT_COUNTRY,
    SEARCH_WEIGHT_DOB
)

# Set up logger
logger = logging.getLogger(__name__)

class CustomerVectorStore:
    """
    Handles operations with the Chroma vector database for customer records.
    """
    
    def __init__(self, persist_directory: str = CHROMA_PERSIST_DIR):
        """
        Initialize Chroma client and collection.
        
        Args:
            persist_directory: Directory to persist Chroma data
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_model = None
        self.initialize()
    
    def initialize(self):
        """Initialize Chroma client, embedding function, and collection."""
        try:
            # Initialize Chroma client
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Initialize embedding function
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            
            # Check if collection exists, create if not
            try:
                self.collection = self.client.get_collection(name=COLLECTION_NAME)
                logger.info(f"Retrieved existing collection '{COLLECTION_NAME}'")
            except Exception:
                # Create a new collection
                self.collection = self.client.create_collection(
                    name=COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Created new collection '{COLLECTION_NAME}'")
                
        except Exception as e:
            logger.error(f"Failed to initialize Chroma: {e}")
            raise
    
    def close(self):
        """Clean up resources."""
        # Chroma client doesn't require explicit closing
        pass
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def _prepare_customer_texts(self, customer: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare customer data as textual representations for embedding.
        
        Args:
            customer: Dictionary containing customer data
            
        Returns:
            Dictionary with field names and their text representations
        """
        text_fields = {}
        
        # Process name and surname (always present)
        text_fields["name"] = str(customer.get("name", "")).lower().strip()
        text_fields["surname"] = str(customer.get("surname", "")).lower().strip()
        
        # Process email (may be None)
        email = customer.get("email")
        text_fields["email"] = str(email).lower().strip() if email else ""
        
        # Process phone (may be None)
        phone = customer.get("mobile_phone_number")
        # Normalize phone numbers - remove non-digit characters
        if phone:
            normalized_phone = ''.join(filter(str.isdigit, str(phone)))
            text_fields["mobile_phone_number"] = normalized_phone
        else:
            text_fields["mobile_phone_number"] = ""
        
        # Process country (may be None)
        country = customer.get("country")
        text_fields["country"] = str(country).lower().strip() if country else ""
        
        # Process date of birth (may be None)
        dob = customer.get("date_of_birth")
        text_fields["date_of_birth"] = str(dob).strip() if dob else ""
        
        return text_fields
    
    def _create_combined_embeddings(self, text_fields: Dict[str, str]) -> Dict[str, List[float]]:
        """
        Create embeddings for each field and field combinations.
        
        Args:
            text_fields: Dictionary with field names and their text representations
            
        Returns:
            Dictionary with field names and their embedding vectors
        """
        embeddings = {}
        
        # Generate embeddings for individual fields if not empty
        for field, text in text_fields.items():
            if text:
                embeddings[field] = self._generate_embeddings([text])[0]
            else:
                # For empty fields, use zero vector
                embeddings[field] = [0.0] * EMBEDDING_DIMENSION
        
        # Create name+surname combined embedding
        name_surname_text = f"{text_fields['name']} {text_fields['surname']}".strip()
        if name_surname_text:
            embeddings["name_surname"] = self._generate_embeddings([name_surname_text])[0]
        else:
            embeddings["name_surname"] = [0.0] * EMBEDDING_DIMENSION
        
        # Create full profile combined embedding with weighted components
        full_profile = {}  # noqa: F841
        if any(text_fields.values()):
            # Generate combination of weighted embeddings for overall profile
            weighted_sum = np.zeros(EMBEDDING_DIMENSION)
            weights_sum = 0
            
            if text_fields["name"]:
                weighted_sum += np.array(embeddings["name"]) * SEARCH_WEIGHT_NAME
                weights_sum += SEARCH_WEIGHT_NAME
                
            if text_fields["surname"]:
                weighted_sum += np.array(embeddings["surname"]) * SEARCH_WEIGHT_SURNAME
                weights_sum += SEARCH_WEIGHT_SURNAME
                
            if text_fields["email"]:
                weighted_sum += np.array(embeddings["email"]) * SEARCH_WEIGHT_EMAIL
                weights_sum += SEARCH_WEIGHT_EMAIL
                
            if text_fields["mobile_phone_number"]:
                weighted_sum += np.array(embeddings["mobile_phone_number"]) * SEARCH_WEIGHT_PHONE
                weights_sum += SEARCH_WEIGHT_PHONE
                
            if text_fields["country"]:
                weighted_sum += np.array(embeddings["country"]) * SEARCH_WEIGHT_COUNTRY
                weights_sum += SEARCH_WEIGHT_COUNTRY
                
            if text_fields["date_of_birth"]:
                weighted_sum += np.array(embeddings["date_of_birth"]) * SEARCH_WEIGHT_DOB
                weights_sum += SEARCH_WEIGHT_DOB
                
            if weights_sum > 0:
                # Normalize
                weighted_sum = weighted_sum / weights_sum
                
            embeddings["full_profile"] = weighted_sum.tolist()
        else:
            embeddings["full_profile"] = [0.0] * EMBEDDING_DIMENSION
            
        return embeddings
    
    def add_customer(self, customer: Dict[str, Any], family_id: str = None) -> str:
        """
        Add a customer record to the vector store.
        
        Args:
            customer: Dictionary containing customer data
            family_id: Optional family ID, generated if not provided
            
        Returns:
            The family_id assigned to the customer
        """
        try:
            # Generate family_id if not provided
            if not family_id:
                family_id = str(uuid.uuid4())
            
            # Prepare ID for Chroma (must be a string)
            chroma_id = str(customer["customer_id"])
            
            # Prepare customer data for text representation
            text_fields = self._prepare_customer_texts(customer)
            
            # Generate embeddings for all fields
            embeddings_dict = self._create_combined_embeddings(text_fields)
            
            # Store the full customer data as metadata
            metadata = customer.copy()
            # Add family_id to metadata
            metadata["family_id"] = family_id
            # Convert all values to strings to avoid Chroma serialization issues
            for key, value in metadata.items():
                if value is None:
                    metadata[key] = ""
                else:
                    metadata[key] = str(value)
            
            # Add the main record with the full profile embedding
            self.collection.add(
                ids=[chroma_id],
                embeddings=[embeddings_dict["full_profile"]],
                metadatas=[metadata],
                documents=[json.dumps(text_fields)]
            )
            
            # For each field, create additional entries with specialized embeddings
            # This enables field-specific similarity searches
            for field, embedding in embeddings_dict.items():
                if field == "full_profile":
                    # Skip full profile as it's already added
                    continue
                
                # Create specialized ID for this field's embedding
                field_id = f"{chroma_id}_{field}"
                
                # Add field-specific metadata
                field_metadata = {
                    "customer_id": metadata["customer_id"],
                    "family_id": family_id, 
                    "field_type": field,
                    "field_value": text_fields[field] if field in text_fields else "",
                    "is_field_embedding": "true"
                }
                
                # Add the field-specific embedding
                self.collection.add(
                    ids=[field_id],
                    embeddings=[embedding],
                    metadatas=[field_metadata],
                    documents=[text_fields[field] if field in text_fields else ""]
                )
            
            logger.info(f"Added customer {chroma_id} with family_id {family_id}")
            return family_id
        
        except Exception as e:
            logger.error(f"Error adding customer to Chroma: {e}")
            raise
    
    def find_similar_customers(self, customer: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Find similar customer records in the vector store.
        
        Args:
            customer: Dictionary containing customer data to compare
            
        Returns:
            List of similar customer records with similarity scores
        """
        try:
            # Prepare customer data for text representation
            text_fields = self._prepare_customer_texts(customer)
            
            # Generate embeddings for all fields
            embeddings_dict = self._create_combined_embeddings(text_fields)
            
            # First check for exact matches on email and phone
            exact_matches = []
            
            # Check email exact match if available
            if text_fields["email"]:
                email_results = self.collection.query(
                    query_embeddings=[embeddings_dict["email"]],
                    where={"field_type": "email", "is_field_embedding": "true"},
                    n_results=5
                )
                
                for i, result_id in enumerate(email_results["ids"][0]):
                    # Extract the customer_id from the field-specific ID
                    customer_id = result_id.split("_")[0]
                    metadata = email_results["metadatas"][0][i]
                    
                    # Only consider exact matches for email
                    if metadata["field_value"].lower() == text_fields["email"].lower():
                        exact_matches.append({
                            "customer_id": customer_id,
                            "match_type": "email_exact",
                            "similarity": 1.0,
                            "field_value": metadata["field_value"]
                        })
            
            # Check phone exact match if available
            if text_fields["mobile_phone_number"]:
                phone_results = self.collection.query(
                    query_embeddings=[embeddings_dict["mobile_phone_number"]],
                    where={"field_type": "mobile_phone_number", "is_field_embedding": "true"},
                    n_results=5
                )
                
                for i, result_id in enumerate(phone_results["ids"][0]):
                    # Extract the customer_id from the field-specific ID
                    customer_id = result_id.split("_")[0]
                    metadata = phone_results["metadatas"][0][i]
                    
                    # Only consider normalized exact matches for phone
                    normalized_stored_phone = ''.join(filter(str.isdigit, metadata["field_value"]))
                    if normalized_stored_phone == text_fields["mobile_phone_number"]:
                        exact_matches.append({
                            "customer_id": customer_id,
                            "match_type": "phone_exact",
                            "similarity": 1.0,
                            "field_value": metadata["field_value"]
                        })
            
            # Next check for name+surname similarities
            name_surname_results = self.collection.query(
                query_embeddings=[embeddings_dict["name_surname"]],
                where={"field_type": "name_surname", "is_field_embedding": "true"},
                n_results=10
            )
            
            name_surname_matches = []
            for i, result_id in enumerate(name_surname_results["ids"][0]):
                customer_id = result_id.split("_")[0]
                similarity = name_surname_results["distances"][0][i]
                # Convert distance to similarity (Chroma returns distances, not similarities)
                similarity = 1.0 - similarity
                
                if similarity >= SIMILARITY_THRESHOLD:
                    name_surname_matches.append({
                        "customer_id": customer_id,
                        "match_type": "name_surname_similar",
                        "similarity": similarity
                    })
            
            # Finally, find similar profiles using full profile embedding
            profile_results = self.collection.query(
                query_embeddings=[embeddings_dict["full_profile"]],
                where={"is_field_embedding": {"$exists": False}},  # Only match main records, not field-specific
                n_results=MAX_RECORDS_TO_COMPARE
            )
            
            profile_matches = []
            # Process results
            for i, result_id in enumerate(profile_results["ids"][0]):
                similarity = 1.0 - profile_results["distances"][0][i]  # Convert distance to similarity
                metadata = profile_results["metadatas"][0][i]
                
                if similarity >= SIMILARITY_THRESHOLD:
                    profile_matches.append({
                        "customer_id": result_id,
                        "match_type": "profile_similar",
                        "similarity": similarity,
                        "family_id": metadata.get("family_id", ""),
                        "metadata": metadata
                    })
            
            # Combine results, prioritizing exact matches
            all_matches = []
            
            # Add exact matches first
            for match in exact_matches:
                # Get full customer details
                customer_details = self.get_customer_by_id(match["customer_id"])
                if customer_details:
                    match["customer_data"] = customer_details
                    all_matches.append(match)
            
            # Add name+surname matches next, if not already included
            existing_ids = {match["customer_id"] for match in all_matches}
            for match in name_surname_matches:
                if match["customer_id"] not in existing_ids:
                    customer_details = self.get_customer_by_id(match["customer_id"])
                    if customer_details:
                        match["customer_data"] = customer_details
                        all_matches.append(match)
                        existing_ids.add(match["customer_id"])
            
            # Add profile matches last, if not already included
            for match in profile_matches:
                if match["customer_id"] not in existing_ids:
                    # We already have metadata in profile matches
                    match["customer_data"] = match["metadata"]
                    all_matches.append(match)
                    existing_ids.add(match["customer_id"])
            
            # Sort by similarity score (highest first)
            all_matches.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Limit results
            all_matches = all_matches[:MAX_RECORDS_TO_COMPARE]
            
            logger.info(f"Found {len(all_matches)} potential matches for customer {customer.get('customer_id', 'new')}")
            return all_matches
            
        except Exception as e:
            logger.error(f"Error finding similar customers: {e}")
            return []
    
    def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a customer record by ID.
        
        Args:
            customer_id: The ID of the customer to retrieve
            
        Returns:
            Customer data dictionary or None if not found
        """
        try:
            # Query for the specific customer
            results = self.collection.get(
                ids=[customer_id],
                where={"is_field_embedding": {"$exists": False}}  # Only match main records
            )
            
            if results and results["ids"] and len(results["ids"]) > 0:
                return results["metadatas"][0]
            return None
        except Exception as e:
            logger.error(f"Error retrieving customer {customer_id}: {e}")
            return None
    
    def get_customers_by_family_id(self, family_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all customer records with a specific family ID.
        
        Args:
            family_id: The family ID to search for
            
        Returns:
            List of customer data dictionaries
        """
        try:
            # Query for customers with the specific family ID
            results = self.collection.get(
                where={"family_id": family_id, "is_field_embedding": {"$exists": False}}
            )
            
            if results and results["metadatas"]:
                return results["metadatas"]
            return []
        except Exception as e:
            logger.error(f"Error retrieving customers with family ID {family_id}: {e}")
            return []
            
    def count_customers(self) -> int:
        """
        Count the total number of main customer records (not field-specific).
        
        Returns:
            Count of customer records
        """
        try:
            results = self.collection.get(
                where={"is_field_embedding": {"$exists": False}}
            )
            return len(results["ids"])
        except Exception as e:
            logger.error(f"Error counting customers: {e}")
            return 0

# Simple test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    store = CustomerVectorStore()
    try:
        count = store.count_customers()
        print(f"Customer count in vector store: {count}")
        
        # Test customer addition
        if count == 0:
            test_customer = {
                "customer_id": "TEST123",
                "name": "John",
                "surname": "Doe",
                "email": "john.doe@example.com",
                "mobile_phone_number": "+1 555-123-4567",
                "country": "United States",
                "date_of_birth": "1980-01-01"
            }
            
            family_id = store.add_customer(test_customer)
            print(f"Added test customer with family_id: {family_id}")
            
            # Test retrieval
            retrieved = store.get_customer_by_id("TEST123")
            print(f"Retrieved customer: {retrieved}")
            
            # Test similarity search
            similar_customer = {
                "customer_id": "TEST456",
                "name": "Jon",  # Slightly different name
                "surname": "Doe",
                "email": "jon.doe@example.com",  # Different email
                "mobile_phone_number": "+1 555-123-4567",  # Same phone
                "country": "United States",
                "date_of_birth": "1980-01-01"
            }
            
            matches = store.find_similar_customers(similar_customer)
            print(f"Found {len(matches)} matches for similar customer")
            for match in matches:
                print(f"  Match: {match['match_type']} with similarity {match['similarity']:.4f} - {match['customer_id']}")
    finally:
        store.close()