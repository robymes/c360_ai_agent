"""
Configuration settings for the Customer MDM system.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# DuckDB configuration
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "../data/duckdb_demo.duckdb")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# Chroma configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "customer_profiles"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384  # Dimension for the all-MiniLM-L6-v2 model

# Similarity search configuration
SIMILARITY_THRESHOLD = 0.75
MAX_RECORDS_TO_COMPARE = 5
SEARCH_WEIGHT_NAME = 0.8
SEARCH_WEIGHT_SURNAME = 0.8
SEARCH_WEIGHT_EMAIL = 1.0
SEARCH_WEIGHT_PHONE = 0.9
SEARCH_WEIGHT_COUNTRY = 0.5
SEARCH_WEIGHT_DOB = 0.7

# CrewAI configuration
AGENT_MODEL = "gpt-4o"  # Use an appropriate model based on your requirements
# Azure OpenAI configuration
AZURE_OPENAI_API_VERSION = "2024-10-01-preview"  # Usa la versione più recente disponibile

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "mdm_process.log")

# Fields for matching
MATCH_FIELDS = [
    "name",
    "surname",
    "email",
    "mobile_phone_number",
    "country",
    "date_of_birth"
]