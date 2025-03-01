# Customer MDM (Master Data Management) System with AI

A Python-based Master Data Management system for customer data that leverages AI to identify and match duplicate customer records across different channels. The system uses Chroma for vector similarity search and CrewAI for intelligent entity matching.

## Overview

This system processes customer records from a DuckDB database to:

1. Find potential duplicate records using vector embeddings and similarity search
2. Use AI-powered analysis to determine if records represent the same person
3. Assign family IDs to group records belonging to the same customer
4. Store processed records with their family IDs in both DuckDB and Chroma

## Features

- **Batch Processing**: Efficiently processes customer records in configurable batches
- **Vector Similarity Search**: Uses Chroma and sentence embeddings to find similar records
- **AI-Powered Matching**: Leverages CrewAI and AI agents to make intelligent matching decisions
- **Field-Level Analysis**: Performs detailed analysis of name variations, typos, and data inconsistencies
- **Robust Duplicate Detection**: Goes beyond exact matching to identify duplicates with variations or missing data
- **Performance Tracking**: Monitors processing rates, match rates, and confidence scores

## Architecture

The system consists of several components:

- **DuckDB Connector**: Reads and updates customer records in DuckDB
- **Vector Store**: Manages the Chroma vector database for customer records
- **Matching Agent**: Uses CrewAI to analyze and determine if records match
- **MDM Pipeline**: Orchestrates the overall process flow

## Requirements

- Python 3.8+
- DuckDB
- Chroma DB
- CrewAI
- LangChain
- Sentence Transformers
- OpenAI API key for CrewAI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/robymes/c360_ai_agent.git
cd c360_ai_agent
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

4. Update the configuration in `config.py` if needed.

## Usage

Run the MDM pipeline with default settings:

```bash
python mdm_pipeline.py
```

### Command-line Options

- `--batch-size`: Number of records to process in each batch (default: 100)
- `--log-level`: Logging level (default: INFO)
- `--log-file`: Path to log file (default: mdm_process.log)

Example:

```bash
python mdm_pipeline.py --batch-size 200 --log-level DEBUG --log-file custom_log.log
```

## How It Works

### 1. Record Retrieval
The system reads customer records in batches from DuckDB.

### 2. Vector Search
For each customer record:
- The system generates embeddings for various fields (name, email, phone, etc.)
- It performs a similarity search in Chroma to find potential matches
- It ranks and filters matches based on similarity scores

### 3. AI Analysis
If potential matches are found:
- The CrewAI agent analyzes the source record and potential matches
- It performs detailed field-by-field analysis
- It considers patterns like name variations, typos, and formatting differences
- It determines if any record is a match and provides a confidence score

### 4. Family ID Assignment
Based on the analysis:
- If a match is found, the record is assigned the family ID of the matching record
- If no match is found, a new family ID is generated

### 5. Database Updates
- The record with its family ID is added to the Chroma vector store
- The family ID is updated in the DuckDB database

## Optimization for Matching

The system uses several techniques to optimize matching:

1. **Field-Specific Embeddings**: Different embeddings are created for each field to enable targeted similarity searches
2. **Weighted Profile Embeddings**: The overall profile embedding combines individual field embeddings with configurable weights
3. **Multi-Level Search**: Searches first for exact matches on email/phone, then for similar names, and finally for overall profile similarity
4. **Detailed Comparison Logic**: The matching agent receives detailed field-by-field comparisons to make informed decisions
5. **Confidence Scoring**: Each match is assigned a confidence score to indicate the reliability of the match

## License

This project is licensed under the MIT License - see the LICENSE file for details.