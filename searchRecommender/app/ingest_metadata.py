#!/usr/bin/env python3
import psycopg2
import requests
import json
import logging
from psycopg2.extras import DictCursor

# Configuration
DB_CONFIG = {
    "dbname": "firestream_db",
    "user": "postgres",
    "password": "postgres",
    "host": "localhost"
}
ES_URL = "http://localhost:9200"
INDEX_NAME = "content_index"
BATCH_SIZE = 100

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_index_exists():
    """Check if the Elasticsearch index exists"""
    try:
        response = requests.head(f"{ES_URL}/{INDEX_NAME}")
        if response.status_code == 200:
            logger.info(f"Index {INDEX_NAME} already exists")
            return True
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking index existence: {str(e)}")
        raise

def create_index():
    """Create Elasticsearch index with proper mapping"""
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "type": {"type": "keyword"},
                "genre": {"type": "keyword"},
                "release_year": {"type": "integer"},
                "duration": {"type": "integer"},
                "director": {"type": "keyword"},
                "actors": {"type": "keyword"},
                "description": {"type": "text"},
                "rating": {"type": "float"},
                "mood_tags": {"type": "keyword"}
            }
        }
    }
    
    try:
        response = requests.put(
            f"{ES_URL}/{INDEX_NAME}",
            headers={"Content-Type": "application/json"},
            json=mapping,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("Successfully created index")
            return True
        elif response.status_code == 400 and "resource_already_exists_exception" in response.text:
            logger.info("Index already exists - continuing")
            return True
        else:
            logger.error(f"Failed to create index: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating index: {str(e)}")
        return False

def index_documents():
    """Index documents from PostgreSQL to Elasticsearch"""
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(**DB_CONFIG, cursor_factory=DictCursor)
        cursor = conn.cursor()
        
        # Get total count of documents
        cursor.execute("SELECT COUNT(*) FROM content")
        total_docs = cursor.fetchone()[0]
        logger.info(f"Found {total_docs} documents to index")
        
        # Process documents in batches
        success_count = 0
        for offset in range(0, total_docs, BATCH_SIZE):
            cursor.execute(
                "SELECT * FROM content ORDER BY content_id LIMIT %s OFFSET %s",
                (BATCH_SIZE, offset)
            )
            
            # Prepare bulk request data
            bulk_data = []
            for row in cursor:
                # Action metadata
                bulk_data.append(json.dumps({
                    "index": {
                        "_index": INDEX_NAME,
                        "_id": row["content_id"]
                    }
                }))
                # Document data
                bulk_data.append(json.dumps({
                    "title": row["title"],
                    "type": row["type"],
                    "genre": row["genre"],
                    "release_year": row["release_year"],
                    "duration": row["duration"],
                    "director": row["director"],
                    "actors": row["actors"],
                    "description": row["description"],
                    "rating": row["rating"],
                    "mood_tags": row["mood_tags"]
                }))
            
            # Send bulk request to Elasticsearch
            try:
                response = requests.post(
                    f"{ES_URL}/_bulk",
                    headers={"Content-Type": "application/x-ndjson"},
                    data="\n".join(bulk_data) + "\n",
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if not result.get("errors"):
                        success_count += len(result["items"])
                        logger.info(f"Indexed {min(offset + BATCH_SIZE, total_docs)}/{total_docs} documents")
                    else:
                        errors = sum(1 for item in result["items"] if "error" in item["index"])
                        logger.warning(f"Batch had {errors} failures")
                else:
                    logger.error(f"Batch failed with status {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                continue
                
        return success_count
        
    except psycopg2.Error as e:
        logger.error(f"Database error: {str(e)}")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 0
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    try:
        logger.info("Starting data ingestion pipeline")
        
        # Check if index exists
        if not check_index_exists():
            # Create index if it doesn't exist
            if not create_index():
                logger.error("Failed to create index - exiting")
                exit(1)
        
        # Index documents
        indexed_count = index_documents()
        if indexed_count > 0:
            logger.info(f"Successfully indexed {indexed_count} documents")
            exit(0)
        else:
            logger.error("No documents were indexed")
            exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        exit(1)