from typing import List, Dict, Optional, Any
import requests
import logging
from pathlib import Path
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentSearcher:
    def __init__(self, es_host: str = "http://localhost:9200", index_name: str = "content_index"):
        """
        Initialize Elasticsearch searcher using requests.
        
        Args:
            es_host: Elasticsearch server URL (e.g., "http://localhost:9200")
            index_name: Name of the index to search against
        """
        self.es_host = es_host
        self.index_name = index_name
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        
        # Verify connection and index
        self._verify_connection()
        self._verify_index()
        
        # Load query templates
        self.query_templates = self._load_query_templates()

    def _verify_connection(self):
        """Verify we can connect to Elasticsearch"""
        try:
            response = self.session.get(self.es_host)
            response.raise_for_status()
            if not response.json().get("version"):
                raise ConnectionError("Invalid Elasticsearch response")
            logger.info(f"Connected to Elasticsearch {response.json()['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise

    def _verify_index(self):
        """Verify the index exists"""
        try:
            url = f"{self.es_host}/{self.index_name}"
            response = self.session.head(url)
            if response.status_code == 404:
                logger.warning(f"Index {self.index_name} does not exist")
                # In production, you might want to create the index here
                # self._create_index()
        except Exception as e:
            logger.error(f"Index verification failed: {str(e)}")
            raise

    def _load_query_templates(self) -> Dict:
        """Load query templates from JSON file if exists"""
        templates_path = Path(__file__).parent / "query_templates.json"
        try:
            if templates_path.exists():
                with open(templates_path) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Failed to load query templates: {str(e)}")
            return {}

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to Elasticsearch"""
        url = f"{self.es_host}/{endpoint}"
        try:
            logger.debug(f"Sending {method} to {url}")
            if data:
                logger.debug(f"Request body: {json.dumps(data, indent=2)}")
            
            if method == "GET":
                response = self.session.get(url, json=data)
            elif method == "POST":
                response = self.session.post(url, json=data)
            elif method == "PUT":
                response = self.session.put(url, json=data)
            elif method == "DELETE":
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def search(self, query: Dict, size: int = 10, scroll: Optional[str] = None) -> Dict:
        """
        Execute search query against Elasticsearch.
        
        Args:
            query: The Elasticsearch query DSL
            size: Number of results to return
            scroll: Time to keep the search context alive (e.g., "1m")
            
        Returns:
            Raw Elasticsearch response
        """
        endpoint = f"{self.index_name}/_search"
        if scroll:
            query["size"] = size
            query["scroll"] = scroll
        else:
            query["size"] = size
            
        return self._make_request("POST", endpoint, query)

    def scroll(self, scroll_id: str, scroll: str = "1m") -> Dict:
        """
        Scroll through search results.
        
        Args:
            scroll_id: The scroll ID from initial search
            scroll: Time to keep the scroll context alive
            
        Returns:
            Next batch of results
        """
        endpoint = "_search/scroll"
        data = {
            "scroll": scroll,
            "scroll_id": scroll_id
        }
        return self._make_request("POST", endpoint, data)

    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Get a single document by ID.
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            The document if found, None otherwise
        """
        endpoint = f"{self.index_name}/_doc/{doc_id}"
        try:
            return self._make_request("GET", endpoint)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def count_documents(self, query: Dict = None) -> int:
        """
        Count documents matching a query.
        
        Args:
            query: Optional query to filter documents
            
        Returns:
            Number of matching documents
        """
        endpoint = f"{self.index_name}/_count"
        data = query if query else {"query": {"match_all": {}}}
        response = self._make_request("POST", endpoint, data)
        return response.get("count", 0)

    def search_all(self, query: Dict = None, batch_size: int = 100) -> List[Dict]:
        """
        Retrieve all documents matching a query using scroll API.
        
        Args:
            query: Optional query to filter documents
            batch_size: Number of documents to retrieve per batch
            
        Returns:
            List of all matching documents
        """
        if query is None:
            query = {"query": {"match_all": {}}}
            
        # Initial search
        search_query = query.copy()
        search_query["size"] = batch_size
        response = self.search(search_query, scroll="1m")
        scroll_id = response.get("_scroll_id")
        hits = response.get("hits", {}).get("hits", [])
        results = [hit["_source"] for hit in hits]
        
        # Continue scrolling
        while hits:
            response = self.scroll(scroll_id)
            scroll_id = response.get("_scroll_id")
            hits = response.get("hits", {}).get("hits", [])
            results.extend([hit["_source"] for hit in hits])
            
        return results

if __name__ == "__main__":
    # Test the searcher
    searcher = ContentSearcher()
    
    # Test basic search
    print("Testing basic search...")
    test_query = {
        "query": {
            "match_all": {}
        },
        "size": 2
    }
    results = searcher.search(test_query)
    print(f"Got {len(results['hits']['hits'])} results")
    for hit in results["hits"]["hits"]:
        print(f"- {hit['_id']}: {hit['_source'].get('title', 'No title')}")
    
    # Test document retrieval
    if results["hits"]["hits"]:
        doc_id = results["hits"]["hits"][0]["_id"]
        print(f"\nTesting get document for ID: {doc_id}")
        doc = searcher.get_document(doc_id)
        if doc:
            print(f"Document found: {doc['_source'].get('title', 'No title')}")
        else:
            print("Document not found")
    
    # Test count
    print("\nTesting document count...")
    count = searcher.count_documents()
    print(f"Total documents in index: {count}")
    
    # Test search_all (commented out as it might return many documents)
    # print("\nTesting search_all (first 5 docs)...")
    # all_docs = searcher.search_all({"size": 5})
    # print(f"Retrieved {len(all_docs)} documents")
    # for doc in all_docs[:5]:
    #     print(f"- {doc.get('title', 'No title')}")