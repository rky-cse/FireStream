from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
import requests
import json
from searchRecommender.app.user_context import UserContext
from searchRecommender.app.intent import SearchIntent
from searchRecommender.app.festival import FestivalDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Recommendation:
    content_id: str
    title: str
    score: float
    match_reasons: List[str]

class ContentRecommender:
    def __init__(self, db_config: Dict, es_host: str = "http://localhost:9200"):
        self.db_config = db_config
        self.festival_detector = FestivalDetector(db_config)
        self.es_host = es_host
        self.index_name = "content_index"
        
        # Verify index exists and has correct mapping
        self._verify_index_mapping()
        
        # Weightings for different recommendation factors
        self.weights = {
            'mood_match': 0.3,
            'genre_preference': 0.25,
            'context_match': 0.2,
            'social_signal': 0.15,
            'popularity': 0.1
        }

    def _verify_index_mapping(self):
        """Ensure the index has correct field mappings"""
        try:
            mapping = requests.get(f"{self.es_host}/{self.index_name}/_mapping").json()
            if self.index_name not in mapping:
                self._create_index_with_mapping()
        except Exception as e:
            logger.error(f"Failed to verify index mapping: {str(e)}")
            raise

    def _create_index_with_mapping(self):
        """Create index with proper field mappings"""
        mapping = {
            "mappings": {
                "properties": {
                    "mood_tags": {"type": "keyword"},
                    "genre": {"type": "keyword"},
                    "actors": {"type": "text"},
                    "title": {
                        "type": "text",
                        "fields": {
                            "keyword": {"type": "keyword"}
                        }
                    }
                }
            }
        }
        try:
            response = requests.put(
                f"{self.es_host}/{self.index_name}",
                headers={"Content-Type": "application/json"},
                json=mapping
            )
            response.raise_for_status()
            logger.info("Created index with proper mappings")
        except Exception as e:
            logger.error(f"Failed to create index: {str(e)}")
            raise

    def _es_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to Elasticsearch"""
        url = f"{self.es_host}/{endpoint}"
        headers = {"Content-Type": "application/json"}
        
        try:
            logger.debug(f"Sending {method} to {url}")
            if data:
                logger.debug(f"Request body: {json.dumps(data, indent=2)}")
            
            if method == "GET":
                response = requests.get(url, headers=headers, json=data, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def _build_es_query(self, intent: SearchIntent, intent_entities: Dict) -> Dict:
        """Build query that will actually match documents"""
        query = {
            "size": 100,
            "query": {
                "bool": {
                    "must": [],
                    "should": [],
                    "filter": []
                }
            }
        }
        
        # Mood search - match any mood in the array
        if intent == SearchIntent.MOOD and 'mood' in intent_entities:
            query["query"]["bool"]["must"].append({
                "match": {
                    "mood_tags": intent_entities['mood']
                }
            })
        
        # Genre search - match any genre in the array
        elif intent == SearchIntent.GENRE and 'genres' in intent_entities:
            query["query"]["bool"]["must"].append({
                "terms": {
                    "genre": intent_entities['genres']
                }
            })
        
        # For general recommendations, return all content
        else:
            query["query"] = {"match_all": {}}
        
        return query

    def _get_base_content(self, intent: SearchIntent, intent_entities: Dict) -> List[Dict]:
        """Get content from Elasticsearch with proper field handling"""
        try:
            query = self._build_es_query(intent, intent_entities)
            logger.debug(f"Executing query: {json.dumps(query, indent=2)}")
            
            result = self._es_request("POST", f"{self.index_name}/_search", query)
            
            content_list = []
            for hit in result.get("hits", {}).get("hits", []):
                content = hit.get("_source", {})
                content["content_id"] = hit.get("_id", "")
                
                # Ensure all required fields exist
                content.setdefault("mood_tags", [])
                content.setdefault("genre", [])
                content.setdefault("rating", 0)
                
                content_list.append(content)
            
            logger.info(f"Found {len(content_list)} matching documents")
            return content_list
            
        except Exception as e:
            logger.error(f"Failed to get base content: {str(e)}")
            return []

    def _score_content(self, content: Dict, context: UserContext, intent: SearchIntent) -> tuple:
        """Calculate recommendation score"""
        score = 0.0
        match_reasons = []
        
        content_moods = content.get("mood_tags", [])
        content_genres = content.get("genre", [])
        content_rating = content.get("rating", 0)
        
        # 1. Mood matching
        if intent == SearchIntent.MOOD and hasattr(context, "mood"):
            if context.mood.lower() in [m.lower() for m in content_moods]:
                score += self.weights["mood_match"]
                match_reasons.append(f"matches {context.mood} mood")
        
        # 2. Genre preference matching
        preferred_genres = getattr(context, "preferred_genres", [])
        if preferred_genres:
            genre_overlap = len(set(content_genres) & set(preferred_genres))
            if genre_overlap > 0:
                score += min(0.5, genre_overlap * 0.1) * self.weights["genre_preference"]
                match_reasons.append(f"matches {genre_overlap} preferred genres")
        
        # [Include rest of your scoring logic...]
        
        return score, match_reasons

    def recommend(self, context: UserContext, intent: SearchIntent, intent_entities: Dict = None, limit: int = 10) -> List[Recommendation]:
        """Generate recommendations with proper error handling"""
        if intent_entities is None:
            intent_entities = {}
        
        try:
            logger.info(f"Generating recommendations for intent: {intent.name}")
            
            base_content = self._get_base_content(intent, intent_entities)
            if not base_content:
                logger.warning("No content found matching criteria")
                return []
            
            # Score and sort content
            scored_content = []
            for content in base_content:
                score, reasons = self._score_content(content, context, intent)
                scored_content.append({
                    "content": content,
                    "score": score,
                    "match_reasons": reasons
                })
            
            # Sort by score and return top results
            scored_content.sort(key=lambda x: x["score"], reverse=True)
            return [
                Recommendation(
                    content_id=item["content"]["content_id"],
                    title=item["content"]["title"],
                    score=item["score"],
                    match_reasons=item["match_reasons"]
                )
                for item in scored_content[:limit]
            ]
            
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            raise

if __name__ == "__main__":
    from searchRecommender.app.user_context import UserContextFetcher
    from searchRecommender.app.intent import SearchIntent
    
    # Enable debug logging
    logging.getLogger().setLevel(logging.DEBUG)
    
    # Test configuration
    TEST_DB_CONFIG = {
        "user": "postgres",
        "password": "postgres",
        "host": "localhost",
        "port": "5432"
    }
    
    print("=== Testing Content Recommender ===")
    
    try:
        # Create test context with required attributes
        context_fetcher = UserContextFetcher(TEST_DB_CONFIG)
        test_context = context_fetcher.fetch_context("USR10001")
        test_context.mood = "happy"  # Explicitly set test mood
        test_context.preferred_genres = ["Comedy", "Drama"]  # Set test preferences
        
        # Initialize recommender
        recommender = ContentRecommender(TEST_DB_CONFIG)
        
        # First verify the index contains data
        print("\nVerifying index content...")
        count_response = requests.get(f"{recommender.es_host}/{recommender.index_name}/_count").json()
        print(f"Total documents in index: {count_response.get('count', 0)}")
        
        # Test mood-based recommendations
        print("\nTesting mood-based recommendations for 'happy':")
        mood_recs = recommender.recommend(
            context=test_context,
            intent=SearchIntent.MOOD,
            intent_entities={"mood": "happy"}
        )
        for rec in mood_recs[:5]:
            print(f"- {rec.title} (Score: {rec.score:.2f}) - {', '.join(rec.match_reasons)}")
        
        # Test genre-based recommendations
        print("\nTesting genre-based recommendations for ['Comedy']:")
        genre_recs = recommender.recommend(
            context=test_context,
            intent=SearchIntent.GENRE,
            intent_entities={"genres": ["Comedy"]}
        )
        for rec in genre_recs[:5]:
            print(f"- {rec.title} (Score: {rec.score:.2f})")
        
        print("\n=== Tests completed ===")
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")