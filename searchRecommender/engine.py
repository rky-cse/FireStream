from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    content_id: str
    title: str
    score: float
    match_reasons: List[str]
    metadata: Dict[str, Any]

class RecommendationEngine:
    def __init__(self, db_config: Dict, es_host: str = "http://localhost:9200"):
        """
        Initialize the recommendation engine with database and Elasticsearch configurations.
        """
        self.db_config = db_config.copy()
        self.es_host = es_host
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all required components"""
        try:
            logger.info("Initializing emotion detector...")
            from searchRecommender.app.emotion import EmotionDetector
            self.emotion_detector = EmotionDetector()
            
            logger.info("Initializing intent classifier...")
            from searchRecommender.app.intent import IntentClassifier
            self.intent_classifier = IntentClassifier()
            
            logger.info("Initializing prosody analyzer...")
            from searchRecommender.app.prosody import ProsodyAnalyzer
            self.prosody_analyzer = ProsodyAnalyzer()
            
            logger.info("Initializing content searcher...")
            from searchRecommender.app.searcher import ContentSearcher
            self.searcher = ContentSearcher(self.es_host)
            
            logger.info("Initializing context fetcher...")
            from searchRecommender.app.user_context import UserContextFetcher
            self.context_fetcher = UserContextFetcher(self.db_config)
            
            logger.info("Initializing festival detector...")
            from searchRecommender.app.festival import FestivalDetector
            self.festival_detector = FestivalDetector(self.db_config)
            
            logger.info("Recommendation engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {str(e)}")
            raise RuntimeError(f"Engine initialization failed: {str(e)}")

    def _analyze_inputs(self, text: Optional[str], prosody: Optional[Dict]) -> Dict[str, Any]:
        """
        Analyze text and prosody inputs to extract emotions, intent, and other signals.
        Returns a dictionary with analysis results.
        """
        analysis = {
            'text_analysis': {'emotion': None, 'intent': None, 'entities': {}},
            'audio_analysis': {'mood': None, 'confidence': 0.0},
            'resolved_mood': 'neutral'
        }
        
        # Analyze text if provided
        if text and text.strip():
            try:
                analysis['text_analysis']['emotion'] = self.emotion_detector.get_dominant_mood(text)
                intent_result = self.intent_classifier.classify_intent(text)
                analysis['text_analysis']['intent'] = intent_result.intent
                analysis['text_analysis']['entities'] = intent_result.entities or {}
            except Exception as e:
                logger.warning(f"Text analysis failed: {str(e)}")
        
        # Analyze prosody if provided
        if prosody:
            try:
                audio_result = self.prosody_analyzer.analyze_prosody(prosody)
                analysis['audio_analysis']['mood'] = audio_result.mood
                analysis['audio_analysis']['confidence'] = audio_result.confidence
            except Exception as e:
                logger.warning(f"Prosody analysis failed: {str(e)}")
        
        # Resolve final mood (prioritize audio if confident)
        if analysis['audio_analysis']['confidence'] > 0.7:
            analysis['resolved_mood'] = analysis['audio_analysis']['mood']
        elif analysis['text_analysis']['emotion']:
            analysis['resolved_mood'] = analysis['text_analysis']['emotion']
        
        return analysis

    def get_recommendations(self, user_id: str, search_text: str, prosody: Optional[Dict] = None) -> List[SearchResult]:
        """
        Main method to get personalized recommendations based on:
        - user_id: Unique identifier for the user
        - search_text: The text query from the user
        - prosody: Audio features dictionary (optional)
        
        Returns a list of SearchResult objects ordered by relevance.
        """
        try:
            # 1. Analyze inputs
            analysis = self._analyze_inputs(search_text, prosody)
            
            # 2. Fetch user context
            try:
                user_context = self.context_fetcher.fetch_context(user_id)
                festivals = self.festival_detector.get_active_festivals(datetime.now())
            except Exception as e:
                logger.warning(f"Context fetching failed, using minimal context: {str(e)}")
                user_context = self._create_minimal_context(user_id)
                festivals = []
            
            # 3. Prepare search query
            search_query = self._build_search_query(
                search_text=search_text,
                analysis=analysis,
                user_context=user_context,
                festivals=festivals
            )
            
            # 4. Execute search
            raw_results = self.searcher.search(search_query)
            
            # 5. Process and return results
            return self._process_results(
                raw_results=raw_results,
                user_context=user_context,
                analysis=analysis
            )
            
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            # Return empty list if something goes wrong
            return []

    def _create_minimal_context(self, user_id: str):
        """Create a minimal user context when full context fetching fails"""
        from searchRecommender.app.user_context import UserContext
        return UserContext(
            user_id=user_id,
            preferred_genres=[],
            watch_history=[]
        )

    def _build_search_query(self, search_text: str, analysis: Dict, user_context, festivals: List[str]) -> Dict:
        """
        Build Elasticsearch query compatible with your index structure
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": search_text,
                                "fields": ["title^3", "description^2", "tags"],
                                "fuzziness": "AUTO"
                            }
                        }
                    ],
                    "filter": [],
                    "should": []
                }
            },
            "size": 10,
            "_source": ["title", "description", "genres", "mood_tags", "rating"]
        }

        # Mood filter if detected
        if analysis['resolved_mood'] != 'neutral':
            query["query"]["bool"]["filter"].append({
                "term": {"mood_tags": analysis['resolved_mood']}
            })

        # Boost preferred genres
        if user_context.preferred_genres:
            query["query"]["bool"]["should"].append({
                "terms": {
                    "genres": user_context.preferred_genres,
                    "boost": 1.5
                }
            })

        # Boost higher rated content if rating field exists
        query["query"]["bool"]["should"].append({
            "range": {
                "rating": {
                    "gte": 7,
                    "boost": 1.3
                }
            }
        })

        return query

    def _process_results(self, raw_results: Dict, user_context, analysis: Dict) -> List[SearchResult]:
        """
        Process results with your actual index fields
        """
        processed = []
        
        for hit in raw_results.get('hits', {}).get('hits', []):
            source = hit.get('_source', {})
            score = hit.get('_score', 0)
            content_id = hit.get('_id', '')
            
            # Generate match reasons
            match_reasons = []
            
            # Mood match
            if (analysis['resolved_mood'] != 'neutral' and 
                analysis['resolved_mood'] in source.get('mood_tags', [])):
                match_reasons.append(f"Mood: {analysis['resolved_mood']}")
            
            # Genre match
            if user_context.preferred_genres and any(
                genre in user_context.preferred_genres 
                for genre in source.get('genres', [])
            ):
                match_reasons.append("Preferred genre")
            
            # Rating
            if source.get('rating', 0) >= 7:
                match_reasons.append(f"Highly rated ({source['rating']}/10)")
            
            processed.append(SearchResult(
                content_id=content_id,
                title=source.get('title', 'Untitled'),
                score=score,
                match_reasons=match_reasons,
                metadata={
                    'genres': source.get('genres', []),
                    'mood': source.get('mood_tags', ['neutral'])[0],
                    'rating': source.get('rating', 0),
                    'description': source.get('description', 'No description available')
                }
            ))
        
        return sorted(processed, key=lambda x: x.score, reverse=True)
# Updated example usage
if __name__ == "__main__":
    # Configuration
    DB_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "database": "firestream_db",
        "user": "postgres",
        "password": "postgres"
    }
    
    # Initialize engine with more logging
    logging.basicConfig(level=logging.DEBUG)
    engine = RecommendationEngine(DB_CONFIG)
    
    # More realistic test cases
    test_cases = [
        {
            "user_id": "USR10001",
            "search_text": "i am feeling sad and looking for uplifting movies",
            "prosody": {'pitch': 120, 'energy': 0.8}
        },
        {
            "user_id": "USR10002", 
            "search_text": "I'm very sad so I'm doing something funny.",
            "prosody": None
        }
    ]
    
    for test in test_cases:
        print(f"\nTesting for user {test['user_id']}: {test['search_text']}")
        results = engine.get_recommendations(
            user_id=test['user_id'],
            search_text=test['search_text'],
            prosody=test['prosody']
        )
        
        print(f"\nTop recommendations:")
        for i, result in enumerate(results[:3], 1):
            print(f"\n{i}. {result.title} (Score: {result.score:.2f})")
            print(f"   Why: {', '.join(result.match_reasons)}")
            print(f"   Genres: {', '.join(result.metadata['genres'])}")
            if result.metadata['rating'] > 0:
                print(f"   Rating: {result.metadata['rating']}/10")
            if result.metadata['description']:
                print(f"   Description: {result.metadata['description'][:100]}...")
