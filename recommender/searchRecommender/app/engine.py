from datetime import datetime
from typing import List, Dict, Optional
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
            from app.emotion import EmotionDetector
            self.emotion_detector = EmotionDetector()
            
            logger.info("Initializing intent classifier...")
            from app.intent import IntentClassifier
            self.intent_classifier = IntentClassifier()
            
            logger.info("Initializing prosody analyzer...")
            from app.prosody import ProsodyAnalyzer
            self.prosody_analyzer = ProsodyAnalyzer()
            
            logger.info("Initializing content searcher...")
            from app.searcher import ContentSearcher
            self.searcher = ContentSearcher(self.es_host)
            
            logger.info("Initializing context fetcher...")
            from app.user_context import UserContextFetcher
            self.context_fetcher = UserContextFetcher(self.db_config)
            
            logger.info("Initializing festival detector...")
            from app.festival import FestivalDetector
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
            
            # 3. Prepare search parameters
            search_params = self._prepare_search_params(
                analysis=analysis,
                user_context=user_context,
                festivals=festivals
            )
            
            # 4. Execute search and get results
            search_results = self.searcher.search(
                query=search_text,
                filters=search_params['filters'],
                boost_params=search_params['boosts'],
                size=10  # Default number of results
            )
            
            # 5. Process and return results
            return self._process_results(
                raw_results=search_results,
                user_context=user_context,
                search_params=search_params
            )
            
        except Exception as e:
            logger.error(f"Recommendation failed: {str(e)}")
            # Return empty list if something goes wrong
            return []

    def _create_minimal_context(self, user_id: str):
        """Create a minimal user context when full context fetching fails"""
        from app.user_context import UserContext
        return UserContext(
            user_id=user_id,
            preferred_genres=[],
            watch_history=[]
        )

    def _prepare_search_params(self, analysis: Dict, user_context, festivals: List[str]) -> Dict:
        """
        Prepare search parameters based on analysis and user context.
        Returns filters and boost parameters for the search.
        """
        params = {
            'filters': {},
            'boosts': {}
        }
        
        # Add mood filters/boosts
        if analysis['resolved_mood'] != 'neutral':
            params['filters']['mood'] = analysis['resolved_mood']
            params['boosts']['mood_match'] = 1.5
            
        # Add user preference boosts
        if user_context.preferred_genres:
            params['boosts']['preferred_genres'] = {
                'values': user_context.preferred_genres,
                'factor': 1.3
            }
        
        # Add festival boosts if active
        if festivals:
            params['boosts']['festival_theme'] = {
                'values': festivals,
                'factor': 1.2
            }
        
        # Add temporal context boosts
        params['boosts']['time_of_day'] = user_context.time_of_day
        params['boosts']['day_type'] = user_context.day_of_week
        
        return params

    def _process_results(self, raw_results: List[Dict], user_context, search_params: Dict) -> List[SearchResult]:
        """
        Process raw search results into SearchResult objects with explanations.
        Applies personalization scoring based on user context.
        """
        processed = []
        
        for result in raw_results:
            # Calculate personalization score
            base_score = result.get('score', 0)
            personalization_factor = 1.0
            
            # Apply genre preference boost
            if user_context.preferred_genres and any(
                genre in user_context.preferred_genres 
                for genre in result.get('genres', [])
            ):
                personalization_factor *= search_params['boosts'].get('preferred_genres', {}).get('factor', 1.0)
            
            # Create match reasons
            match_reasons = []
            if 'mood' in search_params['filters'] and search_params['filters']['mood'] in result.get('mood_tags', []):
                match_reasons.append(f"Mood: {search_params['filters']['mood']}")
            
            if any(genre in user_context.preferred_genres for genre in result.get('genres', [])):
                match_reasons.append("Matches preferred genres")
            
            processed.append(SearchResult(
                content_id=result['id'],
                title=result['title'],
                score=base_score * personalization_factor,
                match_reasons=match_reasons,
                metadata={
                    'genres': result.get('genres', []),
                    'mood': result.get('mood', 'neutral')
                }
            ))
        
        # Sort by final score
        return sorted(processed, key=lambda x: x.score, reverse=True)

# Example usage
if __name__ == "__main__":
    # Configuration
    DB_CONFIG = {
        "host": "localhost",
        "port": 5432,
        "database": "firestream_db",
        "user": "postgres",
        "password": "postgres"
    }
    
    # Initialize engine
    engine = RecommendationEngine(DB_CONFIG)
    
    # Example request
    user_id = "USR10001"
    search_text = "I'm feeling happy and want to watch something fun"
    prosody = {
        'pitch': 120,
        'energy': 0.8,
        'speech_rate': 1.2
    }
    
    # Get recommendations
    results = engine.get_recommendations(user_id, search_text, prosody)
    
    # Print results
    print(f"\nRecommendations for user {user_id}:")
    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. {result.title} (Score: {result.score:.2f})")
        if result.match_reasons:
            print("   Why:", ", ".join(result.match_reasons))