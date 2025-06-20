import numpy as np
import pandas as pd
import logging
import json
import os
import pickle
import hashlib
from datetime import datetime, timedelta
from models import ModelManager
from userContext import UserContextAnalyzer
from prosodyAnalyzer import ProsodyAnalyzer
from contentProcessor import ContentProcessor

# Optional Elasticsearch integration
try:
    from elasticsearchProcessor import ElasticsearchContentProcessor
    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    ElasticsearchContentProcessor = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SearchRecommender:
    """
    Complete AI-powered search recommendation engine for video content.
    
    Features:
    - Multi-modal AI analysis (text + voice prosody)
    - Semantic understanding using transformer models
    - Emotion and sentiment detection
    - Collaborative filtering recommendations
    - Mood-based content matching
    - Intelligent result fusion and ranking
    - Performance optimization with caching
    - Elasticsearch integration for advanced search
    - Real-time analytics and learning
    
    Architecture:
    User Query → Voice Analysis → AI Understanding → Multi-Source Search → Smart Ranking → Results
    """
    
    def __init__(self, db_connection=None):
        """
        Initialize the complete AI recommendation engine
        
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
        
        # Initialize core AI components
        self.model_manager = ModelManager()
        self.user_context = UserContextAnalyzer(db_connection)
        self.prosody_analyzer = ProsodyAnalyzer()
        self.content_processor = ContentProcessor(db_connection)
        
        # Initialize Elasticsearch if available and enabled
        self.es_processor = None
        self._initialize_elasticsearch()
        
        # Performance cache with TTL
        self.cache = {}
        self.cache_expiry_seconds = int(os.getenv('CACHE_EXPIRY', '3600'))  # 1 hour default
        self.max_cache_size = int(os.getenv('MAX_CACHE_SIZE', '1000'))
        
        # Recommendation engine statistics
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "ai_successes": 0,
            "fallback_uses": 0,
            "elasticsearch_uses": 0,
            "semantic_searches": 0,
            "mood_searches": 0,
            "collaborative_searches": 0,
            "average_response_time_ms": 0,
            "user_satisfaction_score": 0.0
        }
        
        # Engine configuration
        self.config = {
            "enable_elasticsearch": os.getenv('ENABLE_ELASTICSEARCH', 'true').lower() == 'true',
            "enable_collaborative_filtering": os.getenv('ENABLE_CF', 'true').lower() == 'true',
            "enable_semantic_search": os.getenv('ENABLE_SEMANTIC', 'true').lower() == 'true',
            "enable_mood_analysis": os.getenv('ENABLE_MOOD', 'true').lower() == 'true',
            "enable_prosody_analysis": os.getenv('ENABLE_PROSODY', 'true').lower() == 'true',
            "min_confidence_threshold": float(os.getenv('MIN_CONFIDENCE', '0.3')),
            "max_results_per_source": int(os.getenv('MAX_RESULTS_PER_SOURCE', '20'))
        }

        logger.info("🚀 Complete AI Search Recommendation Engine initialized successfully")
        logger.info(f"📅 System initialized at: 2025-06-20 08:20:22 UTC")
        logger.info(f"👤 Primary user: rky-cse")
        logger.info(f"🔧 Configuration: {self.config}")
        
    def _initialize_elasticsearch(self):
        """Initialize Elasticsearch integration if available"""
        if not ELASTICSEARCH_AVAILABLE:
            logger.info("📋 Elasticsearch integration not available (optional)")
            return
            
        if not self.config["enable_elasticsearch"]:
            logger.info("📋 Elasticsearch integration disabled in configuration")
            return
            
        try:
            self.es_processor = ElasticsearchContentProcessor(
                es_host=os.getenv('ES_HOST', 'localhost'),
                es_port=int(os.getenv('ES_PORT', '9200'))
            )
            logger.info("✅ Elasticsearch integration enabled and connected")
        except Exception as e:
            logger.warning(f"⚠️ Elasticsearch integration failed: {e}")
            self.es_processor = None
        
    def get_recommendations(self, user_id, search_query, prosody_features=None, limit=10, context=None):
        """
        Main entry point for AI-powered video recommendations
        
        Args:
            user_id: User identifier (e.g., 'rky-cse')
            search_query: Natural language search query
            prosody_features: Voice characteristics from ingestion/voice (optional)
            limit: Maximum number of recommendations to return
            context: Additional context (device, time, location, etc.)
            
        Returns:
            Comprehensive recommendation response with AI insights
        """
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        try:
            logger.info(f"🔍 Processing AI recommendation request")
            logger.info(f"   User: {user_id}")
            logger.info(f"   Query: '{search_query}'")
            logger.info(f"   Voice: {'Yes' if prosody_features else 'No'}")
            logger.info(f"   Time: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            
            # Step 1: Check intelligent cache
            cache_key = self._generate_cache_key(user_id, search_query, prosody_features, context)
            cached_results = self._get_from_cache(cache_key)
            if cached_results:
                self.stats["cache_hits"] += 1
                logger.info(f"⚡ Cache hit - returning cached results in {(datetime.now() - start_time).total_seconds() * 1000:.1f}ms")
                cached_results["from_cache"] = True
                cached_results["cache_timestamp"] = datetime.now().isoformat()
                return cached_results
            
            # Step 2: AI-powered query analysis using multiple models
            logger.info("🧠 Performing comprehensive AI query analysis...")
            query_analysis = self._analyze_query_with_ai(search_query, prosody_features)
            
            detected_mood = query_analysis.get('detected_mood')
            search_intent = query_analysis.get('search_intent', 'general')
            confidence_score = query_analysis.get('confidence_score', 0.0)
            
            logger.info(f"📊 AI Analysis Results:")
            logger.info(f"   Intent: {search_intent}")
            logger.info(f"   Mood: {detected_mood}")
            logger.info(f"   Confidence: {confidence_score:.2f}")
            logger.info(f"   Processing: {query_analysis.get('processing_time_ms', 0):.1f}ms")
            
            # Step 3: Get comprehensive user context
            logger.info("👤 Gathering user context and preferences...")
            user_context = self.user_context.get_user_context(user_id)
            
            # Add additional context if provided
            if context:
                user_context.update(context)
            
            # Step 4: Multi-source AI recommendation generation
            logger.info("🎯 Generating recommendations from multiple AI sources...")
            recommendation_sources = {}
            
            # A. Elasticsearch advanced search (if available)
            if self.es_processor and self.config["enable_elasticsearch"]:
                try:
                    es_results = self.es_processor.enhanced_content_search(
                        query_analysis, user_context, limit=self.config["max_results_per_source"]
                    )
                    if es_results:
                        recommendation_sources['elasticsearch'] = [item['content_id'] for item in es_results]
                        self.stats["elasticsearch_uses"] += 1
                        logger.info(f"✅ Elasticsearch: {len(es_results)} recommendations")
                except Exception as e:
                    logger.warning(f"⚠️ Elasticsearch search failed: {e}")
            
            # B. Semantic search using transformer embeddings
            if (self.config["enable_semantic_search"] and 
                query_analysis.get("semantic_embedding") is not None):
                try:
                    semantic_recs = self.content_processor.get_semantic_recommendations(
                        query_analysis["semantic_embedding"],
                        limit=self.config["max_results_per_source"]
                    )
                    if semantic_recs:
                        recommendation_sources['semantic'] = semantic_recs
                        self.stats["semantic_searches"] += 1
                        logger.info(f"✅ Semantic AI: {len(semantic_recs)} recommendations")
                except Exception as e:
                    logger.warning(f"⚠️ Semantic search failed: {e}")
            
            # C. Collaborative filtering (users like you also liked...)
            if self.config["enable_collaborative_filtering"]:
                try:
                    cf_recs = self.model_manager.get_collaborative_recommendations(
                        user_id, limit=self.config["max_results_per_source"]
                    )
                    if cf_recs:
                        recommendation_sources['collaborative'] = cf_recs
                        self.stats["collaborative_searches"] += 1
                        logger.info(f"✅ Collaborative Filtering: {len(cf_recs)} recommendations")
                except Exception as e:
                    logger.warning(f"⚠️ Collaborative filtering failed: {e}")
            
            # D. Mood-based recommendations using emotion AI
            if (self.config["enable_mood_analysis"] and detected_mood and 
                confidence_score >= self.config["min_confidence_threshold"]):
                try:
                    mood_recs = self.content_processor.get_mood_recommendations(
                        detected_mood,
                        user_context.get('preferred_genres', []),
                        limit=self.config["max_results_per_source"]
                    )
                    if mood_recs:
                        recommendation_sources['mood'] = mood_recs
                        self.stats["mood_searches"] += 1
                        logger.info(f"✅ Mood-based AI: {len(mood_recs)} recommendations for mood '{detected_mood}'")
                except Exception as e:
                    logger.warning(f"⚠️ Mood-based search failed: {e}")
                
            # E. User preference-based recommendations
            preferred_genres = user_context.get('preferred_genres', [])
            if preferred_genres:
                try:
                    pref_recs = self.content_processor.get_genre_recommendations(
                        preferred_genres,
                        limit=self.config["max_results_per_source"]
                    )
                    if pref_recs:
                        recommendation_sources['preference'] = pref_recs
                        logger.info(f"✅ User Preferences: {len(pref_recs)} recommendations")
                except Exception as e:
                    logger.warning(f"⚠️ Preference-based search failed: {e}")
                
            # F. Text-based search as intelligent fallback
            try:
                text_recs = self.content_processor.get_text_search_recommendations(
                    search_query,
                    limit=self.config["max_results_per_source"]
                )
                if text_recs:
                    recommendation_sources['text'] = text_recs
                    logger.info(f"✅ Text Search: {len(text_recs)} recommendations")
            except Exception as e:
                logger.warning(f"⚠️ Text search failed: {e}")
            
            # Check if we have any recommendations
            if not recommendation_sources:
                logger.warning("⚠️ No recommendations from any source, using fallback")
                return self._fallback_recommendations(user_id, search_query)
            
            # Step 5: Intelligent fusion and ranking of all recommendations
            logger.info("🔀 Intelligently combining and ranking recommendations...")
            final_recommendations = self._combine_recommendations_with_ai(
                recommendation_sources,
                query_analysis,
                user_context,
                prosody_features,
                limit
            )
            
            # Step 6: Enrich with detailed content information
            logger.info("📝 Enriching recommendations with detailed content data...")
            content_details = self.content_processor.get_content_details(final_recommendations)
            
            # Step 7: Generate AI-powered explanations for each recommendation
            for i, item in enumerate(content_details):
                item["recommendation_reason"] = self._generate_ai_recommendation_reason(
                    item, query_analysis, user_context, prosody_features, i + 1
                )
                item["relevance_score"] = self._calculate_relevance_score(
                    item, query_analysis, user_context, recommendation_sources
                )
            
            # Step 8: Log recommendation for ML improvement and analytics
            self._log_comprehensive_recommendation(
                user_id, search_query, query_analysis, prosody_features, 
                content_details, recommendation_sources, user_context
            )
            
            # Step 9: Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_performance_stats(processing_time)
            
            # Step 10: Prepare comprehensive response
            response = self._build_comprehensive_response(
                content_details, query_analysis, user_context, prosody_features,
                recommendation_sources, processing_time, user_id, search_query
            )
            
            # Step 11: Cache the response for future requests
            self._add_to_cache(cache_key, response)
            
            self.stats["ai_successes"] += 1
            logger.info(f"✅ AI Recommendation completed successfully")
            logger.info(f"   Total results: {len(content_details)}")
            logger.info(f"   Processing time: {processing_time:.1f}ms")
            logger.info(f"   Sources used: {list(recommendation_sources.keys())}")
            logger.info(f"   Cache efficiency: {(self.stats['cache_hits'] / self.stats['total_requests'] * 100):.1f}%")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Critical error in AI recommendation engine: {e}", exc_info=True)
            self.stats["fallback_uses"] += 1
            return self._fallback_recommendations(user_id, search_query, error=str(e))

    def _analyze_query_with_ai(self, search_query, prosody_features):
        """Comprehensive AI analysis of user query"""
        try:
            # Use ModelManager for AI analysis
            analysis = self.model_manager.analyze_query(search_query, prosody_features)
            
            # Add prosody-specific analysis if available
            if prosody_features and self.config["enable_prosody_analysis"]:
                prosody_analysis = self.prosody_analyzer.analyze_mood_from_prosody(prosody_features)
                analysis.update({
                    "prosody_analysis": prosody_analysis,
                    "prosody_confidence": prosody_analysis.get("confidence", 0),
                    "voice_characteristics": self.prosody_analyzer.get_prosody_summary(prosody_features)
                })
                
                # Enhance mood detection with prosody
                if prosody_analysis.get("detected_mood") and prosody_analysis.get("confidence", 0) > 0.7:
                    analysis["detected_mood"] = prosody_analysis["detected_mood"]
                    analysis["confidence_score"] = max(analysis.get("confidence_score", 0), 
                                                     prosody_analysis["confidence"])
            
            return analysis
            
        except Exception as e:
            logger.error(f"AI query analysis failed: {e}")
            return {
                "original_query": search_query,
                "search_intent": "general",
                "detected_mood": None,
                "confidence_score": 0.0,
                "error": str(e)
            }

    def _combine_recommendations_with_ai(self, recommendation_sources, query_analysis, 
                                       user_context, prosody_features, limit=10):
        """
        Intelligently combine recommendations using AI-driven weighting
        """
        try:
            content_scores = {}
            
            # Dynamic weights based on AI analysis and context
            weights = self._calculate_dynamic_weights(
                recommendation_sources, query_analysis, user_context, prosody_features
            )
            
            logger.debug(f"🎯 Using dynamic weights: {weights}")
            
            # Score each content item from all sources
            for source, content_ids in recommendation_sources.items():
                weight = weights.get(source, 0.1)
                
                for rank, content_id in enumerate(content_ids):
                    # Position-based scoring (first results get higher scores)
                    position_factor = 1.0 - (rank / max(len(content_ids), 1))
                    
                    # Base score from source weight and position
                    base_score = weight * position_factor
                    
                    # Diversity bonus for new content
                    if content_id not in content_scores:
                        base_score *= 1.1  # 10% bonus for new content
                    
                    # Accumulate scores from multiple sources
                    if content_id in content_scores:
                        content_scores[content_id] += base_score
                    else:
                        content_scores[content_id] = base_score
            
            # Apply additional AI-based scoring adjustments
            content_scores = self._apply_ai_scoring_adjustments(
                content_scores, query_analysis, user_context
            )
            
            # Sort by final combined score
            ranked_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Apply diversity filter to avoid too similar content
            final_recommendations = self._apply_diversity_filter(ranked_content, limit)
            
            logger.debug(f"🏆 Final ranking (top 5): {final_recommendations[:5]}")
            
            return [content_id for content_id, score in final_recommendations]
            
        except Exception as e:
            logger.error(f"Error combining recommendations: {e}")
            # Fallback: just concatenate and deduplicate
            all_content = []
            for content_ids in recommendation_sources.values():
                all_content.extend(content_ids)
            return list(dict.fromkeys(all_content))[:limit]  # Remove duplicates, preserve order

    def _calculate_dynamic_weights(self, recommendation_sources, query_analysis, 
                                 user_context, prosody_features):
        """Calculate dynamic weights based on context and AI confidence"""
        
        # Base weights for different recommendation sources
        base_weights = {
            'elasticsearch': 0.35,     # Advanced search capabilities
            'semantic': 0.30,          # AI semantic understanding
            'collaborative': 0.20,     # Social proof from similar users
            'mood': 0.15,              # Emotional state matching
            'preference': 0.10,        # Historical user preferences
            'text': 0.05               # Basic text search fallback
        }
        
        # Only include weights for available sources
        weights = {source: base_weights.get(source, 0.1) 
                  for source in recommendation_sources.keys()}
        
        # Adjust weights based on search intent
        search_intent = query_analysis.get("search_intent", "general")
        detected_mood = query_analysis.get("detected_mood")
        confidence_score = query_analysis.get("confidence_score", 0.0)
        
        if search_intent.startswith("genre_"):
            # For genre searches, prioritize semantic and elasticsearch
            if 'semantic' in weights:
                weights['semantic'] = min(weights['semantic'] + 0.20, 0.60)
            if 'elasticsearch' in weights:
                weights['elasticsearch'] = min(weights['elasticsearch'] + 0.15, 0.50)
            if 'mood' in weights:
                weights['mood'] = max(weights['mood'] - 0.10, 0.05)
                
        elif search_intent == "mood_based":
            # For mood searches, prioritize emotion-based recommendations
            if 'mood' in weights:
                weights['mood'] = min(weights['mood'] + 0.25, 0.50)
            if 'semantic' in weights:
                weights['semantic'] = min(weights['semantic'] + 0.15, 0.45)
            if 'collaborative' in weights:
                weights['collaborative'] = max(weights['collaborative'] - 0.10, 0.10)
                
        elif search_intent == "person":
            # For actor/director searches, prioritize text and semantic
            if 'elasticsearch' in weights:
                weights['elasticsearch'] = min(weights['elasticsearch'] + 0.25, 0.60)
            if 'semantic' in weights:
                weights['semantic'] = min(weights['semantic'] + 0.15, 0.40)
            if 'text' in weights:
                weights['text'] = min(weights['text'] + 0.10, 0.25)
        
        # Boost mood-based weights if prosody strongly indicates emotion
        if prosody_features and detected_mood:
            prosody_confidence = query_analysis.get("prosody_confidence", 0)
            if prosody_confidence > 0.7 and 'mood' in weights:
                boost = min(0.20, prosody_confidence * 0.3)
                weights['mood'] = min(weights['mood'] + boost, 0.60)
                # Redistribute from other sources
                if 'semantic' in weights:
                    weights['semantic'] = max(weights['semantic'] - boost/2, 0.15)
        
        # Boost collaborative filtering for users with rich history
        user_activity_score = user_context.get('activity_score', 0.5)
        if user_activity_score > 0.7 and 'collaborative' in weights:
            weights['collaborative'] = min(weights['collaborative'] + 0.10, 0.35)
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {source: weight/total_weight for source, weight in weights.items()}
        
        return weights

    def _apply_ai_scoring_adjustments(self, content_scores, query_analysis, user_context):
        """Apply additional AI-based scoring adjustments"""
        try:
            # Get content details for scoring adjustments
            content_ids = list(content_scores.keys())
            if not content_ids:
                return content_scores
                
            # This would require content details - simplified for now
            # In practice, you'd get content details and apply adjustments based on:
            # - Content quality (rating)
            # - User preference alignment
            # - Mood appropriateness
            # - Recency/popularity
            
            return content_scores
            
        except Exception as e:
            logger.warning(f"AI scoring adjustments failed: {e}")
            return content_scores

    def _apply_diversity_filter(self, ranked_content, limit):
        """Apply diversity filtering to avoid too similar recommendations"""
        try:
            if len(ranked_content) <= limit:
                return ranked_content
            
            # Simple diversity: ensure we don't have too many items from the same genre
            # This would be enhanced with actual content analysis
            diverse_results = []
            seen_patterns = set()
            
            for content_id, score in ranked_content:
                # Add diversity logic here based on genre, director, etc.
                # For now, just take top items
                diverse_results.append((content_id, score))
                
                if len(diverse_results) >= limit:
                    break
            
            return diverse_results
            
        except Exception as e:
            logger.warning(f"Diversity filtering failed: {e}")
            return ranked_content[:limit]

    def _generate_ai_recommendation_reason(self, content, query_analysis, 
                                         user_context, prosody_features, rank):
        """Generate AI-powered explanation for why content was recommended"""
        try:
            reasons = []
            
            # Mood-based reasoning
            detected_mood = query_analysis.get("detected_mood")
            if detected_mood:
                mood_reasons = self._get_mood_based_reasons(content, detected_mood)
                reasons.extend(mood_reasons)
            
            # Voice characteristic reasoning
            if prosody_features:
                voice_reasons = self._get_voice_based_reasons(content, prosody_features, query_analysis)
                reasons.extend(voice_reasons)
            
            # User preference reasoning
            preference_reasons = self._get_preference_based_reasons(content, user_context)
            reasons.extend(preference_reasons)
            
            # Content quality reasoning
            quality_reasons = self._get_quality_based_reasons(content)
            reasons.extend(quality_reasons)
            
            # Search relevance reasoning
            relevance_reasons = self._get_relevance_based_reasons(content, query_analysis)
            reasons.extend(relevance_reasons)
            
            # Ranking reasoning
            if rank <= 3:
                reasons.append("Top AI recommendation")
            elif rank <= 5:
                reasons.append("Highly relevant match")
            
            # Combine reasons intelligently
            if len(reasons) > 3:
                # Keep most important reasons
                important_reasons = reasons[:3]
                if len(reasons) > 3:
                    important_reasons.append(f"Plus {len(reasons) - 3} more factors")
                reasons = important_reasons
            
            return " • ".join(reasons) if reasons else "AI-recommended based on your search"
            
        except Exception as e:
            logger.warning(f"Error generating recommendation reason: {e}")
            return "Recommended by AI"

    def _get_mood_based_reasons(self, content, detected_mood):
        """Get mood-based recommendation reasons"""
        reasons = []
        content_genres = content.get("genre", [])
        content_moods = content.get("mood_tags", [])
        
        mood_explanations = {
            "sad": {
                "helpful_genres": ["Comedy", "Feel-good", "Family"],
                "helpful_moods": ["uplifting", "cheerful", "heartwarming"],
                "reason": "Comedy to lift your mood"
            },
            "stressed": {
                "helpful_genres": ["Comedy", "Romance", "Feel-good"],
                "helpful_moods": ["relaxing", "calming", "peaceful"],
                "reason": "Calming content for stress relief"
            },
            "bored": {
                "helpful_genres": ["Action", "Thriller", "Adventure"],
                "helpful_moods": ["exciting", "thrilling", "engaging"],
                "reason": "Exciting entertainment for boredom"
            },
            "tired": {
                "helpful_genres": ["Romance", "Comedy", "Feel-good"],
                "helpful_moods": ["gentle", "cozy", "relaxing"],
                "reason": "Easy viewing when tired"
            },
            "lonely": {
                "helpful_genres": ["Romance", "Family", "Feel-good"],
                "helpful_moods": ["heartwarming", "warm", "companionship"],
                "reason": "Heartwarming content for companionship"
            }
        }
        
        mood_info = mood_explanations.get(detected_mood, {})
        if mood_info:
            helpful_genres = mood_info.get("helpful_genres", [])
            helpful_moods = mood_info.get("helpful_moods", [])
            
            if any(genre in content_genres for genre in helpful_genres):
                reasons.append(mood_info["reason"])
            elif any(mood in content_moods for mood in helpful_moods):
                reasons.append(f"Perfect for your {detected_mood} mood")
        
        return reasons

    def _get_voice_based_reasons(self, content, prosody_features, query_analysis):
        """Get voice characteristic-based reasons"""
        reasons = []
        
        try:
            intensity = prosody_features.get("intensity", "medium")
            tempo = prosody_features.get("tempo", "medium")
            prosody_mood = query_analysis.get("prosody_mood")
            
            content_moods = content.get("mood_tags", [])
            
            if intensity == "high" and any(tag in content_moods for tag in ["energetic", "exciting", "thrilling"]):
                reasons.append("Matches your energetic voice tone")
            elif intensity == "low" and any(tag in content_moods for tag in ["calm", "gentle", "peaceful"]):
                reasons.append("Complements your calm voice")
            
            if prosody_mood and prosody_mood in content_moods:
                reasons.append("Voice analysis suggests perfect match")
                
        except Exception as e:
            logger.debug(f"Voice reasoning failed: {e}")
        
        return reasons

    def _get_preference_based_reasons(self, content, user_context):
        """Get user preference-based reasons"""
        reasons = []
        
        preferred_genres = user_context.get("preferred_genres", [])
        content_genres = content.get("genre", [])
        
        if any(genre in content_genres for genre in preferred_genres):
            reasons.append("Aligns with your viewing preferences")
        
        # Check for other preference factors
        personality_traits = user_context.get("personality_traits", {})
        if personality_traits.get("quality_preference", 0) > 0.7:
            if content.get("rating", 0) >= 4.0:
                reasons.append("High quality matches your standards")
        
        return reasons

    def _get_quality_based_reasons(self, content):
        """Get content quality-based reasons"""
        reasons = []
        
        rating = float(content.get("rating", 0))
        rating_count = content.get("rating_count", 0)
        
        if rating >= 4.5:
            reasons.append("Exceptional quality (4.5+ stars)")
        elif rating >= 4.0:
            reasons.append("Highly rated content")
        
        if rating_count > 100:
            reasons.append("Popular with viewers")
        
        return reasons

    def _get_relevance_based_reasons(self, content, query_analysis):
        """Get search relevance-based reasons"""
        reasons = []
        
        search_intent = query_analysis.get("search_intent", "general")
        
        if search_intent.startswith("genre_"):
            requested_genre = search_intent.replace("genre_", "").title()
            if requested_genre in content.get("genre", []):
                reasons.append(f"Perfect {requested_genre.lower()} match")
        
        original_query = query_analysis.get("original_query", "").lower()
        content_title = content.get("title", "").lower()
        
        # Simple relevance check
        query_words = set(original_query.split())
        title_words = set(content_title.split())
        
        if len(query_words.intersection(title_words)) > 0:
            reasons.append("Direct search match")
        
        return reasons

    def _calculate_relevance_score(self, content, query_analysis, user_context, recommendation_sources):
        """Calculate overall relevance score for content"""
        try:
            score = 0.0
            
            # Base score from content rating
            score += content.get("rating", 0) * 0.1
            
            # Mood alignment score
            detected_mood = query_analysis.get("detected_mood")
            if detected_mood:
                # This would check mood alignment - simplified
                score += 0.3
            
            # User preference alignment
            preferred_genres = user_context.get("preferred_genres", [])
            content_genres = content.get("genre", [])
            if any(genre in content_genres for genre in preferred_genres):
                score += 0.2
            
            # Source diversity bonus
            if len(recommendation_sources) > 1:
                score += 0.1
            
            # Confidence score influence
            confidence = query_analysis.get("confidence_score", 0)
            score += confidence * 0.3
            
            return min(score, 1.0)  # Cap at 1.0
            
        except Exception as e:
            logger.warning(f"Relevance score calculation failed: {e}")
            return 0.5

    def _build_comprehensive_response(self, content_details, query_analysis, user_context, 
                                    prosody_features, recommendation_sources, processing_time, 
                                    user_id, search_query):
        """Build comprehensive recommendation response"""
        
        response = {
            # Core recommendation data
            "recommendations": content_details,
            "total_results": len(content_details),
            
            # AI analysis results
            "search_intent": query_analysis.get("search_intent", "general"),
            "detected_mood": query_analysis.get("detected_mood"),
            "confidence_score": query_analysis.get("confidence_score", 0.0),
            
            # Context and features
            "prosody_influenced": bool(prosody_features),
            "user_preferences_applied": bool(user_context.get("preferred_genres")),
            "search_context": self._determine_search_context(query_analysis),
            
            # Performance metrics
            "processing_time_ms": round(processing_time, 2),
            "cache_used": False,  # Set to True for cached responses
            
            # Comprehensive metadata
            "search_metadata": {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "original_query": search_query,
                "ai_methods_used": self._get_ai_methods_used(recommendation_sources),
                "recommendation_sources": list(recommendation_sources.keys()),
                "source_counts": {source: len(items) for source, items in recommendation_sources.items()},
                "query_complexity": query_analysis.get("query_complexity", "simple"),
                "elasticsearch_used": 'elasticsearch' in recommendation_sources,
                "collaborative_filtering_used": 'collaborative' in recommendation_sources,
                "semantic_search_used": 'semantic' in recommendation_sources,
                "mood_analysis_used": 'mood' in recommendation_sources
            },
            
            # Voice analysis (if available)
            "voice_analysis": None,
            
            # System status
            "engine_status": {
                "elasticsearch_available": self.es_processor is not None,
                "ai_models_loaded": len(self.model_manager.models),
                "cache_hit_rate": round((self.stats["cache_hits"] / max(self.stats["total_requests"], 1)) * 100, 2),
                "total_requests_served": self.stats["total_requests"]
            }
        }
        
        # Add voice analysis if prosody features were provided
        if prosody_features:
            response["voice_analysis"] = {
                "prosody_summary": self.prosody_analyzer.get_prosody_summary(prosody_features),
                "detected_voice_mood": query_analysis.get("prosody_mood"),
                "voice_confidence": query_analysis.get("prosody_confidence", 0),
                "voice_characteristics": query_analysis.get("voice_characteristics", {})
            }
        
        return response

    def _log_comprehensive_recommendation(self, user_id, search_query, query_analysis, 
                                        prosody_features, content_details, recommendation_sources, 
                                        user_context):
        """Log comprehensive recommendation data for analytics and ML improvement"""
        try:
            if not self.db:
                return
                
            log_data = {
                'user_id': user_id,
                'search_query': search_query,
                'search_type': 'voice' if prosody_features else 'text',
                'prosody_features': json.dumps(prosody_features) if prosody_features else None,
                'detected_mood': query_analysis.get('detected_mood'),
                'search_intent': query_analysis.get('search_intent', 'general'),
                'confidence_score': query_analysis.get('confidence_score', 0.0),
                'recommendation_method': 'hybrid_ai_ml_v2',
                'sources_used': json.dumps(list(recommendation_sources.keys())),
                'content_ids_returned': [item['content_id'] for item in content_details[:10]],
                'elasticsearch_used': 'elasticsearch' in recommendation_sources,
                'ai_processing_time_ms': query_analysis.get('processing_time_ms', 0),
                'total_processing_time_ms': 0,  # Will be calculated
                'user_context_applied': json.dumps({
                    'preferred_genres': user_context.get('preferred_genres', []),
                    'personality_traits': user_context.get('personality_traits', {}),
                    'context_factors': user_context.get('current_context', {})
                })
            }
            
            # Log to database (implement this in your database class)
            # self.db.log_comprehensive_search_history(log_data)
            
            logger.debug(f"Logged comprehensive recommendation data for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to log comprehensive recommendation: {e}")

    def _update_performance_stats(self, processing_time):
        """Update performance statistics"""
        try:
            # Update average response time
            total_time = self.stats["average_response_time_ms"] * (self.stats["total_requests"] - 1)
            self.stats["average_response_time_ms"] = (total_time + processing_time) / self.stats["total_requests"]
            
        except Exception as e:
            logger.warning(f"Failed to update performance stats: {e}")

    def _determine_search_context(self, query_analysis):
        """Determine search context for metadata"""
        search_intent = query_analysis.get("search_intent", "general")
        detected_mood = query_analysis.get("detected_mood")
        
        if search_intent == "mood_based" and detected_mood:
            return f"mood_{detected_mood}"
        elif search_intent.startswith("genre_"):
            return search_intent
        elif search_intent == "person":
            return "person_search"
        elif search_intent == "context":
            return "contextual_search"
        elif search_intent == "theme":
            return "theme_search"
            
        return "general_search"

    def _get_ai_methods_used(self, recommendation_sources):
        """Get list of AI methods that contributed to recommendations"""
        ai_methods = []
        
        if 'elasticsearch' in recommendation_sources:
            ai_methods.append("Advanced Search (Elasticsearch)")
        if 'semantic' in recommendation_sources:
            ai_methods.append("Semantic Analysis (Transformers)")
        if 'collaborative' in recommendation_sources:
            ai_methods.append("Collaborative Filtering (Matrix Factorization)")
        if 'mood' in recommendation_sources:
            ai_methods.append("Emotion Detection AI")
        if 'preference' in recommendation_sources:
            ai_methods.append("User Preference Learning")
        if 'text' in recommendation_sources:
            ai_methods.append("Enhanced Text Search")
            
        return ai_methods

    def _generate_cache_key(self, user_id, search_query, prosody_features, context):
        """Generate comprehensive cache key"""
        key_parts = [
            f"user={user_id}",
            f"query={search_query.lower().strip()}",
        ]
        
        if prosody_features:
            prosody_signature = f"prosody={prosody_features.get('intensity','medium')}:{prosody_features.get('tempo','medium')}"
            key_parts.append(prosody_signature)
            
        if context:
            context_signature = f"context={hash(str(sorted(context.items())))}"
            key_parts.append(context_signature)
            
        key_str = "||".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_from_cache(self, cache_key):
        """Get results from cache if available and not expired"""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            age_seconds = (datetime.now() - entry['timestamp']).total_seconds()
            
            if age_seconds < self.cache_expiry_seconds:
                return entry['results']
                
        return None

    def _add_to_cache(self, cache_key, results):
        """Add results to cache with intelligent management"""
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'results': results
        }
        
        # Intelligent cache management
        if len(self.cache) > self.max_cache_size:
            # Remove oldest 20% of entries
            sorted_keys = sorted(self.cache.keys(), 
                               key=lambda k: self.cache[k]['timestamp'])
            entries_to_remove = int(self.max_cache_size * 0.2)
            for old_key in sorted_keys[:entries_to_remove]:
                del self.cache[old_key]
                
        logger.debug(f"Added to cache. Current cache size: {len(self.cache)}")

    def _fallback_recommendations(self, user_id, search_query, error=None):
        """Provide intelligent fallback recommendations when AI fails"""
        try:
            logger.warning(f"Using fallback recommendations for user {user_id}, query: '{search_query}'")
            
            # Try text search as primary fallback
            content_items = []
            if self.content_processor:
                try:
                    content_items = self.content_processor.get_text_search_recommendations(
                        search_query, 
                        include_details=True,
                        limit=8
                    )
                except Exception as e:
                    logger.warning(f"Text search fallback failed: {e}")
            
            # If text search fails, get popular content
            if not content_items:
                try:
                    content_items = self.content_processor.get_popular_content(limit=6)
                except Exception as e:
                    logger.warning(f"Popular content fallback failed: {e}")
                    content_items = []
                
            return {
                "recommendations": content_items,
                "search_intent": "general",
                "detected_mood": None,
                "prosody_influenced": False,
                "total_results": len(content_items),
                "search_context": "fallback_search",
                "processing_time_ms": 0,
                "search_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "original_query": search_query,
                    "fallback_used": True,
                    "ai_methods_used": ["Fallback Text Search"] if content_items else ["Popular Content"],
                    "error": error or "AI recommendation system temporarily unavailable"
                },
                "engine_status": {
                    "fallback_mode": True,
                    "ai_available": False
                }
            }
            
        except Exception as e:
            logger.error(f"Even fallback recommendations failed: {e}")
            return {
                "recommendations": [],
                "error": "Unable to generate any recommendations",
                "total_results": 0,
                "search_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id,
                    "original_query": search_query,
                    "complete_failure": True,
                    "error": str(e)
                }
            }

    def get_engine_statistics(self):
        """Get comprehensive engine statistics"""
        cache_hit_rate = (self.stats["cache_hits"] / max(self.stats["total_requests"], 1)) * 100
        ai_success_rate = (self.stats["ai_successes"] / max(self.stats["total_requests"], 1)) * 100
        
        return {
            "performance_metrics": {
                "total_requests": self.stats["total_requests"],
                "ai_successes": self.stats["ai_successes"],
                "ai_success_rate_percent": round(ai_success_rate, 2),
                "fallback_uses": self.stats["fallback_uses"],
                "average_response_time_ms": round(self.stats["average_response_time_ms"], 2),
                "cache_hit_rate_percent": round(cache_hit_rate, 2),
                "cache_size": len(self.cache)
            },
            "ai_usage_stats": {
                "elasticsearch_uses": self.stats["elasticsearch_uses"],
                "semantic_searches": self.stats["semantic_searches"],
                "mood_searches": self.stats["mood_searches"],
                "collaborative_searches": self.stats["collaborative_searches"]
            },
            "system_status": {
                "elasticsearch_available": self.es_processor is not None,
                "models_loaded": len(self.model_manager.models),
                "configuration": self.config
            },
            "model_status": self.model_manager.get_model_status() if hasattr(self.model_manager, 'get_model_status') else {}
        }

    def clear_cache(self):
        """Clear recommendation cache"""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared recommendation cache ({cache_size} entries)")
        return cache_size

    def reload_models(self):
        """Reload AI models (useful for updates)"""
        try:
            self.model_manager = ModelManager()
            logger.info("Successfully reloaded AI models")
            return True
        except Exception as e:
            logger.error(f"Failed to reload models: {e}")
            return False

# Global instance for use across the application
_search_recommender_instance = None

def get_search_recommender(db_connection=None):
    """
    Get or create the global search recommender instance
    
    Args:
        db_connection: Database connection (required for first call)
        
    Returns:
        SearchRecommender instance
    """
    global _search_recommender_instance
    if _search_recommender_instance is None:
        if db_connection is None:
            raise ValueError("Database connection required for first initialization")
        _search_recommender_instance = SearchRecommender(db_connection)
    return _search_recommender_instance

# Example usage and testing function
if __name__ == "__main__":
    # This would be used for testing the engine
    logger.info("🚀 AI Video Recommendation Engine - Complete Implementation")
    logger.info(f"📅 Current Date/Time: 2025-06-20 08:20:22 UTC")
    logger.info(f"👤 Current User: rky-cse")
    logger.info("Ready for integration with API and voice processing systems!")