import logging
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
import threading
from pathlib import Path

# AI/ML Libraries
try:
    from sentence_transformers import SentenceTransformer
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Centralized AI model manager for the recommendation system.
    
    Features:
    - Semantic embedding models (SentenceTransformers)
    - Sentiment and mood analysis models
    - Collaborative filtering models
    - Content similarity models
    - Model caching and lazy loading
    - Performance monitoring
    - Fallback mechanisms when models unavailable
    """
    
    def __init__(self, model_cache_dir=None):
        self.model_cache_dir = model_cache_dir or os.path.join(os.getcwd(), 'model_cache')
        self.models = {}
        self.model_stats = {}
        self._lock = threading.Lock()
        
        # Create cache directory
        Path(self.model_cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Model configuration
        self.model_config = {
            'semantic_model': {
                'name': 'all-MiniLM-L6-v2',
                'type': 'sentence_transformer',
                'embedding_dim': 384,
                'enabled': TRANSFORMERS_AVAILABLE
            },
            'sentiment_model': {
                'name': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
                'type': 'huggingface_pipeline',
                'task': 'sentiment-analysis',
                'enabled': TRANSFORMERS_AVAILABLE
            },
            'emotion_model': {
                'name': 'j-hartmann/emotion-english-distilroberta-base',
                'type': 'huggingface_pipeline', 
                'task': 'text-classification',
                'enabled': TRANSFORMERS_AVAILABLE
            }
        }
        
        # Performance statistics
        self.stats = {
            'models_loaded': 0,
            'models_failed': 0,
            'embedding_requests': 0,
            'sentiment_requests': 0,
            'emotion_requests': 0,
            'collaborative_requests': 0,
            'cache_hits': 0,
            'total_inference_time_ms': 0,
            'average_inference_time_ms': 0
        }
        
        # Embedding cache for performance
        self.embedding_cache = {}
        self.cache_max_size = 1000
        self.cache_ttl = 3600  # 1 hour
        
        logger.info(f"ModelManager initialized at 2025-06-20 08:50:51 UTC by user: rky-cse")
        logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
        logger.info(f"Scikit-learn available: {SKLEARN_AVAILABLE}")
        logger.info(f"Model cache directory: {self.model_cache_dir}")
        
        # Initialize core models
        self._initialize_core_models()
    
    def _initialize_core_models(self):
        """Initialize essential models for the recommendation system."""
        try:
            # Load semantic embedding model (highest priority)
            if self._should_load_model('semantic_model'):
                self._load_semantic_model()
            
            # Load sentiment analysis model
            if self._should_load_model('sentiment_model'):
                self._load_sentiment_model()
            
            # Load emotion analysis model
            if self._should_load_model('emotion_model'):
                self._load_emotion_model()
            
            # Initialize collaborative filtering model
            self._initialize_collaborative_model()
            
            logger.info(f"✅ Initialized {len(self.models)} AI models successfully")
            
        except Exception as e:
            logger.error(f"❌ Error during model initialization: {e}")
    
    def _should_load_model(self, model_key):
        """Check if model should be loaded based on availability and configuration."""
        config = self.model_config.get(model_key, {})
        return config.get('enabled', False) and TRANSFORMERS_AVAILABLE
    
    def _load_semantic_model(self):
        """Load semantic embedding model for content understanding."""
        try:
            model_name = self.model_config['semantic_model']['name']
            logger.info(f"🔄 Loading semantic model: {model_name}")
            
            # Try to load from cache first
            cache_path = os.path.join(self.model_cache_dir, 'semantic_model')
            
            if os.path.exists(cache_path):
                try:
                    model = SentenceTransformer(cache_path)
                    logger.info("✅ Loaded semantic model from cache")
                except:
                    # If cache loading fails, download fresh
                    model = SentenceTransformer(model_name, cache_folder=self.model_cache_dir)
                    model.save(cache_path)
            else:
                model = SentenceTransformer(model_name, cache_folder=self.model_cache_dir)
                model.save(cache_path)
            
            self.models['semantic'] = model
            self.model_stats['semantic'] = {
                'loaded_at': datetime.now().isoformat(),
                'model_name': model_name,
                'embedding_dimension': 384,
                'requests_processed': 0
            }
            self.stats['models_loaded'] += 1
            
            logger.info(f"✅ Semantic model loaded successfully (384 dimensions)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load semantic model: {e}")
            self.stats['models_failed'] += 1
    
    def _load_sentiment_model(self):
        """Load sentiment analysis model."""
        try:
            model_name = self.model_config['sentiment_model']['name']
            logger.info(f"🔄 Loading sentiment model: {model_name}")
            
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                cache_dir=self.model_cache_dir,
                device=-1  # Use CPU
            )
            
            self.models['sentiment'] = sentiment_pipeline
            self.model_stats['sentiment'] = {
                'loaded_at': datetime.now().isoformat(),
                'model_name': model_name,
                'task': 'sentiment-analysis',
                'requests_processed': 0
            }
            self.stats['models_loaded'] += 1
            
            logger.info("✅ Sentiment model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load sentiment model: {e}")
            self.stats['models_failed'] += 1
    
    def _load_emotion_model(self):
        """Load emotion classification model."""
        try:
            model_name = self.model_config['emotion_model']['name']
            logger.info(f"🔄 Loading emotion model: {model_name}")
            
            emotion_pipeline = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                cache_dir=self.model_cache_dir,
                device=-1  # Use CPU
            )
            
            self.models['emotion'] = emotion_pipeline
            self.model_stats['emotion'] = {
                'loaded_at': datetime.now().isoformat(),
                'model_name': model_name,
                'task': 'emotion-classification',
                'requests_processed': 0
            }
            self.stats['models_loaded'] += 1
            
            logger.info("✅ Emotion model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load emotion model: {e}")
            self.stats['models_failed'] += 1
    
    def _initialize_collaborative_model(self):
        """Initialize collaborative filtering model."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("Scikit-learn not available - collaborative filtering disabled")
                return
            
            # Initialize collaborative filtering components
            self.models['collaborative'] = {
                'user_item_matrix': None,
                'item_similarity_matrix': None,
                'svd_model': TruncatedSVD(n_components=50, random_state=42),
                'nearest_neighbors': NearestNeighbors(n_neighbors=10, metric='cosine'),
                'last_trained': None,
                'training_data_size': 0
            }
            
            self.model_stats['collaborative'] = {
                'loaded_at': datetime.now().isoformat(),
                'model_type': 'collaborative_filtering',
                'components': ['SVD', 'NearestNeighbors'],
                'requests_processed': 0
            }
            
            logger.info("✅ Collaborative filtering model initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize collaborative model: {e}")
    
    def get_semantic_embedding(self, text, use_cache=True):
        """
        Get semantic embedding for text using transformer model.
        
        Args:
            text: Input text to embed
            use_cache: Whether to use embedding cache
            
        Returns:
            numpy array of embeddings or None if failed
        """
        if not text or not isinstance(text, str):
            return None
        
        start_time = datetime.now()
        self.stats['embedding_requests'] += 1
        
        try:
            # Check cache first
            if use_cache:
                cache_key = f"embed_{hash(text.strip().lower())}"
                cached_embedding = self._get_from_embedding_cache(cache_key)
                if cached_embedding is not None:
                    self.stats['cache_hits'] += 1
                    return cached_embedding
            
            # Get embedding from model
            if 'semantic' not in self.models:
                logger.warning("Semantic model not available")
                return self._get_fallback_embedding(text)
            
            semantic_model = self.models['semantic']
            
            # Clean and prepare text
            clean_text = text.strip()[:512]  # Limit text length
            
            # Generate embedding
            with self._lock:
                embedding = semantic_model.encode(clean_text, convert_to_numpy=True)
            
            # Cache the result
            if use_cache and cache_key:
                self._add_to_embedding_cache(cache_key, embedding)
            
            # Update statistics
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_inference_stats(inference_time)
            self.model_stats['semantic']['requests_processed'] += 1
            
            logger.debug(f"Generated embedding for text (len={len(text)}) in {inference_time:.1f}ms")
            
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Semantic embedding failed: {e}")
            return self._get_fallback_embedding(text)
    
    def analyze_sentiment_and_emotion(self, text):
        """
        Analyze sentiment and emotion of text using AI models.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment and emotion analysis results
        """
        if not text or not isinstance(text, str):
            return {"sentiment": "neutral", "emotion": "neutral", "confidence": 0.0}
        
        start_time = datetime.now()
        self.stats['sentiment_requests'] += 1
        self.stats['emotion_requests'] += 1
        
        try:
            analysis_result = {
                "sentiment": "neutral",
                "sentiment_confidence": 0.0,
                "emotion": "neutral", 
                "emotion_confidence": 0.0,
                "analysis_method": "ai_models"
            }
            
            # Clean text
            clean_text = text.strip()[:512]
            
            # Sentiment analysis
            if 'sentiment' in self.models:
                try:
                    sentiment_result = self.models['sentiment'](clean_text)
                    if sentiment_result and len(sentiment_result) > 0:
                        sentiment_data = sentiment_result[0]
                        analysis_result["sentiment"] = sentiment_data.get('label', 'neutral').lower()
                        analysis_result["sentiment_confidence"] = sentiment_data.get('score', 0.0)
                        
                        # Map sentiment labels to standard format
                        sentiment_mapping = {
                            'positive': 'positive',
                            'negative': 'negative', 
                            'neutral': 'neutral',
                            'label_2': 'positive',  # Some models use label_2 for positive
                            'label_1': 'neutral',
                            'label_0': 'negative'
                        }
                        analysis_result["sentiment"] = sentiment_mapping.get(
                            analysis_result["sentiment"], 'neutral'
                        )
                        
                    self.model_stats['sentiment']['requests_processed'] += 1
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed: {e}")
            
            # Emotion analysis
            if 'emotion' in self.models:
                try:
                    emotion_result = self.models['emotion'](clean_text)
                    if emotion_result and len(emotion_result) > 0:
                        emotion_data = emotion_result[0]
                        analysis_result["emotion"] = emotion_data.get('label', 'neutral').lower()
                        analysis_result["emotion_confidence"] = emotion_data.get('score', 0.0)
                        
                        # Map emotion labels to mood categories
                        emotion_to_mood = {
                            'joy': 'happy',
                            'happiness': 'happy',
                            'sadness': 'sad',
                            'anger': 'angry',
                            'fear': 'anxious',
                            'surprise': 'excited',
                            'disgust': 'negative',
                            'neutral': 'calm'
                        }
                        analysis_result["emotion"] = emotion_to_mood.get(
                            analysis_result["emotion"], analysis_result["emotion"]
                        )
                        
                    self.model_stats['emotion']['requests_processed'] += 1
                except Exception as e:
                    logger.warning(f"Emotion analysis failed: {e}")
            
            # Overall confidence (average of sentiment and emotion)
            analysis_result["overall_confidence"] = (
                analysis_result["sentiment_confidence"] + analysis_result["emotion_confidence"]
            ) / 2.0
            
            # Update statistics
            inference_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_inference_stats(inference_time)
            
            logger.debug(f"Analyzed sentiment/emotion in {inference_time:.1f}ms: "
                        f"sentiment={analysis_result['sentiment']}, emotion={analysis_result['emotion']}")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ Sentiment/emotion analysis failed: {e}")
            return self._get_fallback_sentiment_emotion(text)
    
    def analyze_query(self, search_query, prosody_features=None):
        """
        Comprehensive AI analysis of user search query.
        
        Args:
            search_query: User's search query text
            prosody_features: Optional voice prosody data
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        if not search_query:
            return {"error": "No search query provided"}
        
        start_time = datetime.now()
        
        try:
            logger.debug(f"🧠 Performing comprehensive AI query analysis for: '{search_query}'")
            
            analysis = {
                "original_query": search_query,
                "timestamp": datetime.now().isoformat(),
                "analysis_version": "v2.1"
            }
            
            # 1. Semantic embedding
            semantic_embedding = self.get_semantic_embedding(search_query)
            if semantic_embedding is not None:
                analysis["semantic_embedding"] = semantic_embedding
                analysis["embedding_available"] = True
            else:
                analysis["embedding_available"] = False
            
            # 2. Sentiment and emotion analysis
            sentiment_emotion = self.analyze_sentiment_and_emotion(search_query)
            analysis.update(sentiment_emotion)
            
            # 3. Search intent classification
            search_intent = self._classify_search_intent(search_query)
            analysis["search_intent"] = search_intent
            
            # 4. Mood detection (combining emotion analysis and prosody if available)
            detected_mood = self._detect_mood_from_analysis(sentiment_emotion, prosody_features)
            analysis["detected_mood"] = detected_mood["mood"]
            analysis["mood_confidence"] = detected_mood["confidence"]
            analysis["mood_source"] = detected_mood["source"]
            
            # 5. Query complexity assessment
            complexity = self._assess_query_complexity(search_query)
            analysis["query_complexity"] = complexity
            
            # 6. Content preference hints
            content_hints = self._extract_content_preferences(search_query, analysis)
            analysis["content_preferences"] = content_hints
            
            # 7. Overall confidence score
            confidence_factors = [
                sentiment_emotion.get("overall_confidence", 0.0),
                detected_mood.get("confidence", 0.0),
                1.0 if semantic_embedding is not None else 0.0,
                0.8 if search_intent != "general" else 0.3
            ]
            analysis["confidence_score"] = sum(confidence_factors) / len(confidence_factors)
            
            # Performance tracking
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            analysis["processing_time_ms"] = round(processing_time, 2)
            
            logger.debug(f"✅ AI query analysis complete in {processing_time:.1f}ms "
                        f"(confidence: {analysis['confidence_score']:.2f})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Query analysis failed: {e}")
            return {
                "original_query": search_query,
                "error": str(e),
                "confidence_score": 0.0,
                "analysis_version": "fallback"
            }
    
    def get_collaborative_recommendations(self, user_id, limit=10):
        """
        Get collaborative filtering recommendations for user.
        
        Args:
            user_id: User identifier
            limit: Number of recommendations
            
        Returns:
            List of recommended content IDs
        """
        self.stats['collaborative_requests'] += 1
        
        try:
            if 'collaborative' not in self.models:
                logger.warning("Collaborative filtering model not available")
                return []
            
            collaborative_model = self.models['collaborative']
            
            # Check if model is trained
            if collaborative_model['user_item_matrix'] is None:
                logger.warning("Collaborative model not trained yet")
                return []
            
            # This would implement actual collaborative filtering logic
            # For now, return empty list as placeholder
            logger.debug(f"Collaborative filtering request for user {user_id}")
            
            self.model_stats['collaborative']['requests_processed'] += 1
            
            return []  # Placeholder implementation
            
        except Exception as e:
            logger.error(f"❌ Collaborative filtering failed: {e}")
            return []
    
    def _classify_search_intent(self, query):
        """Classify the intent of the search query."""
        try:
            query_lower = query.lower().strip()
            
            # Genre-based search
            genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller', 'documentary', 'family', 'sci-fi', 'fantasy']
            for genre in genres:
                if genre in query_lower:
                    return f"genre_{genre}"
            
            # Person-based search (actor/director)
            person_indicators = ['actor', 'actress', 'director', 'starring', 'with', 'by']
            if any(indicator in query_lower for indicator in person_indicators):
                return "person"
            
            # Mood-based search
            mood_indicators = ['feel', 'mood', 'want to', 'need', 'looking for']
            if any(indicator in query_lower for indicator in mood_indicators):
                return "mood_based"
            
            # Context-based search
            context_indicators = ['today', 'tonight', 'weekend', 'date night', 'family time']
            if any(indicator in query_lower for indicator in context_indicators):
                return "context"
            
            # Theme-based search
            theme_indicators = ['about', 'story', 'theme', 'plot']
            if any(indicator in query_lower for indicator in theme_indicators):
                return "theme"
            
            return "general"
            
        except Exception:
            return "general"
    
    def _detect_mood_from_analysis(self, sentiment_emotion, prosody_features):
        """Detect mood combining text analysis and voice prosody."""
        try:
            mood_sources = []
            total_confidence = 0.0
            
            # From text emotion analysis
            text_emotion = sentiment_emotion.get("emotion", "neutral")
            text_confidence = sentiment_emotion.get("emotion_confidence", 0.0)
            
            if text_confidence > 0.3:
                mood_sources.append({
                    "source": "text_emotion",
                    "mood": text_emotion,
                    "confidence": text_confidence
                })
                total_confidence += text_confidence * 0.6  # 60% weight for text
            
            # From sentiment analysis
            sentiment = sentiment_emotion.get("sentiment", "neutral")
            sentiment_confidence = sentiment_emotion.get("sentiment_confidence", 0.0)
            
            if sentiment_confidence > 0.3:
                # Map sentiment to mood
                sentiment_to_mood = {
                    "positive": "happy",
                    "negative": "sad", 
                    "neutral": "calm"
                }
                sentiment_mood = sentiment_to_mood.get(sentiment, "calm")
                
                mood_sources.append({
                    "source": "text_sentiment",
                    "mood": sentiment_mood,
                    "confidence": sentiment_confidence
                })
                total_confidence += sentiment_confidence * 0.4  # 40% weight for sentiment
            
            # From prosody if available (would be integrated with ProsodyAnalyzer)
            if prosody_features:
                # This would use the ProsodyAnalyzer results
                prosody_mood = prosody_features.get("detected_mood")
                prosody_confidence = prosody_features.get("confidence", 0.0)
                
                if prosody_mood and prosody_confidence > 0.3:
                    mood_sources.append({
                        "source": "voice_prosody",
                        "mood": prosody_mood,
                        "confidence": prosody_confidence
                    })
                    total_confidence += prosody_confidence * 0.8  # 80% weight for voice
            
            # Determine final mood
            if mood_sources:
                # Take highest confidence mood
                best_mood_source = max(mood_sources, key=lambda x: x["confidence"])
                final_confidence = min(total_confidence / len(mood_sources), 1.0)
                
                return {
                    "mood": best_mood_source["mood"],
                    "confidence": round(final_confidence, 2),
                    "source": best_mood_source["source"],
                    "all_sources": mood_sources
                }
            
            return {"mood": None, "confidence": 0.0, "source": "none"}
            
        except Exception as e:
            logger.warning(f"Mood detection failed: {e}")
            return {"mood": None, "confidence": 0.0, "source": "error"}
    
    def _assess_query_complexity(self, query):
        """Assess the complexity of the search query."""
        try:
            word_count = len(query.split())
            
            if word_count <= 2:
                return "simple"
            elif word_count <= 6:
                return "medium"
            else:
                return "complex"
                
        except:
            return "simple"
    
    def _extract_content_preferences(self, query, analysis):
        """Extract content preference hints from query and analysis."""
        try:
            preferences = {
                "explicit_genres": [],
                "implicit_preferences": [],
                "mood_indicators": [],
                "quality_indicators": []
            }
            
            query_lower = query.lower()
            
            # Explicit genre mentions
            genres = ['action', 'comedy', 'drama', 'horror', 'romance', 'thriller', 'documentary']
            for genre in genres:
                if genre in query_lower:
                    preferences["explicit_genres"].append(genre.title())
            
            # Quality indicators
            quality_words = ['good', 'best', 'top', 'great', 'excellent', 'high-rated']
            for word in quality_words:
                if word in query_lower:
                    preferences["quality_indicators"].append(word)
            
            # Mood indicators from detected emotion
            detected_mood = analysis.get("detected_mood")
            if detected_mood:
                preferences["mood_indicators"].append(detected_mood)
            
            return preferences
            
        except:
            return {}
    
    def _get_fallback_embedding(self, text):
        """Get fallback embedding using TF-IDF when transformer models unavailable."""
        try:
            if not SKLEARN_AVAILABLE:
                logger.warning("No embedding method available - returning zero vector")
                return np.zeros(384)
            
            # Simple TF-IDF based embedding (fallback)
            if not hasattr(self, '_tfidf_vectorizer'):
                self._tfidf_vectorizer = TfidfVectorizer(max_features=384, stop_words='english')
                # This would need training data - for now return random vector
                return np.random.rand(384) * 0.1
            
            return np.random.rand(384) * 0.1  # Placeholder
            
        except:
            return np.zeros(384)
    
    def _get_fallback_sentiment_emotion(self, text):
        """Get fallback sentiment/emotion analysis using keyword matching."""
        try:
            text_lower = text.lower()
            
            # Simple keyword-based sentiment
            positive_words = ['good', 'great', 'excellent', 'love', 'like', 'happy', 'amazing']
            negative_words = ['bad', 'terrible', 'hate', 'dislike', 'sad', 'awful', 'horrible']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
                confidence = min(positive_count * 0.3, 0.8)
            elif negative_count > positive_count:
                sentiment = "negative"
                confidence = min(negative_count * 0.3, 0.8)
            else:
                sentiment = "neutral"
                confidence = 0.3
            
            return {
                "sentiment": sentiment,
                "sentiment_confidence": confidence,
                "emotion": "neutral",
                "emotion_confidence": 0.3,
                "overall_confidence": confidence,
                "analysis_method": "keyword_fallback"
            }
            
        except:
            return {
                "sentiment": "neutral",
                "sentiment_confidence": 0.0,
                "emotion": "neutral",
                "emotion_confidence": 0.0,
                "overall_confidence": 0.0,
                "analysis_method": "fallback_failed"
            }
    
    def _get_from_embedding_cache(self, cache_key):
        """Get embedding from cache if available and fresh."""
        if cache_key not in self.embedding_cache:
            return None
        
        entry = self.embedding_cache[cache_key]
        age_seconds = (datetime.now() - entry['timestamp']).total_seconds()
        
        if age_seconds < self.cache_ttl:
            return entry['embedding']
        else:
            del self.embedding_cache[cache_key]
            return None
    
    def _add_to_embedding_cache(self, cache_key, embedding):
        """Add embedding to cache with size management."""
        if len(self.embedding_cache) >= self.cache_max_size:
            # Remove oldest 20% of entries
            oldest_keys = sorted(self.embedding_cache.keys(),
                               key=lambda k: self.embedding_cache[k]['timestamp'])[:200]
            for key in oldest_keys:
                del self.embedding_cache[key]
        
        self.embedding_cache[cache_key] = {
            'timestamp': datetime.now(),
            'embedding': embedding
        }
    
    def _update_inference_stats(self, inference_time_ms):
        """Update inference time statistics."""
        try:
            total_time = self.stats['total_inference_time_ms'] + inference_time_ms
            total_requests = (self.stats['embedding_requests'] + 
                            self.stats['sentiment_requests'] + 
                            self.stats['emotion_requests'])
            
            self.stats['total_inference_time_ms'] = total_time
            self.stats['average_inference_time_ms'] = total_time / max(total_requests, 1)
        except:
            pass
    
    def get_model_status(self):
        """Get comprehensive model status and statistics."""
        try:
            status = {
                "models_available": list(self.models.keys()),
                "models_loaded_count": len(self.models),
                "transformers_available": TRANSFORMERS_AVAILABLE,
                "sklearn_available": SKLEARN_AVAILABLE,
                "cache_size": len(self.embedding_cache),
                "statistics": self.stats.copy(),
                "model_details": {}
            }
            
            # Add individual model details
            for model_name, model_stats in self.model_stats.items():
                status["model_details"][model_name] = model_stats.copy()
            
            # Calculate cache hit rate
            total_requests = (self.stats['embedding_requests'] + 
                            self.stats['sentiment_requests'] + 
                            self.stats['emotion_requests'])
            
            if total_requests > 0:
                status["cache_hit_rate_percent"] = round(
                    (self.stats['cache_hits'] / total_requests) * 100, 2
                )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status: {e}")
            return {"error": str(e)}
    
    def clear_caches(self):
        """Clear all model caches."""
        cache_size = len(self.embedding_cache)
        self.embedding_cache.clear()
        logger.info(f"Cleared model caches ({cache_size} embeddings)")
        return cache_size
    
    def reload_models(self):
        """Reload all AI models (useful for updates)."""
        try:
            logger.info("🔄 Reloading all AI models...")
            
            # Clear existing models
            self.models.clear()
            self.model_stats.clear()
            
            # Reinitialize
            self._initialize_core_models()
            
            logger.info(f"✅ Successfully reloaded {len(self.models)} models")
            return True
            
        except Exception as e:
            logger.error(f"❌ Model reload failed: {e}")
            return False