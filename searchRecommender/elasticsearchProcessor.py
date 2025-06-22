import logging
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import ConnectionError, NotFoundError, RequestError
import json
from datetime import datetime
import numpy as np
import os

logger = logging.getLogger(__name__)

class ElasticsearchContentProcessor:
    """
    Advanced Elasticsearch integration for enhanced content search and analytics.
    
    Features:
    - Semantic vector search with dense vectors
    - Advanced full-text search with multi-field matching
    - Real-time search analytics and trending
    - Auto-complete and search suggestions
    - Weather and festival-aware search boosting
    - Performance monitoring and optimization
    """
    
    def __init__(self, es_host="localhost", es_port=9200):
        self.es_host = es_host
        self.es_port = es_port
        self.es = None
        
        # Index names
        self.indices = {
            'content': 'video_content_v2',
            'searches': 'user_searches_v2', 
            'analytics': 'search_analytics_v2',
            'suggestions': 'search_suggestions_v2'
        }
        
        # Search statistics
        self.stats = {
            "total_searches": 0,
            "successful_searches": 0,
            "failed_searches": 0,
            "vector_searches": 0,
            "text_searches": 0,
            "suggestion_requests": 0,
            "average_response_time_ms": 0
        }
        
        # Content boost mappings for environmental factors
        self.weather_boost_mapping = {
            "Rain": {"boost_moods": ["cozy", "romantic", "heartwarming"], "boost_factor": 1.5},
            "Clear": {"boost_moods": ["energetic", "adventurous", "upbeat"], "boost_factor": 1.3},
            "Clouds": {"boost_moods": ["calm", "peaceful", "thoughtful"], "boost_factor": 1.2},
            "Snow": {"boost_moods": ["cozy", "family", "warm"], "boost_factor": 1.4},
            "Thunderstorm": {"boost_moods": ["intense", "dramatic", "thrilling"], "boost_factor": 1.3}
        }
        
        self.festival_boost_mapping = {
            "diwali": {"boost_genres": ["Family", "Drama", "Feel-good"], "boost_factor": 2.0},
            "christmas": {"boost_genres": ["Family", "Holiday", "Feel-good"], "boost_factor": 2.0},
            "valentine": {"boost_genres": ["Romance", "Romantic-Comedy"], "boost_factor": 1.8},
            "halloween": {"boost_genres": ["Horror", "Thriller", "Mystery"], "boost_factor": 1.8},
            "new year": {"boost_genres": ["Comedy", "Feel-good", "Party"], "boost_factor": 1.6}
        }
        
        self._connect_elasticsearch()
        self._ensure_indices_exist()
        
        logger.info(f"ElasticsearchContentProcessor initialized at 2025-06-20 08:41:58 UTC")
        logger.info(f"Connected to: {es_host}:{es_port} by user: rky-cse")
    
    def _connect_elasticsearch(self):
        """Connect to Elasticsearch cluster with retry logic."""
        try:
            self.es = Elasticsearch([{
                'host': self.es_host,
                'port': self.es_port,
                'scheme': 'http',
                'timeout': 30,
                'max_retries': 3,
                'retry_on_timeout': True
            }])
            
            if self.es.ping():
                cluster_info = self.es.info()
                logger.info(f"✅ Connected to Elasticsearch {cluster_info.get('version', {}).get('number', 'unknown')}")
            else:
                logger.error("❌ Elasticsearch ping failed")
                self.es = None
                
        except Exception as e:
            logger.error(f"❌ Elasticsearch connection error: {e}")
            self.es = None
    
    def _ensure_indices_exist(self):
        """Ensure all required indices exist with proper mappings."""
        if not self.es:
            return
        
        try:
            # Content index mapping
            content_mapping = {
                "mappings": {
                    "properties": {
                        "content_id": {"type": "keyword"},
                        "title": {
                            "type": "text",
                            "analyzer": "english",
                            "fields": {
                                "keyword": {"type": "keyword"},
                                "suggest": {"type": "completion"}
                            }
                        },
                        "description": {"type": "text", "analyzer": "english"},
                        "genre": {"type": "keyword"},
                        "mood_tags": {"type": "keyword"},
                        "actors": {"type": "keyword"},
                        "director": {"type": "keyword"},
                        "rating": {"type": "float"},
                        "release_year": {"type": "integer"},
                        "duration": {"type": "integer"},
                        "content_embedding": {"type": "dense_vector", "dims": 384},
                        "popularity_score": {"type": "float"},
                        "view_count": {"type": "long"},
                        "indexed_at": {"type": "date"},
                        "weather_boost_score": {"type": "float"},
                        "festival_boost_score": {"type": "float"}
                    }
                }
            }
            
            # Create content index if it doesn't exist
            if not self.es.indices.exists(index=self.indices['content']):
                self.es.indices.create(index=self.indices['content'], body=content_mapping)
                logger.info(f"Created content index: {self.indices['content']}")
            
            # Search analytics mapping
            analytics_mapping = {
                "mappings": {
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "search_query": {"type": "text", "analyzer": "english"},
                        "search_type": {"type": "keyword"},
                        "detected_mood": {"type": "keyword"},
                        "results_count": {"type": "integer"},
                        "response_time_ms": {"type": "float"},
                        "timestamp": {"type": "date"},
                        "weather_condition": {"type": "keyword"},
                        "active_festivals": {"type": "keyword"},
                        "user_location": {"type": "geo_point"}
                    }
                }
            }
            
            if not self.es.indices.exists(index=self.indices['analytics']):
                self.es.indices.create(index=self.indices['analytics'], body=analytics_mapping)
                logger.info(f"Created analytics index: {self.indices['analytics']}")
            
            logger.info("✅ All Elasticsearch indices ready")
            
        except Exception as e:
            logger.error(f"❌ Failed to create indices: {e}")
    
    def enhanced_content_search(self, query_analysis, user_context, limit=10):
        """
        Enhanced content search using AI query analysis and environmental context.
        
        Args:
            query_analysis: AI analysis from ModelManager
            user_context: User context including environmental factors
            limit: Number of results to return
            
        Returns:
            List of content items with relevance scores
        """
        if not self.es:
            logger.warning("Elasticsearch not available")
            return []
        
        start_time = datetime.now()
        self.stats["total_searches"] += 1
        
        try:
            # Build sophisticated search query
            search_body = self._build_enhanced_search_query(query_analysis, user_context, limit)
            
            # Execute search
            response = self.es.search(
                index=self.indices['content'],
                body=search_body,
                timeout='10s'
            )
            
            # Process and enhance results
            results = self._process_search_results(response, query_analysis, user_context)
            
            # Log search analytics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._log_search_analytics(query_analysis, user_context, len(results), processing_time)
            
            # Update statistics
            self.stats["successful_searches"] += 1
            self._update_response_time_stats(processing_time)
            
            if query_analysis.get("semantic_embedding") is not None:
                self.stats["vector_searches"] += 1
            else:
                self.stats["text_searches"] += 1
            
            logger.debug(f"✅ ES search completed: {len(results)} results in {processing_time:.1f}ms")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Elasticsearch search failed: {e}")
            self.stats["failed_searches"] += 1
            return []
    
    def _build_enhanced_search_query(self, query_analysis, user_context, limit):
        """Build sophisticated Elasticsearch query with AI enhancements."""
        
        original_query = query_analysis.get('original_query', '')
        detected_mood = query_analysis.get('detected_mood')
        search_intent = query_analysis.get('search_intent', 'general')
        semantic_embedding = query_analysis.get('semantic_embedding')
        confidence_score = query_analysis.get('confidence_score', 0.0)
        
        # Get environmental context
        env_context = user_context.get('current_environment', {})
        weather = env_context.get('weather', {})
        festivals = env_context.get('festivals_today', [])
        
        # Base query structure
        query = {
            "size": limit,
            "track_scores": True,
            "_source": [
                "content_id", "title", "description", "genre", "mood_tags", 
                "rating", "actors", "director", "duration", "popularity_score"
            ],
            "query": {
                "function_score": {
                    "query": {"bool": {"should": [], "must": [], "filter": [], "must_not": []}},
                    "functions": [],
                    "score_mode": "sum",
                    "boost_mode": "multiply"
                }
            }
        }
        
        # 1. Semantic vector search (highest priority if available)
        if semantic_embedding is not None and len(semantic_embedding) == 384:
            query["query"]["function_score"]["query"]["bool"]["should"].append({
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'content_embedding') + 1.0",
                        "params": {"query_vector": semantic_embedding.tolist()}
                    },
                    "boost": 3.0  # High boost for semantic similarity
                }
            })
        
        # 2. Multi-field text search
        if original_query:
            query["query"]["function_score"]["query"]["bool"]["should"].append({
                "multi_match": {
                    "query": original_query,
                    "fields": [
                        "title^4",           # Title most important
                        "description^2",     # Description second
                        "actors^2",          # Actors important
                        "director^2",        # Director important
                        "genre^1.5",         # Genre relevant
                        "mood_tags^1.5"      # Mood tags relevant
                    ],
                    "type": "best_fields",
                    "fuzziness": "AUTO",
                    "boost": 2.0
                }
            })
        
        # 3. Mood-based content matching
        if detected_mood and confidence_score > 0.3:
            mood_boost_map = {
                'sad': ['uplifting', 'cheerful', 'heartwarming', 'inspiring', 'hopeful'],
                'stressed': ['relaxing', 'calming', 'peaceful', 'gentle', 'soothing'],
                'bored': ['exciting', 'thrilling', 'engaging', 'fun', 'energetic'],
                'tired': ['gentle', 'cozy', 'easy-watching', 'relaxing', 'comfort'],
                'happy': ['fun', 'energetic', 'exciting', 'upbeat', 'cheerful'],
                'lonely': ['heartwarming', 'romantic', 'feel-good', 'warm', 'companionship'],
                'angry': ['cathartic', 'intense', 'powerful', 'justice', 'action'],
                'anxious': ['calming', 'reassuring', 'gentle', 'positive', 'peaceful']
            }
            
            boost_moods = mood_boost_map.get(detected_mood, [])
            if boost_moods:
                mood_boost = 2.0 * confidence_score  # Scale boost by confidence
                query["query"]["function_score"]["query"]["bool"]["should"].append({
                    "terms": {
                        "mood_tags": boost_moods,
                        "boost": mood_boost
                    }
                })
        
        # 4. User preference boosting
        preferred_genres = user_context.get('preferred_genres', [])
        if preferred_genres:
            query["query"]["function_score"]["query"]["bool"]["should"].append({
                "terms": {
                    "genre": preferred_genres,
                    "boost": 1.5
                }
            })
        
        # 5. Search intent specific handling
        if search_intent.startswith('genre_'):
            specific_genre = search_intent.replace('genre_', '').replace('_', ' ').title()
            query["query"]["function_score"]["query"]["bool"]["must"].append({
                "term": {"genre": specific_genre}
            })
        elif search_intent == 'person':
            # Boost actor/director matches for person searches
            if original_query:
                query["query"]["function_score"]["query"]["bool"]["should"].append({
                    "multi_match": {
                        "query": original_query,
                        "fields": ["actors^3", "director^3"],
                        "boost": 2.5
                    }
                })
        
        # 6. Environmental context boosting
        # Weather-based boosting
        weather_condition = weather.get('condition')
        if weather_condition and weather_condition in self.weather_boost_mapping:
            weather_boost_data = self.weather_boost_mapping[weather_condition]
            boost_moods = weather_boost_data['boost_moods']
            boost_factor = weather_boost_data['boost_factor']
            
            query["query"]["function_score"]["functions"].append({
                "filter": {"terms": {"mood_tags": boost_moods}},
                "weight": boost_factor
            })
        
        # Festival-based boosting
        if festivals:
            for festival in festivals:
                festival_name = festival.get('name', '').lower()
                for festival_key, boost_data in self.festival_boost_mapping.items():
                    if festival_key in festival_name:
                        boost_genres = boost_data['boost_genres']
                        boost_factor = boost_data['boost_factor']
                        
                        query["query"]["function_score"]["functions"].append({
                            "filter": {"terms": {"genre": boost_genres}},
                            "weight": boost_factor
                        })
                        break
        
        # 7. Quality and popularity boosting
        query["query"]["function_score"]["functions"].extend([
            {
                "field_value_factor": {
                    "field": "rating",
                    "factor": 0.2,
                    "modifier": "log1p",
                    "missing": 1
                }
            },
            {
                "field_value_factor": {
                    "field": "popularity_score",
                    "factor": 0.1,
                    "modifier": "log1p",
                    "missing": 1
                }
            }
        ])
        
        # 8. Avoid negative content for certain moods
        if detected_mood in ['sad', 'stressed', 'anxious']:
            negative_moods = ['dark', 'depressing', 'intense', 'scary', 'tragic']
            query["query"]["function_score"]["query"]["bool"]["must_not"].append({
                "terms": {"mood_tags": negative_moods}
            })
        
        # 9. Temporal boosting (recency)
        current_year = datetime.now().year
        query["query"]["function_score"]["functions"].append({
            "gauss": {
                "release_year": {
                    "origin": current_year,
                    "scale": "5y",
                    "decay": 0.8
                }
            },
            "weight": 0.1
        })
        
        # 10. Sorting strategy
        query["sort"] = [
            {"_score": {"order": "desc"}},
            {"rating": {"order": "desc"}},
            {"popularity_score": {"order": "desc"}}
        ]
        
        return query
    
    def _process_search_results(self, response, query_analysis, user_context):
        """Process and enhance Elasticsearch search results."""
        try:
            results = []
            
            for hit in response['hits']['hits']:
                source = hit['_source']
                
                result = {
                    'content_id': source.get('content_id'),
                    'title': source.get('title'),
                    'description': source.get('description'),
                    'genre': source.get('genre', []),
                    'mood_tags': source.get('mood_tags', []),
                    'rating': source.get('rating', 0),
                    'actors': source.get('actors', []),
                    'director': source.get('director', []),
                    'duration': source.get('duration', 0),
                    'popularity_score': source.get('popularity_score', 0),
                    
                    # Search metadata
                    'elasticsearch_score': hit['_score'],
                    'search_rank': len(results) + 1,
                    'match_explanation': self._generate_match_explanation(hit, query_analysis, user_context),
                    'recommendation_reason': self._generate_recommendation_reason(source, query_analysis, user_context)
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing search results: {e}")
            return []
    
    def _generate_match_explanation(self, hit, query_analysis, user_context):
        """Generate explanation for why content matched the search."""
        try:
            explanations = []
            
            # Semantic similarity
            if query_analysis.get('semantic_embedding') is not None:
                explanations.append("AI semantic understanding")
            
            # Text match
            if query_analysis.get('original_query'):
                explanations.append("Text relevance match")
            
            # Mood match
            detected_mood = query_analysis.get('detected_mood')
            if detected_mood:
                explanations.append(f"Mood-appropriate for {detected_mood}")
            
            # Environmental factors
            env_context = user_context.get('current_environment', {})
            if env_context.get('weather', {}).get('condition'):
                explanations.append("Weather-influenced recommendation")
            
            if env_context.get('festivals_today'):
                explanations.append("Festival-appropriate content")
            
            # User preferences
            if user_context.get('preferred_genres'):
                explanations.append("Matches your preferences")
            
            return explanations[:3]  # Limit to top 3 explanations
            
        except Exception:
            return ["Recommended by AI"]
    
    def _generate_recommendation_reason(self, content, query_analysis, user_context):
        """Generate human-readable recommendation reason."""
        try:
            reasons = []
            
            # Mood-based reasoning
            detected_mood = query_analysis.get('detected_mood')
            if detected_mood:
                mood_content_map = {
                    'sad': 'uplifting content to improve your mood',
                    'stressed': 'relaxing content for stress relief',
                    'bored': 'exciting entertainment to engage you',
                    'tired': 'easy-watching content for low energy',
                    'happy': 'fun content matching your good mood',
                    'lonely': 'heartwarming content for companionship'
                }
                reason = mood_content_map.get(detected_mood)
                if reason:
                    reasons.append(reason)
            
            # Quality reasoning
            rating = content.get('rating', 0)
            if rating >= 4.5:
                reasons.append('exceptional quality (4.5+ stars)')
            elif rating >= 4.0:
                reasons.append('highly rated content')
            
            # Environmental reasoning
            env_context = user_context.get('current_environment', {})
            weather = env_context.get('weather', {})
            if weather.get('condition') == 'Rain':
                reasons.append('perfect for a rainy day')
            elif weather.get('condition') == 'Clear':
                reasons.append('bright content for a clear day')
            
            # Festival reasoning
            festivals = env_context.get('festivals_today', [])
            if festivals:
                reasons.append(f'appropriate for {festivals[0].get("name", "festival")}')
            
            # Preference reasoning
            content_genres = set(content.get('genre', []))
            preferred_genres = set(user_context.get('preferred_genres', []))
            if content_genres & preferred_genres:
                reasons.append('matches your genre preferences')
            
            return ' • '.join(reasons[:2]) if reasons else 'AI-recommended based on your search'
            
        except Exception:
            return 'Recommended by AI'
    
    def get_search_suggestions(self, partial_query, limit=5):
        """Get auto-complete search suggestions."""
        if not self.es or not partial_query:
            return []
        
        try:
            self.stats["suggestion_requests"] += 1
            
            # Use completion suggester for fast autocomplete
            suggest_body = {
                "suggest": {
                    "title_suggest": {
                        "prefix": partial_query.lower(),
                        "completion": {
                            "field": "title.suggest",
                            "size": limit,
                            "skip_duplicates": True
                        }
                    }
                }
            }
            
            response = self.es.search(
                index=self.indices['content'],
                body=suggest_body
            )
            
            suggestions = []
            for option in response.get('suggest', {}).get('title_suggest', [{}])[0].get('options', []):
                suggestions.append({
                    'text': option['text'],
                    'score': option['_score']
                })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Search suggestions failed: {e}")
            return []
    
    def get_trending_searches(self, timeframe='24h', limit=10):
        """Get trending search queries."""
        if not self.es:
            return []
        
        try:
            # Query recent searches for trending analysis
            trending_body = {
                "size": 0,
                "query": {
                    "range": {
                        "timestamp": {
                            "gte": f"now-{timeframe}"
                        }
                    }
                },
                "aggs": {
                    "trending_queries": {
                        "terms": {
                            "field": "search_query.keyword",
                            "size": limit,
                            "order": {"_count": "desc"}
                        }
                    }
                }
            }
            
            response = self.es.search(
                index=self.indices['analytics'],
                body=trending_body
            )
            
            trending = []
            for bucket in response.get('aggregations', {}).get('trending_queries', {}).get('buckets', []):
                trending.append({
                    'query': bucket['key'],
                    'count': bucket['doc_count']
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Trending searches failed: {e}")
            return []
    
    def _log_search_analytics(self, query_analysis, user_context, results_count, response_time):
        """Log search analytics for insights and optimization."""
        if not self.es:
            return
        
        try:
            env_context = user_context.get('current_environment', {})
            
            analytics_doc = {
                'user_id': user_context.get('user_id', 'anonymous'),
                'search_query': query_analysis.get('original_query', ''),
                'search_type': 'voice' if query_analysis.get('prosody_features') else 'text',
                'detected_mood': query_analysis.get('detected_mood'),
                'search_intent': query_analysis.get('search_intent'),
                'confidence_score': query_analysis.get('confidence_score', 0.0),
                'results_count': results_count,
                'response_time_ms': response_time,
                'timestamp': datetime.now().isoformat(),
                'weather_condition': env_context.get('weather', {}).get('condition'),
                'active_festivals': [f.get('name') for f in env_context.get('festivals_today', [])],
                'is_weekend': env_context.get('is_weekend', False),
                'semantic_search_used': query_analysis.get('semantic_embedding') is not None,
                'elasticsearch_used': True
            }
            
            self.es.index(
                index=self.indices['analytics'],
                body=analytics_doc
            )
            
        except Exception as e:
            logger.debug(f"Analytics logging failed: {e}")
    
    def _update_response_time_stats(self, response_time):
        """Update average response time statistics."""
        try:
            total_time = self.stats["average_response_time_ms"] * (self.stats["successful_searches"] - 1)
            self.stats["average_response_time_ms"] = (total_time + response_time) / self.stats["successful_searches"]
        except:
            pass
    
    def index_content(self, content_data):
        """Index content in Elasticsearch."""
        if not self.es or not content_data:
            return False
        
        try:
            content_doc = {
                'content_id': content_data.get('content_id'),
                'title': content_data.get('title'),
                'description': content_data.get('description'),
                'genre': content_data.get('genre', []),
                'mood_tags': content_data.get('mood_tags', []),
                'actors': content_data.get('actors', []),
                'director': content_data.get('director', []),
                'rating': content_data.get('rating', 0),
                'release_year': content_data.get('release_year'),
                'duration': content_data.get('duration', 0),
                'content_embedding': content_data.get('content_embedding'),
                'popularity_score': content_data.get('popularity_score', 0),
                'view_count': content_data.get('view_count', 0),
                'indexed_at': datetime.now().isoformat()
            }
            
            self.es.index(
                index=self.indices['content'],
                id=content_data.get('content_id'),
                body=content_doc
            )
            
            logger.debug(f"Indexed content: {content_data.get('content_id')}")
            return True
            
        except Exception as e:
            logger.error(f"Content indexing failed: {e}")
            return False
    
    def get_elasticsearch_statistics(self):
        """Get comprehensive Elasticsearch statistics."""
        try:
            stats = self.stats.copy()
            
            if self.es:
                # Add cluster stats
                cluster_stats = self.es.cluster.stats()
                index_stats = {}
                
                for index_name in self.indices.values():
                    try:
                        if self.es.indices.exists(index=index_name):
                            index_info = self.es.indices.stats(index=index_name)
                            index_stats[index_name] = {
                                'document_count': index_info['_all']['total']['docs']['count'],
                                'size_bytes': index_info['_all']['total']['store']['size_in_bytes']
                            }
                    except:
                        pass
                
                stats.update({
                    'cluster_status': cluster_stats.get('status', 'unknown'),
                    'total_nodes': cluster_stats.get('nodes', {}).get('count', {}).get('total', 0),
                    'index_statistics': index_stats,
                    'elasticsearch_available': True
                })
            else:
                stats['elasticsearch_available'] = False
            
            # Calculate rates
            if stats["total_searches"] > 0:
                stats["success_rate_percent"] = round((stats["successful_searches"] / stats["total_searches"]) * 100, 2)
                stats["vector_search_percent"] = round((stats["vector_searches"] / stats["total_searches"]) * 100, 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics collection failed: {e}")
            return self.stats
    
    def health_check(self):
        """Perform Elasticsearch health check."""
        try:
            if not self.es:
                return {"status": "disconnected", "error": "No Elasticsearch connection"}
            
            # Basic ping
            if not self.es.ping():
                return {"status": "unreachable", "error": "Elasticsearch ping failed"}
            
            # Cluster health
            health = self.es.cluster.health()
            
            # Index status
            indices_status = {}
            for name, index in self.indices.items():
                try:
                    if self.es.indices.exists(index=index):
                        indices_status[name] = "exists"
                    else:
                        indices_status[name] = "missing"
                except:
                    indices_status[name] = "error"
            
            return {
                "status": "healthy",
                "cluster_status": health.get('status', 'unknown'),
                "cluster_name": health.get('cluster_name', 'unknown'),
                "number_of_nodes": health.get('number_of_nodes', 0),
                "indices_status": indices_status,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}