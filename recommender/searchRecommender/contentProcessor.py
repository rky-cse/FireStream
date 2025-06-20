import logging
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
import re

logger = logging.getLogger(__name__)

class ContentProcessor:
    """
    Advanced content processing for AI-powered video recommendations.
    
    Features:
    - Semantic content search using embeddings
    - Mood-based content filtering and matching
    - Genre-based recommendations with preferences
    - Text search with relevance scoring
    - Content quality assessment and ranking
    - Popular content trending analysis
    - Performance optimization with caching
    """
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        
        # Content processing statistics
        self.stats = {
            "semantic_searches": 0,
            "mood_searches": 0,
            "genre_searches": 0,
            "text_searches": 0,
            "content_details_fetched": 0,
            "cache_hits": 0,
            "total_processing_time_ms": 0
        }
        
        # Content cache for performance
        self.content_cache = {}
        self.cache_ttl = 900  # 15 minutes
        self.max_cache_size = 500
        
        # Mood to genre mapping for intelligent recommendations
        self.mood_genre_mapping = {
            'sad': {
                'boost_genres': ['Comedy', 'Feel-good', 'Family', 'Uplifting'],
                'avoid_genres': ['Drama', 'Tragedy', 'Dark'],
                'boost_moods': ['cheerful', 'heartwarming', 'inspiring', 'uplifting', 'hopeful']
            },
            'stressed': {
                'boost_genres': ['Comedy', 'Feel-good', 'Romance', 'Light-Drama'],
                'avoid_genres': ['Thriller', 'Horror', 'Intense'],
                'boost_moods': ['relaxing', 'calming', 'peaceful', 'gentle', 'soothing']
            },
            'bored': {
                'boost_genres': ['Action', 'Adventure', 'Thriller', 'Comedy'],
                'avoid_genres': ['Documentary', 'Slow'],
                'boost_moods': ['exciting', 'thrilling', 'engaging', 'fun', 'energetic']
            },
            'tired': {
                'boost_genres': ['Comedy', 'Feel-good', 'Light-Drama', 'Family'],
                'avoid_genres': ['Complex', 'Intense', 'Long'],
                'boost_moods': ['gentle', 'easy-watching', 'comfortable', 'relaxing']
            },
            'happy': {
                'boost_genres': ['Comedy', 'Adventure', 'Feel-good', 'Action'],
                'avoid_genres': ['Dark', 'Depressing'],
                'boost_moods': ['fun', 'energetic', 'exciting', 'upbeat', 'cheerful']
            },
            'lonely': {
                'boost_genres': ['Romance', 'Feel-good', 'Family', 'Heartwarming'],
                'avoid_genres': ['Dark', 'Isolation'],
                'boost_moods': ['heartwarming', 'romantic', 'warm', 'companionship', 'love']
            },
            'angry': {
                'boost_genres': ['Action', 'Thriller', 'Drama'],
                'avoid_genres': ['Romance', 'Light'],
                'boost_moods': ['cathartic', 'intense', 'powerful', 'justice']
            },
            'anxious': {
                'boost_genres': ['Comedy', 'Feel-good', 'Romance', 'Family'],
                'avoid_genres': ['Horror', 'Thriller', 'Intense'],
                'boost_moods': ['calming', 'reassuring', 'gentle', 'peaceful', 'positive']
            }
        }
        
        # Quality scoring weights
        self.quality_weights = {
            'rating': 0.35,
            'popularity_score': 0.25,
            'view_count': 0.20,
            'recency': 0.10,
            'completion_rate': 0.10
        }
        
        logger.info(f"ContentProcessor initialized at 2025-06-20 08:54:18 UTC by user: rky-cse")
        logger.info(f"Database available: {self.db is not None}")
    
    def get_semantic_recommendations(self, query_embedding, user_preferences=None, limit=20):
        """
        Get content recommendations using semantic similarity search.
        
        Args:
            query_embedding: Query embedding vector (384 dimensions)
            user_preferences: User preference context
            limit: Maximum number of recommendations
            
        Returns:
            List of content IDs ranked by semantic similarity
        """
        if query_embedding is None or not self.db:
            return []
        
        start_time = datetime.now()
        self.stats["semantic_searches"] += 1
        
        try:
            logger.debug(f"🔍 Performing semantic search with {len(query_embedding)}D embedding")
            
            # Convert embedding to list for database query
            embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
            
            with self.db.get_cursor() as (conn, cur):
                # Use PostgreSQL's vector similarity (assuming pgvector extension)
                # If pgvector not available, fall back to manual calculation
                try:
                    # Try pgvector cosine similarity
                    similarity_query = """
                        SELECT content_id, title, genre, mood_tags, rating, popularity_score,
                               (content_embedding <=> %s::vector) as similarity_distance
                        FROM content 
                        WHERE content_embedding IS NOT NULL
                        ORDER BY content_embedding <=> %s::vector
                        LIMIT %s
                    """
                    cur.execute(similarity_query, (embedding_list, embedding_list, limit * 2))
                    
                except Exception:
                    # Fallback to manual similarity calculation
                    logger.debug("pgvector not available, using manual similarity calculation")
                    return self._manual_semantic_search(embedding_list, user_preferences, limit)
                
                results = cur.fetchall()
                
                if not results:
                    logger.debug("No semantic results found")
                    return []
                
                # Process and rank results
                content_recommendations = []
                for row in results:
                    content_id = row[0]
                    similarity_score = 1.0 - row[6]  # Convert distance to similarity
                    
                    # Apply user preference boosting
                    boosted_score = self._apply_user_preference_boost(
                        row, similarity_score, user_preferences
                    )
                    
                    content_recommendations.append({
                        'content_id': content_id,
                        'similarity_score': round(similarity_score, 4),
                        'boosted_score': round(boosted_score, 4),
                        'title': row[1],
                        'genre': row[2],
                        'mood_tags': row[3]
                    })
                
                # Sort by boosted score and return content IDs
                content_recommendations.sort(key=lambda x: x['boosted_score'], reverse=True)
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_processing_stats(processing_time)
                
                logger.debug(f"✅ Semantic search completed: {len(content_recommendations)} results in {processing_time:.1f}ms")
                
                return [item['content_id'] for item in content_recommendations[:limit]]
                
        except Exception as e:
            logger.error(f"❌ Semantic search failed: {e}")
            return []
    
    def _manual_semantic_search(self, query_embedding, user_preferences, limit):
        """Manual semantic similarity calculation when pgvector unavailable."""
        try:
            with self.db.get_cursor() as (conn, cur):
                # Get all content with embeddings
                cur.execute("""
                    SELECT content_id, title, genre, mood_tags, rating, 
                           popularity_score, content_embedding
                    FROM content 
                    WHERE content_embedding IS NOT NULL
                    LIMIT 1000
                """)
                
                content_data = cur.fetchall()
                
                if not content_data:
                    return []
                
                # Calculate similarities manually
                similarities = []
                query_vec = np.array(query_embedding)
                
                for row in content_data:
                    try:
                        content_embedding = np.array(row[6])
                        
                        # Cosine similarity
                        similarity = np.dot(query_vec, content_embedding) / (
                            np.linalg.norm(query_vec) * np.linalg.norm(content_embedding)
                        )
                        
                        # Apply user preference boost
                        boosted_score = self._apply_user_preference_boost(
                            row, similarity, user_preferences
                        )
                        
                        similarities.append({
                            'content_id': row[0],
                            'similarity_score': similarity,
                            'boosted_score': boosted_score
                        })
                        
                    except Exception as e:
                        logger.debug(f"Similarity calculation failed for content {row[0]}: {e}")
                        continue
                
                # Sort and return top results
                similarities.sort(key=lambda x: x['boosted_score'], reverse=True)
                return [item['content_id'] for item in similarities[:limit]]
                
        except Exception as e:
            logger.error(f"Manual semantic search failed: {e}")
            return []
    
    def get_mood_recommendations(self, detected_mood, user_genres=None, limit=20):
        """
        Get content recommendations based on detected mood.
        
        Args:
            detected_mood: Detected user mood (e.g., 'sad', 'happy', 'stressed')
            user_genres: User's preferred genres
            limit: Maximum number of recommendations
            
        Returns:
            List of content IDs appropriate for the mood
        """
        if not detected_mood or not self.db:
            return []
        
        start_time = datetime.now()
        self.stats["mood_searches"] += 1
        
        try:
            logger.debug(f"🎭 Getting mood-based recommendations for mood: {detected_mood}")
            
            mood_config = self.mood_genre_mapping.get(detected_mood.lower(), {})
            boost_genres = mood_config.get('boost_genres', [])
            avoid_genres = mood_config.get('avoid_genres', [])
            boost_moods = mood_config.get('boost_moods', [])
            
            with self.db.get_cursor() as (conn, cur):
                # Build dynamic query based on mood requirements
                query_parts = []
                params = []
                
                # Base query
                base_query = """
                    SELECT content_id, title, genre, mood_tags, rating, 
                           popularity_score, view_count, release_year
                    FROM content
                    WHERE 1=1
                """
                
                # Boost content with appropriate genres
                if boost_genres:
                    genre_conditions = []
                    for genre in boost_genres:
                        genre_conditions.append("genre && %s")
                        params.append([genre])
                    
                    if genre_conditions:
                        query_parts.append(f"AND ({' OR '.join(genre_conditions)})")
                
                # Boost content with appropriate mood tags
                if boost_moods:
                    mood_conditions = []
                    for mood in boost_moods:
                        mood_conditions.append("mood_tags && %s")
                        params.append([mood])
                    
                    if mood_conditions:
                        if query_parts:  # If we already have genre conditions, use OR
                            query_parts.append(f"OR ({' OR '.join(mood_conditions)})")
                        else:
                            query_parts.append(f"AND ({' OR '.join(mood_conditions)})")
                
                # Avoid negative content for certain moods
                if avoid_genres:
                    avoid_conditions = []
                    for genre in avoid_genres:
                        avoid_conditions.append("NOT (genre && %s)")
                        params.append([genre])
                    
                    if avoid_conditions:
                        query_parts.append(f"AND ({' AND '.join(avoid_conditions)})")
                
                # Quality and popularity ordering
                order_by = """
                    ORDER BY 
                        CASE 
                            WHEN rating >= 4.0 THEN rating * 1.2
                            ELSE rating 
                        END DESC,
                        popularity_score DESC,
                        view_count DESC
                    LIMIT %s
                """
                params.append(limit * 2)  # Get extra for filtering
                
                # Execute query
                full_query = base_query + ' '.join(query_parts) + order_by
                cur.execute(full_query, params)
                
                results = cur.fetchall()
                
                if not results:
                    logger.debug(f"No mood-based results found for {detected_mood}")
                    return []
                
                # Score and rank results based on mood appropriateness
                scored_results = []
                for row in results:
                    content_id, title, genres, mood_tags, rating, popularity, view_count, release_year = row
                    
                    mood_score = self._calculate_mood_appropriateness_score(
                        genres or [], mood_tags or [], detected_mood, mood_config
                    )
                    
                    # Apply user genre preferences
                    genre_boost = 1.0
                    if user_genres and genres:
                        common_genres = set(user_genres) & set(genres)
                        genre_boost = 1.0 + (len(common_genres) * 0.2)
                    
                    final_score = mood_score * genre_boost
                    
                    scored_results.append({
                        'content_id': content_id,
                        'mood_score': round(mood_score, 3),
                        'final_score': round(final_score, 3),
                        'title': title
                    })
                
                # Sort by final score
                scored_results.sort(key=lambda x: x['final_score'], reverse=True)
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_processing_stats(processing_time)
                
                logger.debug(f"✅ Mood search completed: {len(scored_results)} results in {processing_time:.1f}ms")
                
                return [item['content_id'] for item in scored_results[:limit]]
                
        except Exception as e:
            logger.error(f"❌ Mood-based search failed: {e}")
            return []
    
    def get_genre_recommendations(self, preferred_genres, exclude_genres=None, limit=20):
        """
        Get content recommendations based on preferred genres.
        
        Args:
            preferred_genres: List of preferred genres
            exclude_genres: List of genres to exclude
            limit: Maximum number of recommendations
            
        Returns:
            List of content IDs matching genre preferences
        """
        if not preferred_genres or not self.db:
            return []
        
        start_time = datetime.now()
        self.stats["genre_searches"] += 1
        
        try:
            logger.debug(f"🎬 Getting genre recommendations for: {preferred_genres}")
            
            with self.db.get_cursor() as (conn, cur):
                # Build genre query
                genre_conditions = []
                params = []
                
                # Include preferred genres
                for genre in preferred_genres:
                    genre_conditions.append("genre && %s")
                    params.append([genre])
                
                where_clause = f"WHERE ({' OR '.join(genre_conditions)})"
                
                # Exclude unwanted genres
                if exclude_genres:
                    exclude_conditions = []
                    for genre in exclude_genres:
                        exclude_conditions.append("NOT (genre && %s)")
                        params.append([genre])
                    
                    where_clause += f" AND ({' AND '.join(exclude_conditions)})"
                
                query = f"""
                    SELECT content_id, title, genre, rating, popularity_score, 
                           view_count, release_year
                    FROM content
                    {where_clause}
                    ORDER BY 
                        rating DESC,
                        popularity_score DESC,
                        view_count DESC
                    LIMIT %s
                """
                params.append(limit)
                
                cur.execute(query, params)
                results = cur.fetchall()
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_processing_stats(processing_time)
                
                logger.debug(f"✅ Genre search completed: {len(results)} results in {processing_time:.1f}ms")
                
                return [row[0] for row in results]
                
        except Exception as e:
            logger.error(f"❌ Genre search failed: {e}")
            return []
    
    def get_text_search_recommendations(self, search_query, include_details=False, limit=20):
        """
        Get content recommendations using text search with relevance scoring.
        
        Args:
            search_query: Text search query
            include_details: Whether to include content details
            limit: Maximum number of recommendations
            
        Returns:
            List of content IDs or detailed content info
        """
        if not search_query or not self.db:
            return []
        
        start_time = datetime.now()
        self.stats["text_searches"] += 1
        
        try:
            logger.debug(f"📝 Performing text search for: '{search_query}'")
            
            # Clean and prepare search query
            clean_query = self._clean_search_query(search_query)
            search_terms = clean_query.lower().split()
            
            with self.db.get_cursor() as (conn, cur):
                # Use PostgreSQL full-text search with ranking
                search_query_sql = """
                    SELECT content_id, title, description, genre, mood_tags, 
                           rating, popularity_score, actors, director,
                           ts_rank(search_vector, plainto_tsquery('english', %s)) as text_rank,
                           CASE 
                               WHEN LOWER(title) ILIKE %s THEN 3.0
                               WHEN LOWER(title) ILIKE %s THEN 2.0
                               ELSE 1.0
                           END as title_boost
                    FROM content
                    WHERE search_vector @@ plainto_tsquery('english', %s)
                       OR LOWER(title) ILIKE %s
                       OR LOWER(description) ILIKE %s
                       OR actors::text ILIKE %s
                       OR director::text ILIKE %s
                    ORDER BY 
                        text_rank * title_boost DESC,
                        rating DESC,
                        popularity_score DESC
                    LIMIT %s
                """
                
                # Prepare search parameters
                exact_match = f"%{clean_query.lower()}%"
                partial_match = f"%{' '.join(search_terms)}%"
                
                params = (
                    clean_query,        # plainto_tsquery
                    exact_match,        # exact title match
                    partial_match,      # partial title match
                    clean_query,        # search_vector query
                    exact_match,        # title ILIKE
                    exact_match,        # description ILIKE
                    exact_match,        # actors ILIKE
                    exact_match,        # director ILIKE
                    limit
                )
                
                cur.execute(search_query_sql, params)
                results = cur.fetchall()
                
                if not results:
                    # Fallback to simpler text search
                    logger.debug("Full-text search returned no results, trying fallback")
                    return self._fallback_text_search(clean_query, include_details, limit)
                
                # Process results
                content_results = []
                for row in results:
                    content_item = {
                        'content_id': row[0],
                        'title': row[1],
                        'description': row[2],
                        'genre': row[3] or [],
                        'mood_tags': row[4] or [],
                        'rating': row[5] or 0,
                        'popularity_score': row[6] or 0,
                        'actors': row[7] or [],
                        'director': row[8] or [],
                        'text_rank': float(row[9]) if row[9] else 0.0,
                        'title_boost': float(row[10]) if row[10] else 1.0,
                        'search_relevance': round(float(row[9]) * float(row[10]), 3) if row[9] and row[10] else 0.0
                    }
                    
                    content_results.append(content_item)
                
                processing_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_processing_stats(processing_time)
                
                logger.debug(f"✅ Text search completed: {len(content_results)} results in {processing_time:.1f}ms")
                
                if include_details:
                    return content_results
                else:
                    return [item['content_id'] for item in content_results]
                
        except Exception as e:
            logger.error(f"❌ Text search failed: {e}")
            return self._fallback_text_search(search_query, include_details, limit)
    
    def _fallback_text_search(self, search_query, include_details, limit):
        """Fallback text search using ILIKE when full-text search unavailable."""
        try:
            with self.db.get_cursor() as (conn, cur):
                search_pattern = f"%{search_query.lower()}%"
                
                fallback_query = """
                    SELECT content_id, title, description, genre, mood_tags, 
                           rating, popularity_score, actors, director
                    FROM content
                    WHERE LOWER(title) ILIKE %s
                       OR LOWER(description) ILIKE %s
                       OR LOWER(actors::text) ILIKE %s
                       OR LOWER(director::text) ILIKE %s
                    ORDER BY 
                        CASE WHEN LOWER(title) ILIKE %s THEN 1 ELSE 2 END,
                        rating DESC,
                        popularity_score DESC
                    LIMIT %s
                """
                
                params = (search_pattern, search_pattern, search_pattern, 
                         search_pattern, search_pattern, limit)
                
                cur.execute(fallback_query, params)
                results = cur.fetchall()
                
                if include_details:
                    return [{
                        'content_id': row[0], 'title': row[1], 'description': row[2],
                        'genre': row[3] or [], 'mood_tags': row[4] or [],
                        'rating': row[5] or 0, 'popularity_score': row[6] or 0,
                        'actors': row[7] or [], 'director': row[8] or [],
                        'search_relevance': 1.0
                    } for row in results]
                else:
                    return [row[0] for row in results]
                    
        except Exception as e:
            logger.error(f"Fallback text search failed: {e}")
            return []
    
    def get_popular_content(self, time_period_days=7, limit=20):
        """
        Get popular content based on recent engagement.
        
        Args:
            time_period_days: Time period for popularity calculation
            limit: Maximum number of recommendations
            
        Returns:
            List of popular content with details
        """
        if not self.db:
            return []
        
        try:
            logger.debug(f"📈 Getting popular content for last {time_period_days} days")
            
            cutoff_date = datetime.now() - timedelta(days=time_period_days)
            
            with self.db.get_cursor() as (conn, cur):
                popular_query = """
                    SELECT c.content_id, c.title, c.description, c.genre, c.mood_tags,
                           c.rating, c.release_year, c.duration, c.director, c.actors,
                           COUNT(DISTINCT wh.user_id) as unique_viewers,
                           COUNT(wh.id) as total_views,
                           AVG(wh.completion_percentage) as avg_completion,
                           AVG(cr.rating) as avg_user_rating,
                           c.popularity_score
                    FROM content c
                    LEFT JOIN watch_history wh ON c.content_id = wh.content_id 
                        AND wh.start_time >= %s
                    LEFT JOIN content_reactions cr ON c.content_id = cr.content_id
                        AND cr.created_at >= %s
                    GROUP BY c.content_id, c.title, c.description, c.genre, c.mood_tags,
                             c.rating, c.release_year, c.duration, c.director, c.actors, c.popularity_score
                    HAVING COUNT(wh.id) > 0 OR c.popularity_score > 0.5
                    ORDER BY 
                        unique_viewers DESC,
                        total_views DESC,
                        avg_completion DESC,
                        c.popularity_score DESC,
                        c.rating DESC
                    LIMIT %s
                """
                
                cur.execute(popular_query, (cutoff_date, cutoff_date, limit))
                results = cur.fetchall()
                
                popular_content = []
                for row in results:
                    popular_content.append({
                        'content_id': row[0],
                        'title': row[1],
                        'description': row[2],
                        'genre': row[3] or [],
                        'mood_tags': row[4] or [],
                        'rating': row[5] or 0,
                        'release_year': row[6],
                        'duration': row[7] or 0,
                        'director': row[8] or [],
                        'actors': row[9] or [],
                        'unique_viewers': row[10] or 0,
                        'total_views': row[11] or 0,
                        'avg_completion': round(row[12] or 0, 1),
                        'avg_user_rating': round(row[13] or 0, 2),
                        'popularity_score': row[14] or 0
                    })
                
                logger.debug(f"✅ Retrieved {len(popular_content)} popular content items")
                return popular_content
                
        except Exception as e:
            logger.error(f"❌ Popular content retrieval failed: {e}")
            return []
    
    def get_content_details(self, content_ids):
        """
        Get detailed information for multiple content items.
        
        Args:
            content_ids: List of content IDs
            
        Returns:
            List of detailed content information
        """
        if not content_ids or not self.db:
            return []
        
        start_time = datetime.now()
        self.stats["content_details_fetched"] += 1
        
        try:
            # Check cache first
            cached_items = []
            uncached_ids = []
            
            for content_id in content_ids:
                cached_item = self._get_from_content_cache(content_id)
                if cached_item:
                    cached_items.append(cached_item)
                    self.stats["cache_hits"] += 1
                else:
                    uncached_ids.append(content_id)
            
            # Fetch uncached items from database
            db_items = []
            if uncached_ids:
                db_items = self.db.get_content_by_ids(uncached_ids)
                
                # Cache the fetched items
                for item in db_items:
                    self._add_to_content_cache(item['content_id'], dict(item))
            
            # Combine cached and database results
            all_items = cached_items + [dict(item) for item in db_items]
            
            # Sort according to original order
            content_id_to_item = {item['content_id']: item for item in all_items}
            ordered_items = []
            
            for content_id in content_ids:
                if content_id in content_id_to_item:
                    item = content_id_to_item[content_id]
                    
                    # Enrich with additional metadata
                    enriched_item = self._enrich_content_item(item)
                    ordered_items.append(enriched_item)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self._update_processing_stats(processing_time)
            
            logger.debug(f"✅ Retrieved details for {len(ordered_items)} content items in {processing_time:.1f}ms")
            
            return ordered_items
            
        except Exception as e:
            logger.error(f"❌ Content details retrieval failed: {e}")
            return []
    
    def _calculate_mood_appropriateness_score(self, genres, mood_tags, detected_mood, mood_config):
        """Calculate how appropriate content is for detected mood."""
        try:
            score = 0.5  # Base score
            
            boost_genres = set(mood_config.get('boost_genres', []))
            avoid_genres = set(mood_config.get('avoid_genres', []))
            boost_moods = set(mood_config.get('boost_moods', []))
            
            # Genre scoring
            content_genres = set(genres)
            if content_genres & boost_genres:
                score += 0.3 * len(content_genres & boost_genres)
            
            if content_genres & avoid_genres:
                score -= 0.4 * len(content_genres & avoid_genres)
            
            # Mood tag scoring
            content_moods = set(mood_tags)
            if content_moods & boost_moods:
                score += 0.4 * len(content_moods & boost_moods)
            
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.5
    
    def _apply_user_preference_boost(self, content_row, base_score, user_preferences):
        """Apply user preference boosting to content score."""
        try:
            if not user_preferences:
                return base_score
            
            boost_factor = 1.0
            
            # Genre preference boost
            preferred_genres = set(user_preferences.get('preferred_genres', []))
            content_genres = set(content_row[2] or [])  # genre field
            
            if content_genres & preferred_genres:
                boost_factor += 0.2 * len(content_genres & preferred_genres)
            
            # Quality preference boost
            rating = content_row[4] or 0  # rating field
            quality_threshold = user_preferences.get('quality_threshold', 3.5)
            
            if rating >= quality_threshold:
                boost_factor += 0.1
            
            return base_score * boost_factor
            
        except Exception:
            return base_score
    
    def _enrich_content_item(self, content_item):
        """Enrich content item with additional metadata and scores."""
        try:
            enriched = content_item.copy()
            
            # Calculate quality score
            rating = content_item.get('rating', 0)
            popularity = content_item.get('popularity_score', 0)
            view_count = content_item.get('view_count', 0)
            
            quality_score = (
                rating * self.quality_weights['rating'] +
                popularity * self.quality_weights['popularity_score'] +
                min(view_count / 1000, 1.0) * self.quality_weights['view_count']
            )
            
            enriched['quality_score'] = round(quality_score, 3)
            
            # Add content type classification
            duration = content_item.get('duration', 0)
            if duration:
                if duration < 45:
                    enriched['content_type'] = 'short'
                elif duration < 120:
                    enriched['content_type'] = 'medium'
                else:
                    enriched['content_type'] = 'long'
            else:
                enriched['content_type'] = 'unknown'
            
            # Add freshness indicator
            release_year = content_item.get('release_year')
            if release_year:
                current_year = datetime.now().year
                if release_year >= current_year - 1:
                    enriched['freshness'] = 'new'
                elif release_year >= current_year - 5:
                    enriched['freshness'] = 'recent'
                else:
                    enriched['freshness'] = 'classic'
            
            return enriched
            
        except Exception as e:
            logger.warning(f"Content enrichment failed: {e}")
            return content_item
    
    def _clean_search_query(self, query):
        """Clean and normalize search query."""
        try:
            # Remove special characters, normalize whitespace
            cleaned = re.sub(r'[^\w\s-]', ' ', query)
            cleaned = re.sub(r'\s+', ' ', cleaned).strip()
            return cleaned
        except:
            return query
    
    def _get_from_content_cache(self, content_id):
        """Get content from cache if available and fresh."""
        if content_id not in self.content_cache:
            return None
        
        entry = self.content_cache[content_id]
        age_seconds = (datetime.now() - entry['timestamp']).total_seconds()
        
        if age_seconds < self.cache_ttl:
            return entry['content']
        else:
            del self.content_cache[content_id]
            return None
    
    def _add_to_content_cache(self, content_id, content_data):
        """Add content to cache with size management."""
        if len(self.content_cache) >= self.max_cache_size:
            # Remove oldest 20% of entries
            oldest_keys = sorted(self.content_cache.keys(),
                               key=lambda k: self.content_cache[k]['timestamp'])[:100]
            for key in oldest_keys:
                del self.content_cache[key]
        
        self.content_cache[content_id] = {
            'timestamp': datetime.now(),
            'content': content_data
        }
    
    def _update_processing_stats(self, processing_time_ms):
        """Update processing time statistics."""
        try:
            self.stats["total_processing_time_ms"] += processing_time_ms
            total_operations = sum([
                self.stats["semantic_searches"],
                self.stats["mood_searches"], 
                self.stats["genre_searches"],
                self.stats["text_searches"],
                self.stats["content_details_fetched"]
            ])
            
            if total_operations > 0:
                self.stats["average_processing_time_ms"] = round(
                    self.stats["total_processing_time_ms"] / total_operations, 2
                )
        except:
            pass
    
    def get_content_statistics(self):
        """Get comprehensive content processing statistics."""
        try:
            stats = self.stats.copy()
            
            # Add cache statistics
            stats.update({
                "content_cache_size": len(self.content_cache),
                "cache_hit_rate_percent": round(
                    (self.stats["cache_hits"] / max(self.stats["content_details_fetched"], 1)) * 100, 2
                ),
                "total_operations": sum([
                    self.stats["semantic_searches"],
                    self.stats["mood_searches"],
                    self.stats["genre_searches"], 
                    self.stats["text_searches"],
                    self.stats["content_details_fetched"]
                ])
            })
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics collection failed: {e}")
            return self.stats
    
    def clear_cache(self):
        """Clear content processing cache."""
        cache_size = len(self.content_cache)
        self.content_cache.clear()
        logger.info(f"Cleared content cache ({cache_size} items)")
        return cache_size