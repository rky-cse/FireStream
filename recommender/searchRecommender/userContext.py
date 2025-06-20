import logging
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict, Counter
import os
import sys

# Add ingestion path for context fetcher
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'ingestion'))

try:
    from context.fetcher import fetch_local_time, fetch_weather, fetch_festivals
    CONTEXT_FETCHER_AVAILABLE = True
except ImportError:
    CONTEXT_FETCHER_AVAILABLE = False

logger = logging.getLogger(__name__)

class UserContextAnalyzer:
    """AI-powered user context analysis for personalized recommendations with real-time environmental data."""
    
    def __init__(self, db_connection=None):
        self.db = db_connection
        self.cache = {}
        self.cache_expiry = 1800  # 30 minutes
        
        self.stats = {"context_requests": 0, "cache_hits": 0, "profiles_analyzed": 0, "weather_fetches": 0, "festival_checks": 0}
        
        # Context weights and mappings
        self.context_weights = {"viewing_history": 0.30, "explicit_preferences": 0.25, "temporal_context": 0.20, "environmental_context": 0.15, "social_influence": 0.10}
        
        self.weather_mood_mapping = {
            "Clear": {"mood": "happy", "genres": ["Adventure", "Comedy", "Romance"]},
            "Clouds": {"mood": "calm", "genres": ["Drama", "Documentary", "Feel-good"]},
            "Rain": {"mood": "cozy", "genres": ["Romance", "Drama", "Feel-good"]},
            "Snow": {"mood": "cozy", "genres": ["Family", "Romance", "Feel-good"]},
            "Thunderstorm": {"mood": "intense", "genres": ["Thriller", "Drama", "Action"]}
        }
        
        self.festival_mapping = {
            "christmas": ["Family", "Feel-good", "Holiday"], "diwali": ["Family", "Feel-good", "Drama"],
            "halloween": ["Horror", "Thriller", "Mystery"], "valentine": ["Romance", "Romantic-Comedy"],
            "new year": ["Comedy", "Feel-good", "Party"], "holi": ["Comedy", "Family", "Feel-good"]
        }
        
        logger.info(f"UserContextAnalyzer initialized at 2025-06-20 08:35:17 UTC by rky-cse")
        logger.info(f"Context fetcher available: {CONTEXT_FETCHER_AVAILABLE}")
    
    def get_user_context(self, user_id, user_location=None, timezone_str=None):
        """Get comprehensive user context with real-time environmental data."""
        self.stats["context_requests"] += 1
        
        try:
            # Check cache
            cache_key = f"user_context_{user_id}_{hash(str(user_location))}"
            cached_context = self._get_from_cache(cache_key)
            if cached_context:
                self.stats["cache_hits"] += 1
                cached_context["current_environment"] = self._get_real_time_context(user_id, user_location, timezone_str)
                return cached_context
            
            logger.debug(f"Analyzing context for user: {user_id}")
            
            context = {"user_id": user_id, "timestamp": datetime.now().isoformat(), "analysis_version": "v2.2_optimized", "context_confidence": 0.0}
            
            if not self.db:
                return self._get_minimal_context(user_id, user_location, timezone_str)
            
            # Get user data
            user_info = self._get_user_basic_info(user_id)
            if user_info:
                context.update(user_info)
                context["context_confidence"] += 0.15
                if not user_location and user_info.get('location'):
                    user_location = self._parse_location(user_info['location'])
            
            # Analyze patterns
            viewing_patterns = self._analyze_viewing_patterns(user_id)
            if viewing_patterns:
                context["viewing_patterns"] = viewing_patterns
                context["context_confidence"] += 0.25
            
            # Get preferences
            preferences = self._get_user_preferences(user_id)
            if preferences:
                context.update(preferences)
                context["context_confidence"] += 0.20
            
            # Temporal analysis
            temporal_context = self._analyze_temporal_patterns(user_id)
            if temporal_context:
                context["temporal_patterns"] = temporal_context
                context["context_confidence"] += 0.15
            
            # Real-time environment
            context["current_environment"] = self._get_real_time_context(user_id, user_location, timezone_str)
            if context["current_environment"].get("weather_available") or context["current_environment"].get("festivals_today"):
                context["context_confidence"] += 0.15
            
            # Social influences
            social_context = self._analyze_social_influences(user_id)
            if social_context:
                context["social_influences"] = social_context
                context["context_confidence"] += 0.10
            
            # Calculate insights and weights
            context["derived_insights"] = self._calculate_insights(context)
            context["personalization_weights"] = self._calculate_weights(context)
            
            # Cache without real-time data
            cache_context = {k: v for k, v in context.items() if k != "current_environment"}
            self._add_to_cache(cache_key, cache_context)
            
            self.stats["profiles_analyzed"] += 1
            logger.debug(f"Context analysis complete for {user_id} (confidence: {context['context_confidence']:.2f})")
            
            return context
            
        except Exception as e:
            logger.error(f"Error analyzing context for {user_id}: {e}")
            return self._get_minimal_context(user_id, user_location, timezone_str, error=str(e))
    
    def _get_user_basic_info(self, user_id):
        """Get basic user information."""
        try:
            with self.db.get_cursor() as (conn, cur):
                cur.execute("SELECT name, age, gender, location, created_at, timezone FROM users WHERE user_id = %s", (user_id,))
                result = cur.fetchone()
                
                if result:
                    return {
                        "name": result[0], "age": result[1], "gender": result[2], "location": result[3],
                        "timezone": result[5] if len(result) > 5 else None,
                        "account_age_days": (datetime.now() - result[4]).days if result[4] else 0,
                        "user_segment": self._get_user_segment(result[1], result[2])
                    }
                return {}
        except Exception as e:
            logger.warning(f"Failed to get user info: {e}")
            return {}
    
    def _get_user_segment(self, age, gender):
        """Determine user demographic segment."""
        try:
            age = int(age) if age else 25
            segment = "young_adult" if age < 25 else "millennial" if age < 35 else "gen_x" if age < 50 else "mature"
            return f"{segment}_{gender}" if gender else segment
        except:
            return "general"
    
    def _analyze_viewing_patterns(self, user_id):
        """Analyze comprehensive viewing behavior."""
        try:
            with self.db.get_cursor() as (conn, cur):
                cur.execute("""
                    SELECT c.genre, c.mood_tags, c.duration, c.rating, wh.completion_percentage,
                           wh.day_of_week, wh.hour_of_day, wh.is_weekend, cr.rating as user_rating
                    FROM watch_history wh
                    JOIN content c ON wh.content_id = c.content_id
                    LEFT JOIN content_reactions cr ON wh.user_id = cr.user_id AND wh.content_id = cr.content_id
                    WHERE wh.user_id = %s
                    ORDER BY wh.start_time DESC
                    LIMIT 200
                """, (user_id,))
                
                viewing_data = cur.fetchall()
                if not viewing_data:
                    return {}
                
                # Analyze patterns efficiently
                genre_stats = defaultdict(lambda: {"count": 0, "completion": 0, "rating": 0, "rating_count": 0})
                mood_stats = defaultdict(lambda: {"count": 0, "completion": 0})
                durations = []
                completions = []
                
                for row in viewing_data:
                    genres, moods, duration, rating, completion = row[0] or [], row[1] or [], row[2] or 0, row[3] or 0, row[4] or 0
                    user_rating = row[8]
                    
                    durations.append(duration)
                    completions.append(completion)
                    
                    # Genre analysis
                    for genre in genres:
                        genre_stats[genre]["count"] += 1
                        genre_stats[genre]["completion"] += completion
                        if user_rating:
                            genre_stats[genre]["rating"] += user_rating
                            genre_stats[genre]["rating_count"] += 1
                    
                    # Mood analysis
                    for mood in moods:
                        mood_stats[mood]["count"] += 1
                        mood_stats[mood]["completion"] += completion
                
                # Calculate preferences
                preferred_genres = []
                for genre, stats in genre_stats.items():
                    if stats["count"] >= 2:
                        avg_completion = stats["completion"] / stats["count"]
                        avg_rating = stats["rating"] / stats["rating_count"] if stats["rating_count"] > 0 else 3.0
                        score = (stats["count"] / len(viewing_data)) * 0.4 + (avg_completion / 100) * 0.35 + (avg_rating / 5.0) * 0.25
                        preferred_genres.append({"genre": genre, "score": round(score, 3), "count": stats["count"]})
                
                preferred_moods = []
                for mood, stats in mood_stats.items():
                    if stats["count"] >= 2:
                        avg_completion = stats["completion"] / stats["count"]
                        score = (stats["count"] / len(viewing_data)) * 0.5 + (avg_completion / 100) * 0.5
                        preferred_moods.append({"mood": mood, "score": round(score, 3)})
                
                preferred_genres.sort(key=lambda x: x["score"], reverse=True)
                preferred_moods.sort(key=lambda x: x["score"], reverse=True)
                
                return {
                    "total_content_watched": len(viewing_data),
                    "average_completion": round(np.mean(completions), 1),
                    "preferred_genres": [g["genre"] for g in preferred_genres[:6]],
                    "preferred_moods": [m["mood"] for m in preferred_moods[:8]],
                    "avg_duration": round(np.mean(durations), 1) if durations else 0,
                    "quality_threshold": 4.0 if np.mean([row[3] for row in viewing_data if row[3]]) > 4.0 else 3.5,
                    "engagement_level": "high" if np.mean(completions) > 80 else "medium" if np.mean(completions) > 60 else "low"
                }
                
        except Exception as e:
            logger.warning(f"Viewing pattern analysis failed: {e}")
            return {}
    
    def _get_user_preferences(self, user_id):
        """Get AI-derived user preferences."""
        try:
            with self.db.get_cursor() as (conn, cur):
                cur.execute("""
                    SELECT preferred_genres, preferred_duration_min, preferred_duration_max,
                           mood_genre_mapping, personality_traits
                    FROM user_preferences WHERE user_id = %s
                """, (user_id,))
                
                result = cur.fetchone()
                if result:
                    return {
                        "preferred_genres": result[0] or [],
                        "preferred_duration": {"min": result[1] or 60, "max": result[2] or 150},
                        "mood_genre_mapping": json.loads(result[3]) if result[3] else {},
                        "personality_traits": json.loads(result[4]) if result[4] else {},
                        "ai_preferences_available": True
                    }
                return {"ai_preferences_available": False}
        except Exception as e:
            logger.warning(f"Failed to get user preferences: {e}")
            return {}
    
    def _analyze_temporal_patterns(self, user_id):
        """Analyze temporal viewing patterns."""
        try:
            with self.db.get_cursor() as (conn, cur):
                cur.execute("""
                    SELECT day_of_week, hour_of_day, is_weekend, COUNT(*) as sessions,
                           AVG(completion_percentage) as avg_completion
                    FROM watch_history WHERE user_id = %s
                    GROUP BY day_of_week, hour_of_day, is_weekend
                    ORDER BY sessions DESC
                """, (user_id,))
                
                temporal_data = cur.fetchall()
                if not temporal_data:
                    return {}
                
                # Find peak hours and days
                hour_stats = defaultdict(int)
                day_stats = defaultdict(int)
                weekend_sessions, weekday_sessions = 0, 0
                
                for row in temporal_data:
                    day, hour, is_weekend, sessions, completion = row
                    hour_stats[hour] += sessions
                    day_stats[day] += sessions
                    
                    if is_weekend:
                        weekend_sessions += sessions
                    else:
                        weekday_sessions += sessions
                
                peak_hours = sorted(hour_stats.items(), key=lambda x: x[1], reverse=True)[:3]
                peak_days = sorted(day_stats.items(), key=lambda x: x[1], reverse=True)[:3]
                
                return {
                    "peak_hours": [hour for hour, _ in peak_hours],
                    "peak_days": [f"day_{day}" for day, _ in peak_days],
                    "prefers_weekend": weekend_sessions > weekday_sessions,
                    "current_time_boost": self._get_current_time_boost()
                }
                
        except Exception as e:
            logger.warning(f"Temporal analysis failed: {e}")
            return {}
    
    def _get_current_time_boost(self):
        """Get current time-based boost factor."""
        hour = datetime.now().hour
        is_weekend = datetime.now().weekday() >= 5
        
        if 6 <= hour < 12:
            period = "morning"
            boost = 1.0
        elif 12 <= hour < 17:
            period = "afternoon"
            boost = 1.1
        elif 17 <= hour < 22:
            period = "evening"
            boost = 1.3
        else:
            period = "night"
            boost = 0.8
        
        return {"period": period, "boost_factor": boost * (1.2 if is_weekend else 1.0), "is_weekend": is_weekend}
    
    def _analyze_social_influences(self, user_id):
        """Analyze social connections and influences."""
        try:
            with self.db.get_cursor() as (conn, cur):
                cur.execute("""
                    SELECT COUNT(*) as connection_count,
                           AVG(connection_strength) as avg_strength
                    FROM user_connections WHERE user_id = %s
                """, (user_id,))
                
                result = cur.fetchone()
                if result and result[0] > 0:
                    return {
                        "has_social_connections": True,
                        "connection_count": result[0],
                        "avg_connection_strength": round(result[1], 2),
                        "social_influence_score": min(result[0] / 10.0, 1.0)
                    }
                return {"has_social_connections": False, "social_influence_score": 0.0}
        except Exception as e:
            logger.warning(f"Social analysis failed: {e}")
            return {"has_social_connections": False}
    
    def _get_real_time_context(self, user_id, user_location=None, timezone_str=None):
        """Get real-time environmental context."""
        try:
            context = {"timestamp": datetime.now().isoformat(), "source": "real_time"}
            
            # Get time context
            if CONTEXT_FETCHER_AVAILABLE:
                try:
                    time_data = fetch_local_time(timezone_str or 'Asia/Kolkata')
                    context.update(time_data)
                    context["time_source"] = "fetcher"
                except Exception as e:
                    logger.warning(f"Time fetch failed: {e}")
                    context.update(self._get_fallback_time())
            else:
                context.update(self._get_fallback_time())
            
            # Get weather context
            if user_location and CONTEXT_FETCHER_AVAILABLE:
                try:
                    weather = fetch_weather(user_location.get('lat', 22.5726), user_location.get('lon', 88.3639))
                    context["weather"] = weather
                    context["weather_available"] = True
                    self.stats["weather_fetches"] += 1
                    
                    # Add weather recommendations
                    weather_rec = self.weather_mood_mapping.get(weather["main"], {})
                    if weather_rec:
                        context["weather_recommendations"] = {
                            "mood": weather_rec["mood"],
                            "genres": weather_rec["genres"],
                            "influence_score": self._get_weather_influence(weather)
                        }
                except Exception as e:
                    logger.warning(f"Weather fetch failed: {e}")
                    context["weather_available"] = False
            
            # Get festival context
            if CONTEXT_FETCHER_AVAILABLE:
                try:
                    festivals = fetch_festivals(country='IN', year=datetime.now().year)
                    today = datetime.now().date().isoformat()
                    today_festivals = [f for f in festivals if f["date"] == today]
                    
                    context["festivals_today"] = today_festivals
                    context["festivals_available"] = True
                    
                    if today_festivals:
                        self.stats["festival_checks"] += 1
                        festival_genres = []
                        for festival in today_festivals:
                            for key, genres in self.festival_mapping.items():
                                if key in festival["name"].lower():
                                    festival_genres.extend(genres)
                        
                        if festival_genres:
                            context["festival_recommendations"] = {
                                "active_festivals": [f["name"] for f in today_festivals],
                                "genres": list(set(festival_genres)),
                                "influence_score": 0.8
                            }
                except Exception as e:
                    logger.warning(f"Festival fetch failed: {e}")
                    context["festivals_available"] = False
            
            # Calculate environmental influence
            context["environmental_influence"] = self._calculate_environmental_influence(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Real-time context failed: {e}")
            return self._get_fallback_time()
    
    def _get_fallback_time(self):
        """Fallback time context."""
        now = datetime.now()
        return {
            "date": now.date().isoformat(),
            "time": now.time().strftime("%H:%M:%S"),
            "is_weekend": now.weekday() >= 5,
            "time_source": "fallback"
        }
    
    def _get_weather_influence(self, weather):
        """Calculate weather influence score."""
        temp = weather.get("temperature", 25)
        condition = weather.get("main", "Clear")
        
        influence = 0.3
        if temp < 10 or temp > 35:
            influence += 0.2
        if condition in ["Rain", "Snow", "Thunderstorm"]:
            influence += 0.3
        
        return min(influence, 1.0)
    
    def _calculate_environmental_influence(self, context):
        """Calculate overall environmental influence."""
        influence = 0.1
        
        if context.get("weather_available"):
            influence += context.get("weather_recommendations", {}).get("influence_score", 0) * 0.4
        
        if context.get("festivals_today"):
            influence += context.get("festival_recommendations", {}).get("influence_score", 0) * 0.3
        
        if context.get("is_weekend"):
            influence += 0.1
        
        return min(influence, 1.0)
    
    def _parse_location(self, location_string):
        """Parse location string to coordinates."""
        coords = {
            "kolkata": {"lat": 22.5726, "lon": 88.3639}, "mumbai": {"lat": 19.0760, "lon": 72.8777},
            "delhi": {"lat": 28.7041, "lon": 77.1025}, "bangalore": {"lat": 12.9716, "lon": 77.5946},
            "chennai": {"lat": 13.0827, "lon": 80.2707}, "hyderabad": {"lat": 17.3850, "lon": 78.4867}
        }
        
        location_lower = location_string.lower()
        for city, coord in coords.items():
            if city in location_lower:
                return coord
        return {"lat": 22.5726, "lon": 88.3639}  # Default to Kolkata
    
    def _calculate_insights(self, context):
        """Calculate high-level user insights."""
        viewing = context.get("viewing_patterns", {})
        engagement = viewing.get("engagement_level", "low")
        total_watched = viewing.get("total_content_watched", 0)
        
        return {
            "user_sophistication": "high" if engagement == "high" and total_watched > 50 else "medium" if total_watched > 20 else "low",
            "discovery_tendency": "high" if len(viewing.get("preferred_genres", [])) > 4 else "medium" if len(viewing.get("preferred_genres", [])) > 2 else "low",
            "viewing_style": "binge" if viewing.get("average_completion", 0) > 85 else "selective" if viewing.get("average_completion", 0) > 70 else "casual",
            "quality_preference": "high" if viewing.get("quality_threshold", 3.5) >= 4.0 else "medium"
        }
    
    def _calculate_weights(self, context):
        """Calculate dynamic personalization weights."""
        weights = self.context_weights.copy()
        
        # Adjust based on data availability
        viewing = context.get("viewing_patterns", {})
        total_watched = viewing.get("total_content_watched", 0)
        env_influence = context.get("current_environment", {}).get("environmental_influence", 0.1)
        
        if total_watched > 50:
            weights["viewing_history"] += 0.10
            weights["environmental_context"] -= 0.05
            weights["explicit_preferences"] -= 0.05
        elif total_watched < 10:
            weights["environmental_context"] += 0.10
            weights["explicit_preferences"] += 0.10
            weights["viewing_history"] -= 0.20
        
        if env_influence > 0.5:
            weights["environmental_context"] += 0.10
            weights["viewing_history"] -= 0.05
            weights["temporal_context"] -= 0.05
        
        # Normalize weights
        total = sum(weights.values())
        return {k: round(v/total, 3) for k, v in weights.items()} if total > 0 else weights
    
    def _get_minimal_context(self, user_id, user_location=None, timezone_str=None, error=None):
        """Minimal context for fallback."""
        context = {
            "user_id": user_id, "timestamp": datetime.now().isoformat(), "context_confidence": 0.1,
            "minimal_mode": True, "preferred_genres": [], "viewing_patterns": {}, "temporal_patterns": {}
        }
        
        if error:
            context["error"] = error
        
        # Always add real-time context
        context["current_environment"] = self._get_real_time_context(user_id, user_location, timezone_str)
        
        return context
    
    def _get_from_cache(self, cache_key):
        """Get from cache if fresh."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if (datetime.now() - entry['timestamp']).total_seconds() < self.cache_expiry:
                return entry['context']
        return None
    
    def _add_to_cache(self, cache_key, context):
        """Add to cache with size management."""
        self.cache[cache_key] = {'timestamp': datetime.now(), 'context': context}
        
        if len(self.cache) > 100:
            # Remove oldest 20 entries
            sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            for old_key in sorted_keys[:20]:
                del self.cache[old_key]
    
    def get_context_statistics(self):
        """Get context analysis statistics."""
        cache_hit_rate = (self.stats["cache_hits"] / max(self.stats["context_requests"], 1)) * 100
        
        return {
            "context_requests": self.stats["context_requests"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "profiles_analyzed": self.stats["profiles_analyzed"],
            "weather_fetches": self.stats["weather_fetches"],
            "festival_checks": self.stats["festival_checks"],
            "context_fetcher_available": CONTEXT_FETCHER_AVAILABLE,
            "cache_size": len(self.cache)
        }
    
    def update_user_location(self, user_id, latitude, longitude):
        """Update user location."""
        try:
            # Clear cache for this user
            cache_keys = [key for key in self.cache.keys() if f"user_context_{user_id}" in key]
            for key in cache_keys:
                del self.cache[key]
            
            if self.db:
                with self.db.get_cursor() as (conn, cur):
                    cur.execute("UPDATE users SET location = %s WHERE user_id = %s", (f"{latitude},{longitude}", user_id))
                    conn.commit()
            
            logger.debug(f"Updated location for {user_id}: {latitude}, {longitude}")
            return True
        except Exception as e:
            logger.error(f"Location update failed: {e}")
            return False