# app/user_context.py
import psycopg2
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import json
import logging
import requests
from psycopg2.extras import RealDictCursor
from pathlib import Path
from searchRecommender.app.festival import FestivalDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UserContext:
    # Core user information
    user_id: str
    age: Optional[int] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    
    # Temporal context
    time_of_day: str = "day"  # morning/afternoon/evening/night
    day_of_week: str = "weekday"  # weekday/weekend
    current_hour: int = 12
    
    # Environmental context
    weather: Optional[str] = None
    temperature: Optional[float] = None
    active_festival: Optional[str] = None
    
    # Device context
    last_device_used: Optional[str] = None
    
    # Content preferences
    preferred_genres: List[str] = None
    mood_genre_mapping: Dict[str, List[str]] = None
    
    # Social context
    friends_recently_watched: List[Dict[str, Any]] = None
    group_watch_history: List[Dict[str, Any]] = None
    
    # Behavioral context
    last_watched_genres: List[str] = None
    typical_watch_time: Optional[str] = None
    recent_searches: List[Dict[str, Any]] = None
    watch_history: List[Dict[str, Any]] = None  # Added to match what's being passed

class UserContextFetcher:
    def __init__(self, db_config: Dict):
        self.db_config = db_config.copy()
        # Ensure consistent database name parameter
        if 'database' in self.db_config:
            self.db_config['database'] = 'firestream_db'
        else:
            self.db_config['dbname'] = 'firestream_db'
        self.weather_api_key = 'd3692e793b48ec4d320c21c617a31e89'
        self.festival_detector = FestivalDetector(db_config)
    
    
    
    def _get_db_connection(self):
        """Create a new database connection with DictCursor"""
        try:
            return psycopg2.connect(
                **self.db_config,
                cursor_factory=RealDictCursor
            )
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise
    
    def _get_time_context(self) -> tuple:
        """Get current time context"""
        now = datetime.now()
        hour = now.hour
        
        time_of_day = (
            'morning' if 5 <= hour < 12 else
            'afternoon' if 12 <= hour < 17 else
            'evening' if 17 <= hour < 21 else
            'night'
        )
        
        day_of_week = 'weekend' if now.weekday() >= 5 else 'weekday'
        
        return time_of_day, day_of_week, hour
    
    def _get_weather_data(self, location: str) -> Optional[Dict]:
        """Fetch weather data from OpenWeatherMap API"""
        if not self.weather_api_key or not location:
            return None
            
        try:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={self.weather_api_key}&units=metric"
            response = requests.get(url, timeout=3)
            response.raise_for_status()
            data = response.json()
            return {
                'weather': data['weather'][0]['main'].lower(),
                'temperature': data['main']['temp']
            }
        except Exception as e:
            logger.error(f"Weather API error: {str(e)}")
            return None
    
    def _get_user_preferences(self, user_id: str, conn) -> Dict:
        """Fetch user preferences from database"""
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT preferred_genres, mood_genre_mapping
                    FROM user_preferences
                    WHERE user_id = %s
                """, (user_id,))
                result = cursor.fetchone()
                return result if result else {
                    'preferred_genres': [],
                    'mood_genre_mapping': '{}'
                }
        except Exception as e:
            logger.error(f"Error fetching preferences: {str(e)}")
            return {
                'preferred_genres': [],
                'mood_genre_mapping': '{}'
            }
    
    def _get_watch_history(self, user_id: str, conn) -> List[Dict]:
        """Get user's watch history"""
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT wh.*, c.genre 
                    FROM watch_history wh
                    JOIN content c ON wh.content_id = c.content_id
                    WHERE wh.user_id = %s
                    ORDER BY wh.start_time DESC
                    LIMIT 20
                """, (user_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching watch history: {str(e)}")
            return []
    
    def _get_last_watch_session(self, user_id: str, conn) -> Dict:
        """Get user's last watch session details"""
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT c.genre, wh.device_id, wh.context_weather, 
                           wh.context_temperature, wh.start_time
                    FROM watch_history wh
                    JOIN content c ON wh.content_id = c.content_id
                    WHERE wh.user_id = %s
                    ORDER BY wh.start_time DESC
                    LIMIT 1
                """, (user_id,))
                return cursor.fetchone() or {}
        except Exception as e:
            logger.error(f"Error fetching watch history: {str(e)}")
            return {}
    
    def _get_friends_activity(self, user_id: str, conn) -> List[Dict]:
        """Get recent watches by user's friends"""
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT c.content_id, c.title, c.genre, c.mood_tags,
                           wh.start_time, u.name as friend_name
                    FROM watch_history wh
                    JOIN content c ON wh.content_id = c.content_id
                    JOIN users u ON wh.user_id = u.user_id
                    JOIN user_connections uc ON wh.user_id = uc.friend_id
                    WHERE uc.user_id = %s
                    ORDER BY wh.start_time DESC
                    LIMIT 5
                """, (user_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching friends activity: {str(e)}")
            return []
    
    def _get_group_activity(self, user_id: str, conn) -> List[Dict]:
        """Get user's recent group watch sessions"""
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT gs.content_id, c.title, gs.start_time, 
                           gs.dominant_emotion, gs.group_sentiment_score,
                           COUNT(gm.user_id) as participants_count
                    FROM group_sessions gs
                    JOIN content c ON gs.content_id = c.content_id
                    JOIN group_members gm ON gs.group_id = gm.group_id
                    WHERE gm.user_id = %s
                    GROUP BY gs.content_id, c.title, gs.start_time, 
                             gs.dominant_emotion, gs.group_sentiment_score
                    ORDER BY gs.start_time DESC
                    LIMIT 3
                """, (user_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching group activity: {str(e)}")
            return []
    
    def _get_typical_watch_time(self, user_id: str, conn) -> Optional[str]:
        """Calculate user's most common watch time"""
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN EXTRACT(HOUR FROM start_time) BETWEEN 5 AND 11 THEN 'morning'
                            WHEN EXTRACT(HOUR FROM start_time) BETWEEN 12 AND 16 THEN 'afternoon'
                            WHEN EXTRACT(HOUR FROM start_time) BETWEEN 17 AND 20 THEN 'evening'
                            ELSE 'night'
                        END as time_of_day,
                        COUNT(*) as count
                    FROM watch_history
                    WHERE user_id = %s
                    GROUP BY time_of_day
                    ORDER BY count DESC
                    LIMIT 1
                """, (user_id,))
                result = cursor.fetchone()
                return result['time_of_day'] if result else None
        except Exception as e:
            logger.error(f"Error calculating typical watch time: {str(e)}")
            return None
    
    def _get_recent_searches(self, user_id: str, conn) -> List[Dict]:
        """Get user's recent search queries"""
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT search_query, search_timestamp, detected_mood, search_intent
                    FROM search_history
                    WHERE user_id = %s
                    ORDER BY search_timestamp DESC
                    LIMIT 5
                """, (user_id,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Error fetching recent searches: {str(e)}")
            return []
    
    def fetch_context(self, user_id: str) -> UserContext:
        """Main method to gather all contextual signals"""
        try:
            with self._get_db_connection() as conn:
                # Get base user info
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT age, gender, location 
                        FROM users 
                        WHERE user_id = %s
                    """, (user_id,))
                    user_data = cursor.fetchone()
                    if not user_data:
                        raise ValueError(f"User {user_id} not found")
                
                # Get time context
                time_of_day, day_of_week, current_hour = self._get_time_context()
                
                # Get preferences
                preferences = self._get_user_preferences(user_id, conn)
                
                # Get watch history
                watch_history = self._get_watch_history(user_id, conn)
                
                # Get last watch session
                last_watch = self._get_last_watch_session(user_id, conn)
                
                # Get weather data - prefer last recorded weather if available
                weather_data = None
                if last_watch and last_watch.get('context_weather'):
                    weather_data = {
                        'weather': last_watch['context_weather'],
                        'temperature': last_watch['context_temperature']
                    }
                else:
                    weather_data = self._get_weather_data(user_data['location'])
                
                # Get festival context
                festival_context = self.festival_detector.get_festival_recommendation_context()
                active_festival = festival_context.get('active_festival')
                
                # Get social activity
                friends_activity = self._get_friends_activity(user_id, conn)
                group_activity = self._get_group_activity(user_id, conn)
                
                # Get behavioral patterns
                typical_watch_time = self._get_typical_watch_time(user_id, conn)
                recent_searches = self._get_recent_searches(user_id, conn)
                
                # Prepare mood-genre mapping
                mood_mapping = {}
                try:
                    mood_mapping = json.loads(preferences['mood_genre_mapping'])
                except (json.JSONDecodeError, KeyError, TypeError):
                    mood_mapping = {}
                
                return UserContext(
                    user_id=user_id,
                    age=user_data.get('age'),
                    gender=user_data.get('gender'),
                    location=user_data.get('location'),
                    time_of_day=time_of_day,
                    day_of_week=day_of_week,
                    current_hour=current_hour,
                    weather=weather_data['weather'] if weather_data else None,
                    temperature=weather_data['temperature'] if weather_data else None,
                    active_festival=active_festival,
                    last_device_used=last_watch.get('device_id'),
                    preferred_genres=preferences.get('preferred_genres', []),
                    mood_genre_mapping=mood_mapping,
                    friends_recently_watched=friends_activity,
                    group_watch_history=group_activity,
                    last_watched_genres=[last_watch.get('genre')] if last_watch.get('genre') else [],
                    typical_watch_time=typical_watch_time,
                    recent_searches=recent_searches,
                    watch_history=watch_history
                )
                
        except Exception as e:
            logger.error(f"Error fetching context for user {user_id}: {str(e)}")
            # Return minimal context if there's an error
            time_of_day, day_of_week, current_hour = self._get_time_context()
            return UserContext(
                user_id=user_id,
                time_of_day=time_of_day,
                day_of_week=day_of_week,
                current_hour=current_hour,
                preferred_genres=[],
                mood_genre_mapping={},
                friends_recently_watched=[],
                group_watch_history=[],
                last_watched_genres=[],
                recent_searches=[],
                watch_history=[]
            )

# Smoke Test
if __name__ == "__main__":
    # Test configuration - replace with your actual DB credentials
    TEST_DB_CONFIG = {
        'host': 'localhost',
        'port': '5432',
        'database': 'firestream_db',  # Consistent parameter name
        'user': 'postgres',
        'password': 'postgres'
    }
    
    fetcher = UserContextFetcher(TEST_DB_CONFIG)
    
    try:
        print("Fetching context for test user...")
        # Use one of the user_ids from your populated database
        context = fetcher.fetch_context("USR10001")
        
        print("\n=== User Context ===")
        print(f"User ID: {context.user_id}")
        print(f"Location: {context.location}")
        print(f"Time: {context.time_of_day} (Hour: {context.current_hour})")
        print(f"Weather: {context.weather} at {context.temperature}°C")
        print(f"Festival: {context.active_festival or 'None'}")
        print(f"Preferred Genres: {context.preferred_genres}")
        print(f"Friends Activity: {len(context.friends_recently_watched)} recent watches")
        print(f"Group History: {len(context.group_watch_history)} sessions")
        print(f"Typical Watch Time: {context.typical_watch_time}")
        print(f"Recent Searches: {[s['search_query'] for s in context.recent_searches]}")
        print(f"Watch History Items: {len(context.watch_history)}")
        
        print("\nSmoke test completed successfully!")
    except Exception as e:
        print(f"Test failed: {str(e)}")