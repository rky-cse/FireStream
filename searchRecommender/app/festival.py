# app/festival.py
import psycopg2
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from dataclasses import dataclass
import logging
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Festival:
    name: str
    start_date: datetime
    end_date: datetime
    typical_mood: str
    common_genres: List[str]

class FestivalDetector:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.db_config['database'] = 'firestream_db'
        
        # Common festivals cache
        self._festival_cache = None
    
    def _get_db_connection(self):
        """Create a new database connection with DictCursor"""
        return psycopg2.connect(
            **self.db_config,
            cursor_factory=RealDictCursor
        )
    
    def _get_all_festivals(self) -> List[Festival]:
        """Load all festivals from the database"""
        if self._festival_cache is not None:
            return self._festival_cache
            
        festivals = []
        try:
            with self._get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT DISTINCT active_festival, timestamp 
                        FROM user_contexts 
                        WHERE active_festival IS NOT NULL
                        ORDER BY timestamp DESC
                    """)
                    festival_names = {row['active_festival'] for row in cursor.fetchall()}
                    
                    for name in festival_names:
                        # Get most recent occurrence dates
                        cursor.execute("""
                            SELECT timestamp FROM user_contexts
                            WHERE active_festival = %s
                            ORDER BY timestamp DESC
                            LIMIT 1
                        """, (name,))
                        recent_occurrence = cursor.fetchone()['timestamp']
                        
                        festivals.append(Festival(
                            name=name,
                            start_date=recent_occurrence.replace(year=datetime.now().year),
                            end_date=(recent_occurrence + timedelta(days=3)).replace(year=datetime.now().year),
                            typical_mood=self._get_festival_mood(name),
                            common_genres=self._get_festival_genres(name)
                        ))
        
        except Exception as e:
            logger.error(f"Error loading festivals: {str(e)}")
        
        self._festival_cache = festivals
        return festivals
    
    def _get_festival_mood(self, festival_name: str) -> str:
        """Map festival to typical mood"""
        mood_mapping = {
            'Christmas': 'joyful',
            'New Year': 'celebratory',
            'Halloween': 'spooky',
            'Easter': 'happy',
            'Thanksgiving': 'grateful'
        }
        return mood_mapping.get(festival_name, 'happy')
    
    def _get_festival_genres(self, festival_name: str) -> List[str]:
        """Map festival to common genres"""
        genre_mapping = {
            'Christmas': ['family', 'comedy'],
            'New Year': ['party', 'music'],
            'Halloween': ['horror', 'thriller'],
            'Easter': ['family', 'animation'],
            'Thanksgiving': ['drama', 'family']
        }
        return genre_mapping.get(festival_name, [])
    
    def get_active_festivals(self, date: Optional[datetime] = None) -> List[Festival]:
        """Get festivals active around the given date (default: current date)"""
        if date is None:
            date = datetime.now()
            
        return [
            festival for festival in self._get_all_festivals()
            if festival.start_date <= date <= festival.end_date
        ]
    
    def get_festival_recommendation_context(self) -> Dict[str, any]:
        """Get recommendation context for active festivals"""
        active_festivals = self.get_active_festivals()
        if not active_festivals:
            return {}
        
        # For simplicity, use the first active festival
        festival = active_festivals[0]
        return {
            'active_festival': festival.name,
            'recommended_mood': festival.typical_mood,
            'recommended_genres': festival.common_genres
        }

# Smoke Test
if __name__ == "__main__":
    # Test configuration matching your setup
    TEST_DB_CONFIG = {
        'user': 'postgres',
        'password': 'postgres',
        'host': 'localhost',
        'port': '5432'
    }
    
    print("Running festival detector smoke test...")
    detector = FestivalDetector(TEST_DB_CONFIG)
    
    # Simulate Christmas period
    christmas_time = datetime(datetime.now().year, 12, 24)
    active_festivals = detector.get_active_festivals(christmas_time)
    
    if active_festivals:
        print("\nActive Festivals:")
        for fest in active_festivals:
            print(f"- {fest.name} ({fest.start_date} to {fest.end_date})")
            print(f"  Typical Mood: {fest.typical_mood}")
            print(f"  Common Genres: {', '.join(fest.common_genres)}")
        
        context = detector.get_festival_recommendation_context()
        print("\nRecommendation Context:")
        print(context)
    else:
        print("No active festivals found in database")
    
    print("\nCurrent Date Test:")
    current_festivals = detector.get_active_festivals()
    if current_festivals:
        print(f"Currently active: {', '.join(f.name for f in current_festivals)}")
    else:
        print("No festivals currently active")
    
    print("\nSmoke test completed!")