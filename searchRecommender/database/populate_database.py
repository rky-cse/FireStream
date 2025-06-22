import psycopg2
import json
import random
from datetime import datetime, timedelta
from faker import Faker
import numpy as np
from database_connection import db
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

fake = Faker()

def populate_users():
    """Populate users table with diverse profiles"""
    logger.info("Populating users...")
    
    users_data = [
        ('rky-cse', 'Ravi Kumar', 28, 'male', 'San Francisco'),
        ('sarah_chen', 'Sarah Chen', 25, 'female', 'New York'),
        ('mike_jones', 'Mike Jones', 32, 'male', 'Los Angeles'), 
        ('emma_davis', 'Emma Davis', 29, 'female', 'Chicago'),
        ('alex_kim', 'Alex Kim', 26, 'non-binary', 'Austin'),
        ('carlos_ruiz', 'Carlos Ruiz', 35, 'male', 'Miami'),
        ('lily_wang', 'Lily Wang', 23, 'female', 'Seattle'),
        ('david_brown', 'David Brown', 41, 'male', 'Boston'),
        ('zoe_taylor', 'Zoe Taylor', 27, 'female', 'Portland'),
        ('raj_patel', 'Raj Patel', 30, 'male', 'Denver')
    ]
    
    with db.get_cursor() as (conn, cur):
        for user_data in users_data:
            cur.execute("""
                INSERT INTO users (user_id, name, age, gender, location)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
            """, user_data)
    
    logger.info("✅ Users populated")

def populate_content():
    """Populate content table with diverse video metadata"""
    logger.info("Populating content...")
    
    content_data = [
        {
            'content_id': 'mv_001',
            'title': 'The Midnight Garden',
            'type': 'movie',
            'genre': ['Drama', 'Feel-good', 'Family'],
            'release_year': 2023,
            'duration': 108,
            'director': ['Sarah Mitchell'],
            'actors': ['Emma Stone', 'Tom Hanks', 'Lupita Nyongo'],
            'description': 'A heartwarming story about finding hope in dark times through the magic of an enchanted garden that blooms only at midnight.',
            'rating': 4.2,
            'mood_tags': ['uplifting', 'hopeful', 'warm', 'inspiring', 'heartwarming']
        },
        {
            'content_id': 'mv_002', 
            'title': 'Quantum Paradox',
            'type': 'movie',
            'genre': ['Sci-Fi', 'Thriller', 'Mind-bending'],
            'release_year': 2024,
            'duration': 142,
            'director': ['Christopher Chen'],
            'actors': ['Oscar Isaac', 'Tilda Swinton', 'John Boyega'],
            'description': 'Mind-bending sci-fi thriller about parallel realities and quantum consciousness where nothing is as it seems.',
            'rating': 4.6,
            'mood_tags': ['intense', 'thought-provoking', 'mysterious', 'complex', 'cerebral']
        },
        {
            'content_id': 'mv_003',
            'title': 'Laugh Out Loud',
            'type': 'movie', 
            'genre': ['Comedy', 'Romantic-Comedy', 'Feel-good'],
            'release_year': 2023,
            'duration': 95,
            'director': ['Judd Apatow'],
            'actors': ['Ryan Reynolds', 'Amy Poehler', 'Kevin Hart'],
            'description': 'The funniest comedy of the year that will cure any bad mood with non-stop laughter and witty dialogue.',
            'rating': 4.0,
            'mood_tags': ['hilarious', 'light-hearted', 'fun', 'cheerful', 'witty']
        },
        {
            'content_id': 'mv_004',
            'title': 'Silent Tears',
            'type': 'movie',
            'genre': ['Drama', 'Emotional', 'Character-Study'],
            'release_year': 2023,
            'duration': 126,
            'director': ['Chloé Zhao'],
            'actors': ['Frances McDormand', 'Mahershala Ali', 'Saoirse Ronan'],
            'description': 'A deeply moving drama about loss, grief, and the journey to redemption through human connection.',
            'rating': 4.4,
            'mood_tags': ['emotional', 'contemplative', 'profound', 'cathartic', 'melancholic']
        },
        {
            'content_id': 'mv_005',
            'title': 'Adventure Seekers',
            'type': 'movie',
            'genre': ['Action', 'Adventure', 'Thriller'],
            'release_year': 2024,
            'duration': 118,
            'director': ['Denis Villeneuve'],
            'actors': ['Zendaya', 'Michael B. Jordan', 'Gal Gadot'],
            'description': 'High-octane action adventure across exotic locations with breathtaking stunts and spectacular cinematography.',
            'rating': 4.1,
            'mood_tags': ['exciting', 'energetic', 'thrilling', 'adventurous', 'adrenaline']
        },
        {
            'content_id': 'mv_006',
            'title': 'The Cozy Café',
            'type': 'movie',
            'genre': ['Romance', 'Feel-good', 'Small-Town'],
            'release_year': 2023,
            'duration': 102,
            'director': ['Nancy Meyers'],
            'actors': ['Anne Hathaway', 'Hugh Jackman', 'Sandra Bullock'],
            'description': 'A gentle romantic story set in a charming small town café where love blooms over coffee and pastries.',
            'rating': 3.8,
            'mood_tags': ['cozy', 'warm', 'relaxing', 'sweet', 'romantic']
        },
        {
            'content_id': 'mv_007',
            'title': 'Mind Games',
            'type': 'movie',
            'genre': ['Thriller', 'Psychological', 'Mystery'],
            'release_year': 2024,
            'duration': 134,
            'director': ['Jordan Peele'],
            'actors': ['LaKeith Stanfield', 'Lupita Nyongo', 'Daniel Kaluuya'],
            'description': 'Psychological thriller that will keep you guessing until the very last scene with its intricate plot twists.',
            'rating': 4.5,
            'mood_tags': ['intense', 'gripping', 'suspenseful', 'dark', 'mind-bending']
        },
        {
            'content_id': 'mv_008',
            'title': 'Family Picnic',
            'type': 'movie',
            'genre': ['Family', 'Animated', 'Adventure'],
            'release_year': 2023,
            'duration': 89,
            'director': ['Pete Docter'],
            'actors': ['Chris Pratt', 'Scarlett Johansson', 'Tom Holland'],
            'description': 'Wholesome family entertainment with beautiful animation, life lessons, and adventures for all ages.',
            'rating': 4.3,
            'mood_tags': ['joyful', 'innocent', 'fun', 'wholesome', 'family-friendly']
        },
        {
            'content_id': 'mv_009',
            'title': 'Horror Night',
            'type': 'movie',
            'genre': ['Horror', 'Supernatural', 'Thriller'],
            'release_year': 2024,
            'duration': 105,
            'director': ['Ari Aster'],
            'actors': ['Anya Taylor-Joy', 'Robert Pattinson', 'Thomasin McKenzie'],
            'description': 'Spine-chilling horror that will haunt your dreams with supernatural terror and psychological scares.',
            'rating': 4.0,
            'mood_tags': ['scary', 'intense', 'dark', 'unsettling', 'terrifying']
        },
        {
            'content_id': 'mv_010',
            'title': 'Future Earth Documentary',
            'type': 'documentary',
            'genre': ['Documentary', 'Environmental', 'Educational'],
            'release_year': 2024,
            'duration': 112,
            'director': ['Werner Herzog'],
            'actors': ['David Attenborough', 'Greta Thunberg', 'Bill Gates'],
            'description': 'Thought-provoking documentary about climate change solutions and hope for the future of our planet.',
            'rating': 4.7,
            'mood_tags': ['educational', 'inspiring', 'serious', 'hopeful', 'eye-opening']
        },
        {
            'content_id': 'mv_011',
            'title': 'Space Odyssey 2025',
            'type': 'movie',
            'genre': ['Sci-Fi', 'Adventure', 'Epic'],
            'release_year': 2024,
            'duration': 156,
            'director': ['Denis Villeneuve'],
            'actors': ['Timothée Chalamet', 'Zendaya', 'Rebecca Ferguson'],
            'description': 'Epic space adventure about humanity\'s journey to the stars and the challenges they face.',
            'rating': 4.8,
            'mood_tags': ['epic', 'inspiring', 'adventurous', 'visually-stunning', 'thought-provoking']
        },
        {
            'content_id': 'mv_012',
            'title': 'Kitchen Chronicles',
            'type': 'movie',
            'genre': ['Drama', 'Food', 'Feel-good'],
            'release_year': 2023,
            'duration': 98,
            'director': ['Jon Favreau'],
            'actors': ['Adam Driver', 'Scarlett Johansson', 'John Krasinski'],
            'description': 'Heartwarming story about a chef rediscovering passion for cooking and life through food.',
            'rating': 4.1,
            'mood_tags': ['heartwarming', 'inspiring', 'cozy', 'feel-good', 'culinary']
        }
    ]
    
    with db.get_cursor() as (conn, cur):
        for content in content_data:
            cur.execute("""
                INSERT INTO content (
                    content_id, title, type, genre, release_year, duration,
                    director, actors, description, rating, mood_tags
                ) VALUES (
                    %(content_id)s, %(title)s, %(type)s, %(genre)s, %(release_year)s,
                    %(duration)s, %(director)s, %(actors)s, %(description)s, 
                    %(rating)s, %(mood_tags)s
                ) ON CONFLICT (content_id) DO NOTHING
            """, content)
    
    logger.info("✅ Content populated")

def populate_watch_history():
    """Generate realistic watch history"""
    logger.info("Populating watch history...")
    
    with db.get_cursor() as (conn, cur):
        # Get all users and content
        cur.execute("SELECT user_id FROM users")
        user_ids = [row[0] for row in cur.fetchall()]
        
        cur.execute("SELECT content_id, duration FROM content")
        content_data = cur.fetchall()
        
        weather_conditions = ['sunny', 'rainy', 'cloudy', 'snowy', 'stormy']
        devices = ['smart_tv', 'laptop', 'tablet', 'phone', 'roku', 'chromecast']
        watch_modes = ['solo', 'group']
        
        for user_id in user_ids:
            # Generate 40-100 watch history entries per user
            num_entries = random.randint(40, 100)
            
            for _ in range(num_entries):
                content_id, duration = random.choice(content_data)
                
                # Generate realistic watch time (last 8 months)
                start_time = fake.date_time_between(start_date='-8M', end_date='now')
                
                # Calculate completion with realistic distribution
                completion = random.choices(
                    [
                        random.uniform(5, 25),    # Didn't like it
                        random.uniform(25, 60),   # Partial watch
                        random.uniform(85, 100)   # Completed
                    ], 
                    weights=[0.15, 0.25, 0.6]     # 60% complete most content
                )[0]
                
                # Calculate end time based on completion
                watch_duration = int(duration * completion / 100)
                end_time = start_time + timedelta(minutes=watch_duration)
                
                # Day and time context
                day_of_week = start_time.weekday()  # 0=Monday
                hour_of_day = start_time.hour
                is_weekend = day_of_week >= 5  # Saturday, Sunday
                
                try:
                    cur.execute("""
                        INSERT INTO watch_history (
                            user_id, content_id, start_time, end_time, completion_percentage,
                            day_of_week, hour_of_day, is_weekend, context_weather,
                            context_temperature, device_id, watch_mode
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, content_id, start_time) DO NOTHING
                    """, (
                        user_id, content_id, start_time, end_time, completion,
                        day_of_week, hour_of_day, is_weekend,
                        random.choice(weather_conditions),
                        random.uniform(50, 85),  # Temperature
                        random.choice(devices),
                        random.choice(watch_modes)
                    ))
                except Exception as e:
                    logger.warning(f"Could not insert watch history entry: {e}")
                    continue
    
    logger.info("✅ Watch history populated")

def populate_content_reactions():
    """Generate user reactions"""
    logger.info("Populating content reactions...")
    
    with db.get_cursor() as (conn, cur):
        # Get watch history to generate reactions
        cur.execute("""
            SELECT DISTINCT user_id, content_id, completion_percentage 
            FROM watch_history 
            WHERE completion_percentage > 50
        """)
        watch_data = cur.fetchall()
        
        reaction_types = ['loved', 'liked', 'laughed', 'cried', 'scared', 'excited', 'bored', 'inspired']
        
        for user_id, content_id, completion in watch_data:
            # 70% chance of rating content they watched significantly
            if random.random() < 0.7:
                # Rating influenced by completion percentage
                if completion > 90:
                    rating = random.uniform(3.5, 5.0)
                    liked = True
                elif completion > 70:
                    rating = random.uniform(2.5, 4.5)
                    liked = random.choice([True, False])
                else:
                    rating = random.uniform(1.0, 3.5)
                    liked = False
                
                cur.execute("""
                    INSERT INTO content_reactions (
                        user_id, content_id, rating, liked, reaction_type
                    ) VALUES (%s, %s, %s, %s, %s)
                """, (
                    user_id, content_id, rating, liked, 
                    random.choice(reaction_types)
                ))
    
    logger.info("✅ Content reactions populated")

def populate_search_history():
    """Generate search history with prosody features"""
    logger.info("Populating search history...")
    
    with db.get_cursor() as (conn, cur):
        cur.execute("SELECT user_id FROM users")
        user_ids = [row[0] for row in cur.fetchall()]
        
        # Comprehensive search queries
        search_queries = [
            # Mood-based searches
            ("I'm feeling sad, show me something funny", "mood_based"),
            ("I'm stressed and need something relaxing", "mood_based"), 
            ("I'm bored, surprise me with something good", "mood_based"),
            ("feeling lonely tonight", "mood_based"),
            ("need motivation and inspiration", "mood_based"),
            ("something to cheer me up", "mood_based"),
            ("I'm angry, need something cathartic", "mood_based"),
            ("feeling anxious, want something calming", "mood_based"),
            
            # Genre searches
            ("comedy movies", "genre"),
            ("action thriller", "genre"),
            ("romantic drama", "genre"),
            ("sci-fi movies", "genre"),
            ("horror films", "genre"),
            ("family movies", "genre"),
            ("documentary films", "genre"),
            ("psychological thriller", "genre"),
            
            # Actor/Director searches
            ("Tom Hanks movies", "actor"),
            ("Emma Stone films", "actor"),
            ("Christopher Nolan movies", "director"),
            ("Jordan Peele films", "director"),
            ("Ryan Reynolds comedy", "actor"),
            ("Denis Villeneuve sci-fi", "director"),
            
            # Contextual searches
            ("movies for date night", "context"),
            ("something to watch with family", "context"),
            ("short movie under 90 minutes", "context"),
            ("weekend binge watch", "context"),
            ("late night viewing", "context"),
            ("feel-good movies for Sunday", "context"),
            
            # Specific content searches
            ("space movies", "theme"),
            ("cooking shows", "theme"),
            ("mind-bending plots", "theme"),
            ("uplifting stories", "theme")
        ]
        
        ai_methods = ['semantic_similarity', 'collaborative_filtering', 'mood_based', 'hybrid']
        
        for user_id in user_ids:
            num_searches = random.randint(20, 50)
            
            for _ in range(num_searches):
                query, intent = random.choice(search_queries)
                search_time = fake.date_time_between(start_date='-4M', end_date='now')
                
                # 45% voice searches, 55% text
                search_type = random.choices(['voice', 'text'], weights=[0.45, 0.55])[0]
                
                prosody_features = None
                detected_mood = None
                ai_confidence = random.uniform(0.6, 0.95)
                
                if search_type == 'voice':
                    # Generate realistic prosody features
                    prosody_features = {
                        "intensity": random.choice(["low", "medium", "high"]),
                        "tempo": random.choice(["slow", "medium", "fast"]), 
                        "pitch_level": random.choice(["low", "medium", "high"]),
                        "energy": round(random.uniform(0.1, 0.8), 3),
                        "duration": round(random.uniform(1.5, 6.0), 1),
                        "method": "enhanced_python_analysis",
                        "rms_energy": round(random.uniform(0.05, 0.4), 4),
                        "zero_crossing_rate": round(random.uniform(10, 80), 2)
                    }
                    
                    # Detect mood from query and prosody
                    if "sad" in query.lower():
                        detected_mood = "sad"
                    elif "stress" in query.lower() or "anxious" in query.lower():
                        detected_mood = "stressed"
                    elif "bored" in query.lower():
                        detected_mood = "bored"
                    elif "lonely" in query.lower():
                        detected_mood = "lonely"
                    elif "funny" in query.lower() or "cheer" in query.lower():
                        detected_mood = "happy"
                    elif "angry" in query.lower():
                        detected_mood = "angry"
                    elif "motivat" in query.lower():
                        detected_mood = "motivated"
                
                cur.execute("""
                    INSERT INTO search_history (
                        user_id, search_query, search_type, prosody_features,
                        detected_mood, search_intent, search_timestamp,
                        ai_confidence_score, recommendation_method
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, query, search_type,
                    json.dumps(prosody_features) if prosody_features else None,
                    detected_mood, intent, search_time,
                    ai_confidence, random.choice(ai_methods)
                ))
    
    logger.info("✅ Search history populated")

def populate_user_preferences():
    """Generate AI-derived user preferences"""
    logger.info("Populating user preferences...")
    
    with db.get_cursor() as (conn, cur):
        cur.execute("SELECT user_id FROM users")
        user_ids = [row[0] for row in cur.fetchall()]
        
        for user_id in user_ids:
            # Analyze user's watch history to derive preferences
            cur.execute("""
                SELECT UNNEST(c.genre) as genre, AVG(cr.rating) as avg_rating, COUNT(*) as watch_count
                FROM watch_history wh
                JOIN content c ON wh.content_id = c.content_id
                LEFT JOIN content_reactions cr ON wh.user_id = cr.user_id AND wh.content_id = cr.content_id
                WHERE wh.user_id = %s AND wh.completion_percentage > 70
                GROUP BY UNNEST(c.genre)
                HAVING COUNT(*) > 1
                ORDER BY avg_rating DESC, watch_count DESC
            """, (user_id,))
            
            genre_preferences = cur.fetchall()
            
            # Extract preferred genres (top 6)
            preferred_genres = [genre[0] for genre in genre_preferences[:6]]
            
            # Generate AI-learned mood-genre mapping
            mood_mapping = {
                "sad": ["Comedy", "Feel-good", "Family", "Romantic-Comedy"],
                "happy": ["Action", "Adventure", "Comedy", "Sci-Fi"],
                "stressed": ["Comedy", "Romance", "Feel-good", "Documentary"],
                "bored": ["Action", "Thriller", "Horror", "Mind-bending"],
                "lonely": ["Romance", "Family", "Feel-good", "Romantic-Comedy"],
                "tired": ["Romance", "Drama", "Documentary", "Feel-good"],
                "angry": ["Action", "Thriller", "Psychological"],
                "motivated": ["Adventure", "Epic", "Inspiring", "Documentary"]
            }
            
            # Generate AI-derived personality traits
            personality_traits = {
                "openness_to_new_content": round(random.uniform(0.3, 0.9), 2),
                "social_influence_factor": round(random.uniform(0.2, 0.8), 2),
                "mood_sensitivity": round(random.uniform(0.4, 0.9), 2),
                "binge_tendency": round(random.uniform(0.1, 0.8), 2),
                "genre_diversity_score": round(random.uniform(0.2, 0.8), 2),
                "quality_preference": round(random.uniform(0.5, 1.0), 2),
                "novelty_seeking": round(random.uniform(0.1, 0.9), 2)
            }
            
            # Generate a simple user embedding (random for demo, would be AI-generated in practice)
            user_embedding = [round(random.uniform(-1, 1), 4) for _ in range(384)]
            
            cur.execute("""
                INSERT INTO user_preferences (
                    user_id, preferred_genres, preferred_duration_min, preferred_duration_max,
                    mood_genre_mapping, personality_traits
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO UPDATE SET
                    preferred_genres = EXCLUDED.preferred_genres,
                    mood_genre_mapping = EXCLUDED.mood_genre_mapping,
                    personality_traits = EXCLUDED.personality_traits,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                user_id, preferred_genres, 
                random.randint(60, 90),    # min duration preference
                random.randint(120, 180),  # max duration preference
                json.dumps(mood_mapping),
                json.dumps(personality_traits)
            ))
    
    logger.info("✅ User preferences populated")

def populate_user_connections():
    """Create social connections between users"""
    logger.info("Populating user connections...")
    
    with db.get_cursor() as (conn, cur):
        # Create realistic social connections
        connections = [
            ('rky-cse', 'sarah_chen', 0.8),     # Strong connection
            ('rky-cse', 'alex_kim', 0.6),       # Moderate connection
            ('rky-cse', 'raj_patel', 0.7),      # Good connection
            ('sarah_chen', 'emma_davis', 0.9),  # Very strong
            ('sarah_chen', 'lily_wang', 0.6),
            ('mike_jones', 'carlos_ruiz', 0.8),
            ('mike_jones', 'david_brown', 0.5),
            ('emma_davis', 'zoe_taylor', 0.7),
            ('alex_kim', 'lily_wang', 0.6),
            ('alex_kim', 'zoe_taylor', 0.8),
            ('david_brown', 'raj_patel', 0.7),
            ('lily_wang', 'zoe_taylor', 0.9),
            ('carlos_ruiz', 'raj_patel', 0.4)
        ]
        
        for user_id, friend_id, strength in connections:
            # Insert bidirectional connections
            cur.execute("""
                INSERT INTO user_connections (user_id, friend_id, connection_strength)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, friend_id) DO NOTHING
            """, (user_id, friend_id, strength))
            
            cur.execute("""
                INSERT INTO user_connections (user_id, friend_id, connection_strength)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, friend_id) DO NOTHING
            """, (friend_id, user_id, strength))
    
    logger.info("✅ User connections populated")

def populate_user_contexts():
    """Generate environmental context data"""
    logger.info("Populating user contexts...")
    
    with db.get_cursor() as (conn, cur):
        cur.execute("SELECT user_id FROM users")
        user_ids = [row[0] for row in cur.fetchall()]
        
        weather_conditions = ['sunny', 'rainy', 'cloudy', 'snowy', 'stormy', 'foggy']
        festivals = ['', 'Christmas', 'New Year', 'Halloween', 'Thanksgiving', 'Valentine\'s Day', 'Independence Day']
        times_of_day = ['morning', 'afternoon', 'evening', 'night']
        
        for user_id in user_ids:
            # Generate context data for the last 45 days
            for i in range(45):
                date = datetime.now() - timedelta(days=i)
                
                cur.execute("""
                    INSERT INTO user_contexts (
                        user_id, timestamp, weather_condition, temperature, day_of_week,
                        is_weekend, active_festival, time_of_day
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    user_id, date, random.choice(weather_conditions),
                    random.randint(35, 95), date.weekday(),
                    date.weekday() >= 5,  # Weekend
                    random.choice(festivals) if random.random() < 0.08 else None,
                    random.choice(times_of_day)
                ))
    
    logger.info("✅ User contexts populated")

def create_sample_video_file():
    """Create a placeholder for sample video file"""
    logger.info("Setting up sample video file...")
    
    import os
    os.makedirs("../../data", exist_ok=True)
    os.makedirs("../../data/thumbnails", exist_ok=True)
    
    # Create a placeholder file
    sample_video_path = "../../data/sample_video.mp4"
    if not os.path.exists(sample_video_path):
        with open(sample_video_path, 'w') as f:
            f.write("# Placeholder for sample video file\n")
            f.write("# Replace this with an actual video file for testing\n")
        logger.info("📹 Created placeholder sample video file")
        logger.info("⚠️  Please replace ../../data/sample_video.mp4 with an actual video file")
    
    logger.info("✅ Sample video setup complete")

if __name__ == "__main__":
    logger.info("🚀 Starting comprehensive database population...")
    logger.info(f"📅 Current time: 2025-06-20 07:18:28 UTC")
    logger.info(f"👤 Current user: rky-cse")
    
    try:
        create_sample_video_file()
        populate_users()
        populate_content()
        populate_watch_history()
        populate_content_reactions()
        populate_search_history()
        populate_user_preferences()
        populate_user_connections()
        populate_user_contexts()
        
        logger.info("✅ Complete database population finished successfully!")
        logger.info("\n📊 Database now contains:")
        logger.info("   - 10 diverse users with realistic profiles")
        logger.info("   - 12 movies with rich metadata and mood tags")
        logger.info("   - 400-1000 watch history entries with environmental context")
        logger.info("   - Realistic user reactions and ratings")
        logger.info("   - Search history with prosody features and AI confidence scores")
        logger.info("   - AI-derived user preferences and personality traits")
        logger.info("   - Social connections between users")
        logger.info("   - Environmental context data for personalization")
        logger.info("   - Infrastructure for AI model performance tracking")
        
    except Exception as e:
        logger.error(f"❌ Error during population: {e}")
        logger.error("Please check your database connection and schema")
        raise