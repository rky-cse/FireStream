import psycopg2
import random
from datetime import datetime, timedelta

# Adjust these as needed
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "firestream_db"
DB_USER = "postgres"
DB_PASS = "postgres"

def create_tables(cur):
    """
    Create all tables from the schema (using IF NOT EXISTS to avoid errors).
    """
    # Users
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100),
        age INT,
        gender VARCHAR(20),
        location VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Content
    cur.execute("""
    CREATE TABLE IF NOT EXISTS content (
        content_id VARCHAR(50) PRIMARY KEY,
        title VARCHAR(200) NOT NULL,
        type VARCHAR(50),
        genre VARCHAR(100)[],
        release_year INT,
        duration INT,
        director VARCHAR(100)[],
        actors VARCHAR(100)[],
        description TEXT,
        rating FLOAT,
        mood_tags VARCHAR(50)[]
    );
    """)

    # Watch history
    cur.execute("""
    CREATE TABLE IF NOT EXISTS watch_history (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        content_id VARCHAR(50) REFERENCES content(content_id),
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        completion_percentage FLOAT,
        day_of_week INT,
        hour_of_day INT,
        is_weekend BOOLEAN,
        context_weather VARCHAR(50),
        context_temperature FLOAT,
        device_id VARCHAR(50),
        watch_mode VARCHAR(20),
        UNIQUE (user_id, content_id, start_time)
    );
    """)

    # Content reactions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS content_reactions (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        content_id VARCHAR(50) REFERENCES content(content_id),
        rating FLOAT,
        liked BOOLEAN,
        reaction_type VARCHAR(20),
        reaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # User connections
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_connections (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        friend_id VARCHAR(50) REFERENCES users(user_id),
        connection_strength FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE (user_id, friend_id)
    );
    """)

    # Viewing groups
    cur.execute("""
    CREATE TABLE IF NOT EXISTS viewing_groups (
        group_id VARCHAR(50) PRIMARY KEY,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_active_at TIMESTAMP
    );
    """)

    # Group members
    cur.execute("""
    CREATE TABLE IF NOT EXISTS group_members (
        id SERIAL PRIMARY KEY,
        group_id VARCHAR(50) REFERENCES viewing_groups(group_id),
        user_id VARCHAR(50) REFERENCES users(user_id),
        UNIQUE (group_id, user_id)
    );
    """)

    # Group sessions
    cur.execute("""
    CREATE TABLE IF NOT EXISTS group_sessions (
        id SERIAL PRIMARY KEY,
        group_id VARCHAR(50) REFERENCES viewing_groups(group_id),
        content_id VARCHAR(50) REFERENCES content(content_id),
        start_time TIMESTAMP,
        end_time TIMESTAMP,
        group_sentiment_score FLOAT,
        dominant_emotion VARCHAR(50)
    );
    """)

    # User contexts
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_contexts (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        weather_condition VARCHAR(50),
        temperature FLOAT,
        day_of_week INT,
        is_weekend BOOLEAN,
        active_festival VARCHAR(100),
        time_of_day VARCHAR(20)
    );
    """)

    # Content tags
    cur.execute("""
    CREATE TABLE IF NOT EXISTS content_tags (
        id SERIAL PRIMARY KEY,
        content_id VARCHAR(50) REFERENCES content(content_id),
        timestamp_start INT,
        timestamp_end INT,
        object_detected VARCHAR(100),
        scene_type VARCHAR(50),
        product_reference_id VARCHAR(50)
    );
    """)

    # Search history
    cur.execute("""
    CREATE TABLE IF NOT EXISTS search_history (
        id SERIAL PRIMARY KEY,
        user_id VARCHAR(50) REFERENCES users(user_id),
        search_query TEXT NOT NULL,
        search_type VARCHAR(20),
        prosody_features JSONB,
        detected_mood VARCHAR(50),
        search_intent VARCHAR(50),
        search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        clicked_content_ids VARCHAR(50)[]
    );
    """)

    # User preferences
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
        preferred_genres VARCHAR(100)[],
        preferred_duration_min INT DEFAULT 60,
        preferred_duration_max INT DEFAULT 180,
        mood_genre_mapping JSONB,
        personality_traits JSONB,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Indexes
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history(user_id);""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_search_history_timestamp ON search_history(search_timestamp);""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_watch_history_user_content ON watch_history(user_id, content_id);""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_content_genre ON content USING GIN(genre);""")
    cur.execute("""CREATE INDEX IF NOT EXISTS idx_content_mood_tags ON content USING GIN(mood_tags);""")

def populate_users(cur):
    """
    Insert sample user data with meaningful names.
    """
    sample_users = [
        ("USR10001", "John Smith", 25, "male", "New York"),
        ("USR10002", "Alice Johnson", 30, "female", "Los Angeles"),
        ("USR10003", "Bob Davis", 42, "male", "Chicago"),
        ("USR10004", "Emily Brown", 35, "female", "London"),
        ("USR10005", "David Wilson", 28, "male", "Paris"),
        ("USR10006", "Sophia Miller", 45, "female", "Berlin"),
        ("USR10007", "Daniel Garcia", 22, "male", "Madrid"),
        ("USR10008", "Linda Lee", 31, "female", "Sydney"),
        ("USR10009", "Michael Taylor", 29, "male", "Toronto"),
        ("USR10010", "Sarah Anderson", 33, "female", "San Francisco"),
    ]

    for user in sample_users:
        cur.execute(
            """
            INSERT INTO users (user_id, name, age, gender, location)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING;
            """,
            user
        )

def populate_content(cur):
    """
    Insert sample content data with meaningful titles.
    Reusing the same video file path 'video/sample.mp4'
    but varying the metadata.
    """
    sample_content = [
        {
            "content_id": "MOV10001",
            "title": "The Mysterious Adventure",
            "type": "movie",
            "genre": ["Adventure", "Mystery"],
            "release_year": 2012,
            "duration": 120,
            "director": ["Christopher Green"],
            "actors": ["James White", "Anne Grey"],
            "description": "A suspenseful and thrilling adventure. video at video/sample.mp4",
            "rating": 4.2,
            "mood_tags": ["suspenseful", "thrilling"]
        },
        {
            "content_id": "MOV10002",
            "title": "Love in Springtime",
            "type": "movie",
            "genre": ["Romance", "Drama"],
            "release_year": 2019,
            "duration": 90,
            "director": ["Patricia Bloom"],
            "actors": ["Kate Peterson", "Evan Stone"],
            "description": "A heartwarming tale of finding love unexpected. video at video/sample.mp4",
            "rating": 3.8,
            "mood_tags": ["heartwarming", "sad"]
        },
        {
            "content_id": "MOV10003",
            "title": "Galaxy Explorers",
            "type": "movie",
            "genre": ["Sci-Fi", "Action"],
            "release_year": 2022,
            "duration": 110,
            "director": ["Zoe Sterling"],
            "actors": ["Mark Brown", "Olivia Wilde"],
            "description": "Interstellar missions and cosmic conflicts. video at video/sample.mp4",
            "rating": 4.5,
            "mood_tags": ["thrilling", "intense"]
        },
        {
            "content_id": "MOV10004",
            "title": "Hilarious Night Out",
            "type": "movie",
            "genre": ["Comedy"],
            "release_year": 2016,
            "duration": 100,
            "director": ["Ben Miller"],
            "actors": ["Sara Long", "Chris Porter"],
            "description": "Comedy film with a group of friends on a wild night. video at video/sample.mp4",
            "rating": 4.0,
            "mood_tags": ["funny", "lighthearted"]
        },
        {
            "content_id": "MOV10005",
            "title": "Whispers of the Past",
            "type": "movie",
            "genre": ["Drama"],
            "release_year": 2021,
            "duration": 115,
            "director": ["Jonathan Reeves"],
            "actors": ["Claire Adams", "Tom Green"],
            "description": "A drama focusing on life choices and reconciliations. video at video/sample.mp4",
            "rating": 4.1,
            "mood_tags": ["sad", "thoughtful"]
        },
        {
            "content_id": "MOV10006",
            "title": "Furry Friends",
            "type": "movie",
            "genre": ["Family", "Comedy"],
            "release_year": 2015,
            "duration": 85,
            "director": ["Emily Clark"],
            "actors": ["Mia Kelly", "Owen Brooks"],
            "description": "Family-friendly comedy about talking pets. video at video/sample.mp4",
            "rating": 3.9,
            "mood_tags": ["funny", "lighthearted"]
        },
        {
            "content_id": "MOV10007",
            "title": "Shadows in the Dark",
            "type": "movie",
            "genre": ["Horror"],
            "release_year": 2020,
            "duration": 95,
            "director": ["Allison Graye"],
            "actors": ["Janet Morris", "Ethan Scott"],
            "description": "A chilling horror film. video at video/sample.mp4",
            "rating": 3.5,
            "mood_tags": ["scary", "dark"]
        },
        {
            "content_id": "MOV10008",
            "title": "City Life Chronicles",
            "type": "show",
            "genre": ["Drama", "Comedy"],
            "release_year": 2018,
            "duration": 40,  # per episode
            "director": ["Various Directors"],
            "actors": ["Multiple Cast Members"],
            "description": "TV show depicting humorous and dramatic city life stories. video at video/sample.mp4",
            "rating": 4.0,
            "mood_tags": ["funny", "sad"]
        },
        {
            "content_id": "MOV10009",
            "title": "Mystic Tales",
            "type": "movie",
            "genre": ["Fantasy", "Adventure"],
            "release_year": 2017,
            "duration": 130,
            "director": ["Barbara Knight"],
            "actors": ["Adam King", "Diana Rose"],
            "description": "A magical fantasy adventure in a mysterious realm. video at video/sample.mp4",
            "rating": 4.3,
            "mood_tags": ["thrilling", "imaginative"]
        },
        {
            "content_id": "MOV10010",
            "title": "Everyday Heroes",
            "type": "documentary",
            "genre": ["Documentary"],
            "release_year": 2023,
            "duration": 70,
            "director": ["Ryan Carter"],
            "actors": ["Real People"],
            "description": "Documentary about unsung heroes. video at video/sample.mp4",
            "rating": 4.6,
            "mood_tags": ["heartwarming", "inspiring"]
        },
    ]

    for entry in sample_content:
        cur.execute(
            """
            INSERT INTO content (
                content_id, title, type, genre, release_year, duration,
                director, actors, description, rating, mood_tags
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (content_id) DO NOTHING;
            """,
            (
                entry["content_id"],
                entry["title"],
                entry["type"],
                entry["genre"],
                entry["release_year"],
                entry["duration"],
                entry["director"],
                entry["actors"],
                entry["description"],
                entry["rating"],
                entry["mood_tags"]
            )
        )

def populate_watch_history(cur, records=30):
    """
    Insert dummy watch_history records referencing the inserted users and content.
    """
    # Grab user_ids
    cur.execute("SELECT user_id FROM users;")
    user_ids = [row[0] for row in cur.fetchall()]

    # Grab content_ids
    cur.execute("SELECT content_id FROM content;")
    content_ids = [row[0] for row in cur.fetchall()]

    weathers = ["sunny", "rainy", "cloudy", "snowy"]
    devices = ["TV_living_room", "Laptop", "Smartphone", "Tablet"]
    watch_modes = ["solo", "group"]

    for _ in range(records):
        user_id = random.choice(user_ids)
        content_id = random.choice(content_ids)

        start_time = datetime.now() - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))
        duration = timedelta(minutes=random.randint(30, 120))
        end_time = start_time + duration if random.choice([True, False]) else None
        completion_percentage = round(random.uniform(0, 100), 2)

        day_of_week = start_time.weekday()  # Monday=0, Sunday=6
        hour_of_day = start_time.hour
        is_weekend = (day_of_week >= 5)

        weather = random.choice(weathers)
        temperature = round(random.uniform(-5, 35), 1)
        device_id = random.choice(devices)
        watch_mode = random.choice(watch_modes)

        try:
            cur.execute(
                """
                INSERT INTO watch_history (
                    user_id, content_id, start_time, end_time, completion_percentage,
                    day_of_week, hour_of_day, is_weekend, context_weather, context_temperature,
                    device_id, watch_mode
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT DO NOTHING;
                """,
                (
                    user_id, content_id, start_time, end_time, completion_percentage,
                    day_of_week, hour_of_day, is_weekend, weather, temperature,
                    device_id, watch_mode
                )
            )
        except Exception as ex:
            print("Error inserting watch_history:", ex)

def main():
    print("Connecting to the database...")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    conn.autocommit = True

    with conn.cursor() as cur:
        print("Creating tables...")
        create_tables(cur)

        print("Populating users with meaningful names...")
        populate_users(cur)

        print("Populating content with meaningful titles...")
        populate_content(cur)

        print("Populating watch history...")
        populate_watch_history(cur, records=50)

    conn.close()
    print("Done. Database populated with sample data.")

if __name__ == "__main__":
    main()