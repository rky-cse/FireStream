-- User profiles
CREATE TABLE users (
    user_id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(100),
    age INT,
    gender VARCHAR(20),
    location VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Content metadata
CREATE TABLE content (
    content_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(200) NOT NULL,
    type VARCHAR(50), -- movie, show, episode, etc.
    genre VARCHAR(100)[],
    release_year INT,
    duration INT, -- in minutes
    director VARCHAR(100)[],
    actors VARCHAR(100)[],
    description TEXT,
    rating FLOAT,
    mood_tags VARCHAR(50)[] -- e.g., funny, suspenseful, heartwarming
);

-- Watch history with detailed analytics
CREATE TABLE watch_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    content_id VARCHAR(50) REFERENCES content(content_id),
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    completion_percentage FLOAT,
    day_of_week INT, -- 0-6 (Sunday-Saturday)
    hour_of_day INT, -- 0-23
    is_weekend BOOLEAN,
    context_weather VARCHAR(50),
    context_temperature FLOAT,
    device_id VARCHAR(50),
    watch_mode VARCHAR(20), -- solo or group
    UNIQUE (user_id, content_id, start_time)
);

-- User ratings and reactions
CREATE TABLE content_reactions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    content_id VARCHAR(50) REFERENCES content(content_id),
    rating FLOAT, -- 1-5 stars
    liked BOOLEAN,
    reaction_type VARCHAR(20), -- loved, laughed, scared, etc.
    reaction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Social connections (friend relationships)
CREATE TABLE user_connections (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    friend_id VARCHAR(50) REFERENCES users(user_id),
    connection_strength FLOAT, -- calculated based on similarity
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (user_id, friend_id)
);

-- Group sessions
CREATE TABLE viewing_groups (
    group_id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active_at TIMESTAMP
);

-- Users in groups
CREATE TABLE group_members (
    id SERIAL PRIMARY KEY,
    group_id VARCHAR(50) REFERENCES viewing_groups(group_id),
    user_id VARCHAR(50) REFERENCES users(user_id),
    UNIQUE (group_id, user_id)
);

-- Group watch sessions
CREATE TABLE group_sessions (
    id SERIAL PRIMARY KEY,
    group_id VARCHAR(50) REFERENCES viewing_groups(group_id),
    content_id VARCHAR(50) REFERENCES content(content_id),
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    group_sentiment_score FLOAT, -- aggregated sentiment
    dominant_emotion VARCHAR(50) -- most prevalent emotion during session
);

-- Contextual data tracking
CREATE TABLE user_contexts (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    weather_condition VARCHAR(50),
    temperature FLOAT,
    day_of_week INT,
    is_weekend BOOLEAN,
    active_festival VARCHAR(100),
    time_of_day VARCHAR(20) -- morning, afternoon, evening, night
);

-- Scene and object tags for content
CREATE TABLE content_tags (
    id SERIAL PRIMARY KEY,
    content_id VARCHAR(50) REFERENCES content(content_id),
    timestamp_start INT, -- timestamp in seconds from start
    timestamp_end INT,
    object_detected VARCHAR(100),
    scene_type VARCHAR(50),
    product_reference_id VARCHAR(50) -- for e-commerce linkage
);


-- Search history with prosody features
CREATE TABLE search_history (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(50) REFERENCES users(user_id),
    search_query TEXT NOT NULL,
    search_type VARCHAR(20), -- 'text' or 'voice'
    prosody_features JSONB, -- voice characteristics when using voice search
    detected_mood VARCHAR(50), -- sad, happy, stressed, etc.
    search_intent VARCHAR(50), -- genre, mood_based, actor, title, etc.
    search_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    clicked_content_ids VARCHAR(50)[] -- which content user clicked from results
);

-- User preferences (derived from behavior)
CREATE TABLE user_preferences (
    user_id VARCHAR(50) PRIMARY KEY REFERENCES users(user_id),
    preferred_genres VARCHAR(100)[],
    preferred_duration_min INT DEFAULT 60,
    preferred_duration_max INT DEFAULT 180,
    mood_genre_mapping JSONB, -- e.g., {"sad": ["comedy", "feel-good"]}
    personality_traits JSONB, -- introvert/extrovert, openness to new content, etc.
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for performance
CREATE INDEX idx_search_history_user_id ON search_history(user_id);
CREATE INDEX idx_search_history_timestamp ON search_history(search_timestamp);
CREATE INDEX idx_watch_history_user_content ON watch_history(user_id, content_id);
CREATE INDEX idx_content_genre ON content USING GIN(genre);
CREATE INDEX idx_content_mood_tags ON content USING GIN(mood_tags);