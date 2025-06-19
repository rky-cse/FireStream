import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

def build_user_embedding(user_id, watch_history_df, user_features_df, context_data):
    """
    Build a rich user embedding that incorporates:
    - Historical watch patterns
    - Demographic information
    - Current context (time, weather, festivals)
    """
    # Extract user watch history features
    user_history = watch_history_df[watch_history_df['user_id'] == user_id]
    
    # Genre preferences (weighted by completion percentage and recency)
    genre_preferences = {}
    now = datetime.now()
    
    for _, row in user_history.iterrows():
        content_genres = get_content_genres(row['content_id'])
        days_old = (now - row['start_time']).days
        recency_weight = 1.0 / (1.0 + 0.1 * days_old)  # Decay weight for older watches
        completion_weight = row['completion_percentage'] / 100.0
        weight = recency_weight * completion_weight
        
        for genre in content_genres:
            if genre in genre_preferences:
                genre_preferences[genre] += weight
            else:
                genre_preferences[genre] = weight
    
    # Time pattern features
    time_patterns = {}
    for hour in range(24):
        hour_watches = user_history[user_history['hour_of_day'] == hour].shape[0]
        time_patterns[f'hour_{hour}'] = hour_watches / max(user_history.shape[0], 1)
    
    for day in range(7):
        day_watches = user_history[user_history['day_of_week'] == day].shape[0]
        time_patterns[f'day_{day}'] = day_watches / max(user_history.shape[0], 1)
    
    # Current context encoding
    context_features = {
        'is_weekend': 1.0 if context_data['Context Time']['is_weekend'] else 0.0,
        'current_hour': float(context_data['Context Time']['time'].split(':')[0]),
        'temperature': context_data['Weather']['temperature'] / 50.0,  # Normalize
        'is_rainy': 1.0 if 'rain' in context_data['Weather']['description'].lower() else 0.0,
        'is_cloudy': 1.0 if 'cloud' in context_data['Weather']['description'].lower() else 0.0,
        'is_sunny': 1.0 if 'sun' in context_data['Weather']['description'].lower() or 'clear' in context_data['Weather']['description'].lower() else 0.0,
    }
    
    # Festival context - add any upcoming festivals (within 3 days)
    current_date = datetime.strptime(context_data['Context Time']['date'], '%Y-%m-%d')
    for festival in context_data['Festivals']:
        festival_date = datetime.strptime(festival['date'], '%Y-%m-%d')
        days_to_festival = (festival_date - current_date).days
        if 0 <= days_to_festival <= 3:
            context_features[f'festival_{festival["name"].replace(" ", "_")}'] = 1.0 - (days_to_festival / 4.0)
            
    # Combine all features into a single embedding
    embedding = {**genre_preferences, **time_patterns, **context_features}
    
    # If we have demographic info, include it
    if user_id in user_features_df.index:
        user_demo = user_features_df.loc[user_id]
        # One-hot encode categorical features
        if 'gender' in user_demo:
            embedding[f'gender_{user_demo["gender"]}'] = 1.0
        if 'age' in user_demo:
            # Age buckets
            age_buckets = ['0-18', '19-24', '25-34', '35-44', '45-54', '55+']
            for bucket in age_buckets:
                low, high = bucket.split('-') if '-' in bucket else (bucket.replace('+', ''), '100')
                if int(low) <= user_demo['age'] <= int(high):
                    embedding[f'age_{bucket}'] = 1.0
    
    return embedding

def build_content_embedding(content_id, content_df, content_tags_df):
    """
    Build content embedding with genre, mood, and scene information
    """
    content_row = content_df[content_df['content_id'] == content_id].iloc[0]
    
    # Basic content features
    embedding = {
        'type': content_row['type'],
        'release_year': content_row['release_year'] / 2025.0,  # Normalize by current year
        'duration': content_row['duration'] / 180.0,  # Normalize by max expected duration
        'rating': content_row['rating'] / 5.0,  # Normalize assuming 5-star scale
    }
    
    # Add genres as one-hot features
    for genre in content_row['genre']:
        embedding[f'genre_{genre}'] = 1.0
    
    # Add mood tags
    for mood in content_row['mood_tags']:
        embedding[f'mood_{mood}'] = 1.0
    
    # Add scene and object information
    content_objects = content_tags_df[content_tags_df['content_id'] == content_id]
    object_counts = content_objects['object_detected'].value_counts()
    scene_counts = content_objects['scene_type'].value_counts()
    
    # Normalize object and scene counts
    for obj, count in object_counts.items():
        embedding[f'object_{obj}'] = min(count / 10.0, 1.0)  # Cap at 1.0
        
    for scene, count in scene_counts.items():
        embedding[f'scene_{scene}'] = min(count / 5.0, 1.0)  # Cap at 1.0
    
    return embedding

def get_content_genres(content_id, content_df):
    """Helper to get genres for a content ID"""
    if content_id in content_df.index:
        return content_df.loc[content_id]['genre']
    return []