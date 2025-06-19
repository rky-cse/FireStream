import numpy as np # type: ignore
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pandas as pd

def calculate_user_similarity(user_id, all_user_ids, watch_history_df, content_df):
    """
    Calculate similarity between a user and all other users based on watch history
    
    Args:
        user_id: Target user ID
        all_user_ids: List of all user IDs to compare against
        watch_history_df: DataFrame of watch history
        content_df: DataFrame of content metadata
        
    Returns:
        Dictionary mapping user_id to similarity score
    """
    # Get all unique content IDs
    all_content_ids = content_df['content_id'].unique()
    content_to_idx = {c_id: idx for idx, c_id in enumerate(all_content_ids)}
    
    # Create user-content matrix (weighted by completion percentage)
    user_vectors = {}
    
    # Process all users
    for uid in all_user_ids:
        user_watches = watch_history_df[watch_history_df['user_id'] == uid]
        vector = np.zeros(len(all_content_ids))
        
        for _, row in user_watches.iterrows():
            if row['content_id'] in content_to_idx:
                idx = content_to_idx[row['content_id']]
                # Weight by completion
                weight = row['completion_percentage'] / 100.0
                vector[idx] = weight
        
        user_vectors[uid] = vector
    
    # Calculate cosine similarity
    target_vector = user_vectors.get(user_id, np.zeros(len(all_content_ids)))
    similarity_scores = {}
    
    for uid, vector in user_vectors.items():
        if uid == user_id:
            continue
            
        # Calculate cosine similarity
        # Handle zero vectors (no watch overlap)
        if np.sum(target_vector) == 0 or np.sum(vector) == 0:
            similarity_scores[uid] = 0.0
        else:
            similarity = cosine_similarity([target_vector], [vector])[0][0]
            similarity_scores[uid] = similarity
    
    return similarity_scores

def cluster_users_by_watch_patterns(watch_history_df, content_df, users_df, n_clusters=5):
    """
    Cluster users based on watch patterns and demographics
    
    Args:
        watch_history_df: DataFrame of watch history
        content_df: DataFrame of content metadata
        users_df: DataFrame of user data
        n_clusters: Number of clusters to form
        
    Returns:
        DataFrame with user_id and cluster assignments
    """
    from sklearn.cluster import KMeans
    
    # Create user-genre matrix
    all_genres = []
    for genres in content_df['genre']:
        all_genres.extend(genres)
    unique_genres = list(set(all_genres))
    
    # Map content to genres
    content_genres = {}
    for _, row in content_df.iterrows():
        content_genres[row['content_id']] = row['genre']
    
    # Create feature matrix
    user_ids = []
    feature_rows = []
    
    for user_id in watch_history_df['user_id'].unique():
        user_watches = watch_history_df[watch_history_df['user_id'] == user_id]
        
        # Skip users with too little history
        if len(user_watches) < 3:
            continue
        
        # Genre preferences
        genre_counts = {genre: 0 for genre in unique_genres}
        total_watch_time = 0
        
        for _, row in user_watches.iterrows():
            content_id = row['content_id']
            if content_id in content_genres:
                watch_weight = row['completion_percentage'] / 100.0
                for genre in content_genres[content_id]:
                    genre_counts[genre] += watch_weight
                total_watch_time += watch_weight
        
        # Normalize genre counts
        if total_watch_time > 0:
            genre_prefs = [genre_counts[g] / total_watch_time for g in unique_genres]
        else:
            genre_prefs = [0] * len(unique_genres)
        
        # Time patterns (hour of day, day of week)
        hour_counts = [0] * 24
        day_counts = [0] * 7
        
        for _, row in user_watches.iterrows():
            hour = row['hour_of_day']
            day = row['day_of_week']
            hour_counts[hour] += 1
            day_counts[day] += 1
        
        # Normalize time patterns
        total_watches = len(user_watches)
        if total_watches > 0:
            hour_prefs = [h / total_watches for h in hour_counts]
            day_prefs = [d / total_watches for d in day_counts]
        else:
            hour_prefs = [0] * 24
            day_prefs = [0] * 7
        
        # Combine features
        features = genre_prefs + hour_prefs + day_prefs
        
        # Add demographic features if available
        if user_id in users_df['user_id'].values:
            user_row = users_df[users_df['user_id'] == user_id].iloc[0]
            
            # Age bucket (normalized)
            age = user_row.get('age', 30)
            age_normalized = age / 100.0  # Simple normalization
            features.append(age_normalized)
            
            # Gender (one-hot)
            gender = user_row.get('gender', 'unknown')
            gender_female = 1.0 if gender.lower() == 'female' else 0.0
            gender_male = 1.0 if gender.lower() == 'male' else 0.0
            features.extend([gender_female, gender_male])
        else:
            # Default values if user not in users_df
            features.extend([0.3, 0, 0])  # Default age 30, unknown gender
        
        user_ids.append(user_id)
        feature_rows.append(features)
    
    # Perform clustering
    if len(feature_rows) < n_clusters:
        # Not enough data for requested clusters
        return pd.DataFrame({'user_id': user_ids, 'cluster': [0] * len(user_ids)})
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(feature_rows)
    
    # Return user-cluster mapping
    return pd.DataFrame({'user_id': user_ids, 'cluster': clusters})

def calculate_social_group_weights(user_id, friends_ids, watch_history_df, content_df, users_df):
    """
    Calculate weights for each friend in a group based on similarity and recent interactions
    
    Args:
        user_id: Main user ID
        friends_ids: List of friend user IDs
        watch_history_df: DataFrame of watch history
        content_df: DataFrame of content metadata
        users_df: DataFrame of user data
        
    Returns:
        Dictionary mapping user_id to weight (including main user)
    """
    # Calculate base similarity scores
    similarity_scores = calculate_user_similarity(
        user_id, 
        [user_id] + friends_ids, 
        watch_history_df, 
        content_df
    )
    
    # Add the main user with maximum similarity (to self)
    similarity_scores[user_id] = 1.0
    
    # Get demographic info
    user_demo = users_df[users_df['user_id'] == user_id].iloc[0] if user_id in users_df['user_id'].values else {}
    friend_demos = {}
    for fid in friends_ids:
        if fid in users_df['user_id'].values:
            friend_demos[fid] = users_df[users_df['user_id'] == fid].iloc[0]
    
    # Adjust weights based on demographics (optional)
    weights = {uid: score for uid, score in similarity_scores.items()}
    
    # Calculate age similarity boost if data available
    if 'age' in user_demo:
        user_age = user_demo['age']
        for fid in friends_ids:
            if fid in friend_demos and 'age' in friend_demos[fid]:
                friend_age = friend_demos[fid]['age']
                # Small boost for similar age
                age_diff = abs(user_age - friend_age)
                age_boost = 1.0 + (1.0 / (1.0 + 0.1 * age_diff)) * 0.2  # Max 20% boost for same age
                weights[fid] = weights.get(fid, 0) * age_boost
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {uid: w/total_weight for uid, w in weights.items()}
        return normalized_weights
    else:
        # Equal weights if no data
        equal_weight = 1.0 / (len(friends_ids) + 1)
        return {uid: equal_weight for uid in [user_id] + friends_ids}