import numpy as np
from datetime import datetime, timedelta

class ContextualReranker:
    """
    Reranks base recommendations based on:
    1. Current context (time, weather, etc.)
    2. Group dynamics
    3. Friend activity
    """
    
    def __init__(self, content_df, watch_history_df):
        self.content_df = content_df
        self.watch_history_df = watch_history_df
        # Index content by ID for faster lookups
        self.content_map = {row['content_id']: row for _, row in content_df.iterrows()}
    
    def rerank_for_context(self, recommendations, context_data):
        """
        Rerank based on contextual factors
        
        Args:
            recommendations: List of (content_id, score) tuples
            context_data: Dictionary with Context Time, Weather, Festivals
        
        Returns:
            Reranked list of (content_id, score) tuples
        """
        current_hour = int(context_data['Context Time']['time'].split(':')[0])
        is_weekend = context_data['Context Time']['is_weekend']
        weather = context_data['Weather']['description'].lower()
        temperature = context_data['Weather']['temperature']
        
        # Define context boosts
        boosts = {}
        
        # Time-based boosts
        if 6 <= current_hour < 10:  # Morning
            boosts['family'] = 1.2
            boosts['news'] = 1.3
            boosts['documentary'] = 1.15
        elif 10 <= current_hour < 15:  # Mid-day
            boosts['comedy'] = 1.15
            boosts['talk_show'] = 1.2
        elif 15 <= current_hour < 19:  # Afternoon
            boosts['family'] = 1.15
            boosts['animation'] = 1.15
        elif 19 <= current_hour < 23:  # Prime time
            boosts['drama'] = 1.2
            boosts['action'] = 1.15
            if is_weekend:
                boosts['thriller'] = 1.2
                boosts['horror'] = 1.25
        else:  # Late night
            boosts['thriller'] = 1.3
            boosts['horror'] = 1.3
            boosts['comedy'] = 1.2
        
        # Weather-based boosts
        if 'rain' in weather or 'storm' in weather:
            boosts['drama'] = boosts.get('drama', 1.0) * 1.15
            boosts['thriller'] = boosts.get('thriller', 1.0) * 1.1
        elif 'sunny' in weather or 'clear' in weather:
            boosts['comedy'] = boosts.get('comedy', 1.0) * 1.1
            boosts['adventure'] = boosts.get('adventure', 1.0) * 1.15
        elif 'cloud' in weather:
            boosts['drama'] = boosts.get('drama', 1.0) * 1.05
        
        if temperature > 30:  # Hot weather
            boosts['animation'] = boosts.get('animation', 1.0) * 1.1  # Indoor activity
        elif temperature < 15:  # Cold weather
            boosts['family'] = boosts.get('family', 1.0) * 1.1  # Cozy watching
        
        # Festival boosts
        current_date = datetime.strptime(context_data['Context Time']['date'], '%Y-%m-%d')
        for festival in context_data['Festivals']:
            festival_date = datetime.strptime(festival['date'], '%Y-%m-%d')
            days_to_festival = (festival_date - current_date).days
            
            if -1 <= days_to_festival <= 3:  # Around festival time
                festival_name = festival['name'].lower()
                
                if 'christmas' in festival_name:
                    boosts['family'] = boosts.get('family', 1.0) * 1.3
                    boosts['animation'] = boosts.get('animation', 1.0) * 1.2
                elif 'independence day' in festival_name:
                    boosts['action'] = boosts.get('action', 1.0) * 1.2
                    boosts['documentary'] = boosts.get('documentary', 1.0) * 1.3
                elif 'diwali' in festival_name:
                    boosts['family'] = boosts.get('family', 1.0) * 1.3
                elif any(h in festival_name for h in ['halloween', 'horror']):
                    boosts['horror'] = boosts.get('horror', 1.0) * 1.4
                    boosts['thriller'] = boosts.get('thriller', 1.0) * 1.2
        
        # Apply boosts to scores
        boosted_recs = []
        for content_id, score in recommendations:
            if content_id in self.content_map:
                content = self.content_map[content_id]
                
                # Start with original score
                boosted_score = score
                
                # Apply relevant genre boosts
                for genre in content['genre']:
                    genre_key = genre.lower().replace(' ', '_')
                    if genre_key in boosts:
                        boosted_score *= boosts[genre_key]
                
                boosted_recs.append((content_id, boosted_score))
            else:
                # Keep original score if content not found
                boosted_recs.append((content_id, score))
        
        # Sort by boosted score
        return sorted(boosted_recs, key=lambda x: x[1], reverse=True)
    
    def rerank_for_group(self, recommendations, user_ids, group_weights=None):
        """
        Rerank recommendations for a group of users
        
        Args:
            recommendations: Dict mapping user_id to list of (content_id, score) tuples
            user_ids: List of user IDs in the group
            group_weights: Optional dict mapping user_id to weight in group decision
            
        Returns:
            List of (content_id, aggregate_score) tuples
        """
        if not recommendations or not user_ids:
            return []
        
        # Default to equal weighting if not specified
        if group_weights is None:
            group_weights = {user_id: 1.0/len(user_ids) for user_id in user_ids}
        
        # Normalize weights to sum to 1
        total_weight = sum(group_weights.values())
        normalized_weights = {uid: w/total_weight for uid, w in group_weights.items()}
        
        # Aggregate scores across all users
        content_scores = {}
        
        for user_id in user_ids:
            if user_id not in recommendations:
                continue
                
            user_weight = normalized_weights.get(user_id, 0)
            for content_id, score in recommendations[user_id]:
                if content_id in content_scores:
                    content_scores[content_id] += score * user_weight
                else:
                    content_scores[content_id] = score * user_weight
        
        # Return sorted by aggregate score
        return sorted(
            [(content_id, score) for content_id, score in content_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
    
    def generate_friend_activity_notifications(self, user_id, friends_ids, 
                                               watch_history_df, content_reactions_df,
                                               days_threshold=7, min_completion=0.5):
        """
        Generate notifications about friend activity
        
        Args:
            user_id: User to generate notifications for
            friends_ids: List of friend user IDs
            watch_history_df: DataFrame of watch history
            content_reactions_df: DataFrame of content reactions
            days_threshold: How recent the activity should be (in days)
            min_completion: Minimum watch completion to consider
            
        Returns:
            List of notification dicts with content_id, friend_ids, and type
        """
        now = datetime.now()
        cutoff_date = now - timedelta(days=days_threshold)
        
        # Get user's watch history (to avoid notifications for content they've already seen)
        user_watched = set(watch_history_df[
            (watch_history_df['user_id'] == user_id) & 
            (watch_history_df['completion_percentage'] >= min_completion * 100)
        ]['content_id'].unique())
        
        # Get recent friend watch activity
        recent_friend_watches = watch_history_df[
            (watch_history_df['user_id'].isin(friends_ids)) &
            (watch_history_df['start_time'] >= cutoff_date) &
            (watch_history_df['completion_percentage'] >= min_completion * 100)
        ]
        
        # Get friend reactions (likes, high ratings)
        friend_reactions = content_reactions_df[
            (content_reactions_df['user_id'].isin(friends_ids)) &
            (content_reactions_df['reaction_timestamp'] >= cutoff_date) &
            ((content_reactions_df['liked'] == True) | 
             (content_reactions_df['rating'] >= 4.0) |
             (content_reactions_df['reaction_type'].isin(['loved', 'amazing'])))
        ]
        
        # Group watched content by friends
        watch_groups = {}
        for _, row in recent_friend_watches.iterrows():
            if row['content_id'] not in user_watched:  # Skip if user already watched
                if row['content_id'] not in watch_groups:
                    watch_groups[row['content_id']] = []
                watch_groups[row['content_id']].append(row['user_id'])
        
        # Group reactions by content
        reaction_groups = {}
        for _, row in friend_reactions.iterrows():
            if row['content_id'] not in user_watched:  # Skip if user already watched
                if row['content_id'] not in reaction_groups:
                    reaction_groups[row['content_id']] = []
                reaction_groups[row['content_id']].append({
                    'user_id': row['user_id'],
                    'rating': row.get('rating'),
                    'liked': row.get('liked'),
                    'reaction_type': row.get('reaction_type')
                })
        
        # Generate notifications
        notifications = []
        
        # Watched by multiple friends (social proof)
        for content_id, friend_list in watch_groups.items():
            if len(friend_list) >= 2:  # At least 2 friends watched
                notifications.append({
                    'content_id': content_id,
                    'friend_ids': friend_list,
                    'type': 'watched_by_friends',
                    'message': f"{len(friend_list)} friends watched this recently"
                })
                
        # Highly rated or liked content
        for content_id, reactions in reaction_groups.items():
            if len(reactions) > 0:
                # Count different types of positive reactions
                high_ratings = sum(1 for r in reactions if r.get('rating', 0) >= 4.5)
                likes = sum(1 for r in reactions if r.get('liked') == True)
                loves = sum(1 for r in reactions if r.get('reaction_type') == 'loved')
                
                if high_ratings >= 2 or likes >= 3 or loves >= 2:
                    friend_ids = list(set(r['user_id'] for r in reactions))
                    
                    # Determine the most appropriate notification type
                    if high_ratings >= 2:
                        notif_type = 'highly_rated_by_friends'
                        message = f"Rated {high_ratings}+ stars by {len(friend_ids)} friends"
                    elif loves >= 2:
                        notif_type = 'loved_by_friends'
                        message = f"Loved by {len(friend_ids)} friends"
                    else:
                        notif_type = 'liked_by_friends'
                        message = f"Liked by {len(friend_ids)} friends"
                    
                    notifications.append({
                        'content_id': content_id,
                        'friend_ids': friend_ids,
                        'type': notif_type,
                        'message': message
                    })
        
        return notifications