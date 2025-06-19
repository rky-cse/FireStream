from fastapi import FastAPI, HTTPException, Depends, Query
from typing import List, Optional, Dict, Any
import os
import sys
from datetime import datetime
import pandas as pd
import json

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from recommender.trainer.train_lightfm import FireTVRecommender
from recommender.reranker.rerank import ContextualReranker
from recommender.shared.utils import calculate_social_group_weights

app = FastAPI(title="FireTV AI Recommender API")

# Load models and data
def get_recommender():
    """Dependency to load the recommender model"""
    try:
        recommender = FireTVRecommender.load_model()
        return recommender
    except Exception as e:
        # In production, implement proper logging
        print(f"Error loading recommender model: {e}")
        # Return a fallback model or raise appropriate error
        raise HTTPException(status_code=500, detail="Recommender model not available")

def get_db_connection():
    """Database connection pool (simplified example)"""
    # In production, use a proper connection pool
    conn = {
        'watch_history': pd.read_csv("data/watch_history.csv"),
        'content': pd.read_csv("data/content.csv"),
        'users': pd.read_csv("data/users.csv"),
        'content_reactions': pd.read_csv("data/content_reactions.csv"),
        'user_connections': pd.read_csv("data/user_connections.csv")
    }
    return conn

def get_reranker(db = Depends(get_db_connection)):
    """Dependency to load the reranker"""
    return ContextualReranker(db['content'], db['watch_history'])

@app.get("/")
def read_root():
    return {"message": "FireTV AI Recommender API"}

@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: str,
    group_mode: bool = False,
    friend_ids: Optional[List[str]] = Query(None),
    num_recommendations: int = 20,
    context: Optional[Dict[str, Any]] = None,
    recommender = Depends(get_recommender),
    reranker = Depends(get_reranker),
    db = Depends(get_db_connection)
):
    """Get personalized recommendations for a user or group"""
    try:
        # If context not provided, use basic context
        if context is None:
            now = datetime.now()
            context = {
                "Context Time": {
                    "date": now.strftime("%Y-%m-%d"),
                    "time": now.strftime("%H:%M:%S"),
                    "is_weekend": now.weekday() >= 5
                },
                "Weather": {
                    "main": "Clear",
                    "description": "clear sky",
                    "temperature": 25.0,
                    "humidity": 50,
                    "wind_speed": 2.0
                },
                "Festivals": []
            }
        
        # Generate base recommendations
        if group_mode and friend_ids:
            # Group recommendations
            all_users = [user_id] + friend_ids
            
            # Get weights for the group
            group_weights = calculate_social_group_weights(
                user_id, friend_ids, db['watch_history'], db['content'], db['users']
            )
            
            # Get individual recommendations for each group member
            user_recs = {}
            for uid in all_users:
                try:
                    # Get more recommendations for each user to have a good pool
                    user_recs[uid] = recommender.predict_for_user(uid, top_n=num_recommendations * 2)
                except ValueError:
                    # Skip users not in the model
                    continue
            
            # Rerank for the group
            recommendations = reranker.rerank_for_group(user_recs, all_users, group_weights)
            
            # Get friend activity for enhancing recommendations with social signals
            friend_notifications = reranker.generate_friend_activity_notifications(
                user_id, friend_ids, db['watch_history'], db['content_reactions']
            )
            
        else:
            # Individual recommendations
            base_recommendations = recommender.predict_for_user(user_id, top_n=num_recommendations * 2)
            
            # Rerank based on context
            recommendations = reranker.rerank_for_context(base_recommendations, context)
            
            # Get friend activity if friend_ids provided
            friend_notifications = []
            if friend_ids:
                friend_notifications = reranker.generate_friend_activity_notifications(
                    user_id, friend_ids, db['watch_history'], db['content_reactions']
                )
        
        # Enrich recommendations with content metadata
        enriched_recs = []
        for content_id, score in recommendations[:num_recommendations]:
            # Find content in database
            content_data = db['content'][db['content']['content_id'] == content_id]
            if len(content_data) > 0:
                content_row = content_data.iloc[0]
                
                # Check if this content has any friend activity notifications
                related_notifications = [n for n in friend_notifications if n['content_id'] == content_id]
                social_signal = related_notifications[0] if related_notifications else None
                
                enriched_recs.append({
                    "content_id": content_id,
                    "title": content_row["title"],
                    "type": content_row["type"],
                    "genre": content_row["genre"],
                    "score": float(score),
                    "social_signal": social_signal,
                    "reason": get_recommendation_reason(content_id, score, social_signal, context)
                })
        
        return {
            "user_id": user_id,
            "group_mode": group_mode,
            "recommendations": enriched_recs,
            "context": context
        }
        
    except Exception as e:
        # In production, add proper logging
        print(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def get_recommendation_reason(content_id, score, social_signal, context):
    """Generate a human-readable explanation for the recommendation"""
    reasons = []
    
    # Social reason (prioritize this if available)
    if social_signal:
        reasons.append(social_signal["message"])
    
    # Contextual reasons
    current_hour = int(context["Context Time"]["time"].split(":")[0])
    if 6 <= current_hour < 12:
        reasons.append("Perfect for your morning")
    elif 12 <= current_hour < 17:
        reasons.append("Great for your afternoon")
    elif 17 <= current_hour < 22:
        reasons.append("Ideal for your evening")
    else:
        reasons.append("Good for late night viewing")
    
    # Weather-based reason
    weather = context["Weather"]["description"].lower()
    if "rain" in weather or "storm" in weather:
        reasons.append("Great for this rainy weather")
    elif "clear" in weather or "sunny" in weather:
        reasons.append("Popular on sunny days like today")
    
    # Festival-based reason
    current_date = datetime.strptime(context["Context Time"]["date"], "%Y-%m-%d")
    for festival in context["Festivals"]:
        festival_date = datetime.strptime(festival["date"], "%Y-%m-%d")
        days_to_festival = (festival_date - current_date).days
        if -1 <= days_to_festival <= 3:
            reasons.append(f"Trending for {festival['name']}")
            break
    
    # Score-based reason
    if score > 0.8:
        reasons.append("Highly matched to your taste")
    elif score > 0.6:
        reasons.append("Matches your viewing patterns")
    
    # Return top 2 most relevant reasons
    return reasons[:2]

@app.get("/social-notifications/{user_id}")
def get_social_notifications(
    user_id: str,
    reranker = Depends(get_reranker),
    db = Depends(get_db_connection)
):
    """Get social notifications about friend activity"""
    try:
        # Get user's friends
        connections = db['user_connections']
        friends = connections[connections['user_id'] == user_id]['friend_id'].tolist()
        
        if not friends:
            return {"notifications": []}
        
        # Generate notifications
        notifications = reranker.generate_friend_activity_notifications(
            user_id, friends, db['watch_history'], db['content_reactions']
        )
        
        # Enrich notifications with content details
        enriched_notifications = []
        for notification in notifications:
            content_id = notification['content_id']
            content_data = db['content'][db['content']['content_id'] == content_id]
            
            if len(content_data) > 0:
                content_row = content_data.iloc[0]
                enriched_notifications.append({
                    "notification_id": hash(f"{content_id}-{notification['type']}"),
                    "content_id": content_id,
                    "title": content_row["title"],
                    "type": content_row["type"],
                    "thumbnail": f"https://example.com/thumbnails/{content_id}.jpg",
                    "message": notification["message"],
                    "notification_type": notification["type"],
                    "friend_count": len(notification["friend_ids"])
                })
        
        return {"notifications": enriched_notifications}
        
    except Exception as e:
        # In production, add proper logging
        print(f"Error generating social notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)