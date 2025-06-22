import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import psycopg2
import json
import logging
import os
from datetime import datetime, timedelta
from database.database_connection import db
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import train_test_split
import pickle
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class IntelligentRecommendationEngine:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.load_models()
        
    def load_models(self):
        """Load and initialize open-source models"""
        try:
            # 1. Sentence Transformer for semantic similarity
            logger.info("Loading SentenceTransformer...")
            self.models['sentence_transformer'] = SentenceTransformer('all-MiniLM-L6-v2')
            
            # 2. Sentiment Analysis Model
            logger.info("Loading sentiment analysis model...")
            self.models['sentiment'] = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True
            )
            
            # 3. Emotion Detection Model
            logger.info("Loading emotion detection model...")
            self.models['emotion'] = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            
            # 4. TF-IDF Vectorizer for content similarity
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # 5. Topic Modeling (LDA)
            self.models['lda'] = LatentDirichletAllocation(
                n_components=10,
                random_state=42,
                max_iter=20
            )
            
            # 6. Load or train collaborative filtering model
            self.load_collaborative_filtering_model()
            
            logger.info("✅ All models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Fallback to basic models
            self.models = {}
    
    def load_collaborative_filtering_model(self):
        """Load or train collaborative filtering model using Surprise library"""
        try:
            model_path = "ml_models/trained_models/cf_model.pkl"
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models['collaborative_filtering'] = pickle.load(f)
                logger.info("Loaded existing collaborative filtering model")
            else:
                logger.info("Training new collaborative filtering model...")
                self.train_collaborative_filtering_model()
                
        except Exception as e:
            logger.error(f"Error with collaborative filtering model: {e}")
            self.models['collaborative_filtering'] = None
    
    def train_collaborative_filtering_model(self):
        """Train collaborative filtering model on user ratings"""
        try:
            # Get rating data from database
            conn = db.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT user_id, content_id, rating
                FROM content_reactions
                WHERE rating IS NOT NULL
            """)
            
            ratings_data = cur.fetchall()
            cur.close()
            conn.close()
            
            if len(ratings_data) < 10:
                logger.warning("Not enough rating data for collaborative filtering")
                return
            
            # Create Surprise dataset
            df = pd.DataFrame(ratings_data, columns=['user_id', 'content_id', 'rating'])
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(df[['user_id', 'content_id', 'rating']], reader)
            
            # Train SVD model
            trainset = data.build_full_trainset()
            algo = SVD(n_factors=50, random_state=42)
            algo.fit(trainset)
            
            self.models['collaborative_filtering'] = algo
            
            # Save model
            os.makedirs("ml_models/trained_models", exist_ok=True)
            with open("ml_models/trained_models/cf_model.pkl", 'wb') as f:
                pickle.dump(algo, f)
            
            logger.info("✅ Collaborative filtering model trained and saved")
            
        except Exception as e:
            logger.error(f"Error training collaborative filtering: {e}")
            self.models['collaborative_filtering'] = None
    
    def analyze_user_query(self, query, prosody_features=None):
        """Advanced query analysis using NLP models"""
        analysis = {
            "original_query": query,
            "sentiment": None,
            "emotions": None,
            "detected_mood": None,
            "search_intent": "general",
            "semantic_embedding": None,
            "prosody_mood": None
        }
        
        try:
            # 1. Sentiment Analysis
            if 'sentiment' in self.models:
                sentiment_result = self.models['sentiment'](query)
                analysis["sentiment"] = max(sentiment_result[0], key=lambda x: x['score'])
            
            # 2. Emotion Detection
            if 'emotion' in self.models:
                emotion_result = self.models['emotion'](query)
                analysis["emotions"] = sorted(emotion_result[0], key=lambda x: x['score'], reverse=True)
                
                # Map emotions to moods
                top_emotion = analysis["emotions"][0]['label'].lower()
                emotion_to_mood = {
                    'joy': 'happy',
                    'sadness': 'sad',
                    'anger': 'angry',
                    'fear': 'scared',
                    'surprise': 'excited',
                    'disgust': 'angry',
                    'love': 'romantic'
                }
                analysis["detected_mood"] = emotion_to_mood.get(top_emotion, top_emotion)
            
            # 3. Semantic Embedding
            if 'sentence_transformer' in self.models:
                analysis["semantic_embedding"] = self.models['sentence_transformer'].encode(query)
            
            # 4. Search Intent Classification
            analysis["search_intent"] = self.classify_search_intent(query)
            
            # 5. Prosody-based mood enhancement
            if prosody_features:
                analysis["prosody_mood"] = self.analyze_prosody_mood(prosody_features)
                
                # Combine text mood with prosody mood
                if analysis["prosody_mood"] and not analysis["detected_mood"]:
                    analysis["detected_mood"] = analysis["prosody_mood"]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in query analysis: {e}")
            return analysis
    
    def classify_search_intent(self, query):
        """Classify search intent using pattern matching and ML"""
        query_lower = query.lower()
        
        # Genre keywords
        genre_patterns = {
            'action': ['action', 'fight', 'adventure', 'superhero'],
            'comedy': ['comedy', 'funny', 'laugh', 'humor', 'hilarious'],
            'drama': ['drama', 'emotional', 'serious', 'deep'],
            'horror': ['horror', 'scary', 'frightening', 'terror'],
            'romance': ['romance', 'love', 'romantic', 'date'],
            'sci-fi': ['sci-fi', 'science fiction', 'futuristic', 'space'],
            'thriller': ['thriller', 'suspense', 'mystery', 'psychological']
        }
        
        # Check for genre matches
        for genre, keywords in genre_patterns.items():
            if any(keyword in query_lower for keyword in keywords):
                return f"genre_{genre}"
        
        # Mood-based searches
        mood_keywords = ['sad', 'happy', 'bored', 'stressed', 'tired', 'excited', 'lonely']
        if any(mood in query_lower for mood in mood_keywords):
            return "mood_based"
        
        # Person searches
        if any(word in query_lower for word in ['actor', 'director', 'starring', 'cast']):
            return "person"
        
        # Context searches
        if any(word in query_lower for word in ['family', 'date', 'alone', 'friends', 'weekend']):
            return "context"
        
        return "general"
    
    def analyze_prosody_mood(self, prosody_features):
        """Analyze mood from prosody features using ML"""
        try:
            intensity = prosody_features.get("intensity", "medium")
            tempo = prosody_features.get("tempo", "medium")
            energy = prosody_features.get("rms_energy", 0.5)
            pitch = prosody_features.get("estimated_pitch_hz", 150)
            
            # Create feature vector for mood classification
            features = np.array([
                1 if intensity == "high" else 0.5 if intensity == "medium" else 0,
                1 if tempo == "fast" else 0.5 if tempo == "medium" else 0,
                min(energy * 10, 1),  # Normalize energy
                min(pitch / 300, 1)   # Normalize pitch
            ])
            
            # Simple rule-based mood detection (can be replaced with trained model)
            if features[0] > 0.7 and features[1] > 0.7:  # High intensity, fast tempo
                return "excited"
            elif features[0] < 0.3 and features[1] < 0.3:  # Low intensity, slow tempo
                return "sad" if features[2] < 0.3 else "tired"
            elif features[2] < 0.3:  # Low energy
                return "tired"
            elif features[0] > 0.6:  # High intensity
                return "happy"
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing prosody mood: {e}")
            return None
    
    def get_content_embeddings(self):
        """Get or compute content embeddings"""
        try:
            embeddings_path = "ml_models/trained_models/content_embeddings.pkl"
            
            if os.path.exists(embeddings_path):
                with open(embeddings_path, 'rb') as f:
                    return pickle.load(f)
            
            # Compute embeddings
            conn = db.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT content_id, title, description, 
                       array_to_string(genre, ' ') as genres,
                       array_to_string(mood_tags, ' ') as moods
                FROM content
            """)
            
            content_data = cur.fetchall()
            cur.close()
            conn.close()
            
            if not content_data:
                return {}
            
            # Create text representation for each content
            content_texts = []
            content_ids = []
            
            for content_id, title, description, genres, moods in content_data:
                text = f"{title} {description} {genres or ''} {moods or ''}"
                content_texts.append(text)
                content_ids.append(content_id)
            
            # Generate embeddings
            embeddings = self.models['sentence_transformer'].encode(content_texts)
            
            # Create mapping
            embedding_map = dict(zip(content_ids, embeddings))
            
            # Save embeddings
            os.makedirs("ml_models/trained_models", exist_ok=True)
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embedding_map, f)
            
            return embedding_map
            
        except Exception as e:
            logger.error(f"Error computing content embeddings: {e}")
            return {}
    
    def get_semantic_recommendations(self, query_analysis, user_id, top_k=10):
        """Get recommendations based on semantic similarity"""
        try:
            if not query_analysis.get("semantic_embedding") is not None:
                return []
            
            content_embeddings = self.get_content_embeddings()
            if not content_embeddings:
                return []
            
            query_embedding = query_analysis["semantic_embedding"]
            
            # Calculate similarities
            similarities = []
            for content_id, content_embedding in content_embeddings.items():
                similarity = cosine_similarity(
                    query_embedding.reshape(1, -1),
                    content_embedding.reshape(1, -1)
                )[0][0]
                similarities.append((content_id, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return [content_id for content_id, _ in similarities[:top_k]]
            
        except Exception as e:
            logger.error(f"Error in semantic recommendations: {e}")
            return []
    
    def get_collaborative_recommendations(self, user_id, top_k=10):
        """Get collaborative filtering recommendations"""
        try:
            if not self.models.get('collaborative_filtering'):
                return []
            
            # Get all content IDs
            conn = db.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT content_id FROM content")
            all_content_ids = [row[0] for row in cur.fetchall()]
            cur.close()
            conn.close()
            
            # Get predictions for all content
            predictions = []
            for content_id in all_content_ids:
                try:
                    pred = self.models['collaborative_filtering'].predict(user_id, content_id)
                    predictions.append((content_id, pred.est))
                except:
                    continue
            
            # Sort by predicted rating
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return [content_id for content_id, _ in predictions[:top_k]]
            
        except Exception as e:
            logger.error(f"Error in collaborative recommendations: {e}")
            return []
    
    def get_mood_based_recommendations(self, detected_mood, user_id, top_k=10):
        """Get recommendations based on detected mood"""
        try:
            conn = db.get_connection()
            cur = conn.cursor()
            
            # Mood to content mapping
            mood_mappings = {
                'sad': ['uplifting', 'cheerful', 'heartwarming', 'inspiring'],
                'happy': ['exciting', 'fun', 'energetic', 'adventurous'],
                'stressed': ['relaxing', 'calming', 'peaceful', 'cozy'],
                'bored': ['exciting', 'thrilling', 'intense', 'mind-bending'],
                'lonely': ['heartwarming', 'romantic', 'warm', 'feel-good'],
                'angry': ['cathartic', 'intense', 'powerful'],
                'tired': ['gentle', 'peaceful', 'relaxing', 'cozy'],
                'excited': ['thrilling', 'energetic', 'intense', 'adventurous']
            }
            
            target_moods = mood_mappings.get(detected_mood, [])
            if not target_moods:
                return []
            
            # Query for content with matching mood tags
            mood_conditions = " OR ".join([f"%s = ANY(mood_tags)" for _ in target_moods])
            
            cur.execute(f"""
                SELECT content_id, rating,
                       (SELECT COUNT(*) FROM unnest(mood_tags) AS tag 
                        WHERE tag = ANY(%s)) as mood_match_count
                FROM content
                WHERE {mood_conditions}
                ORDER BY mood_match_count DESC, rating DESC
                LIMIT %s
            """, target_moods + target_moods + [top_k])
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            return [row[0] for row in results]
            
        except Exception as e:
            logger.error(f"Error in mood-based recommendations: {e}")
            return []
    
    def get_recommendations(self, user_id, search_query, prosody_features=None):
        """Main recommendation function using multiple ML approaches"""
        try:
            # 1. Analyze the query
            query_analysis = self.analyze_user_query(search_query, prosody_features)
            
            # 2. Get recommendations from different approaches
            recommendation_sources = {}
            
            # Semantic similarity recommendations
            semantic_recs = self.get_semantic_recommendations(query_analysis, user_id)
            recommendation_sources['semantic'] = semantic_recs
            
            # Collaborative filtering recommendations
            collaborative_recs = self.get_collaborative_recommendations(user_id)
            recommendation_sources['collaborative'] = collaborative_recs
            
            # Mood-based recommendations
            if query_analysis.get('detected_mood'):
                mood_recs = self.get_mood_based_recommendations(
                    query_analysis['detected_mood'], user_id
                )
                recommendation_sources['mood'] = mood_recs
            
            # 3. Combine and rank recommendations
            final_recommendations = self.combine_recommendations(
                recommendation_sources, query_analysis, user_id
            )
            
            # 4. Get content details
            content_details = self.get_content_details(final_recommendations)
            
            return {
                "recommendations": content_details,
                "search_intent": query_analysis,
                "total_results": len(content_details),
                "prosody_influenced": bool(prosody_features),
                "ml_methods_used": list(recommendation_sources.keys()),
                "user_context_applied": True
            }
            
        except Exception as e:
            logger.error(f"Error in get_recommendations: {e}")
            return self.fallback_recommendations(search_query, user_id)
    
    def combine_recommendations(self, recommendation_sources, query_analysis, user_id):
        """Combine recommendations from different sources using weighted scoring"""
        content_scores = {}
        
        # Weights for different recommendation sources
        weights = {
            'semantic': 0.4,
            'collaborative': 0.3,
            'mood': 0.3
        }
        
        # If prosody detected mood, increase mood weight
        if query_analysis.get('prosody_mood'):
            weights['mood'] = 0.5
            weights['semantic'] = 0.3
            weights['collaborative'] = 0.2
        
        # Combine scores
        for source, content_list in recommendation_sources.items():
            weight = weights.get(source, 0.1)
            
            for i, content_id in enumerate(content_list):
                # Score decreases with position
                score = weight * (1.0 - (i / len(content_list)))
                
                if content_id in content_scores:
                    content_scores[content_id] += score
                else:
                    content_scores[content_id] = score
        
        # Sort by combined score
        ranked_content = sorted(content_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [content_id for content_id, score in ranked_content[:10]]
    
    def get_content_details(self, content_ids):
        """Get detailed content information"""
        if not content_ids:
            return []
        
        try:
            conn = db.get_connection()
            cur = conn.cursor()
            
            placeholders = ','.join(['%s'] * len(content_ids))
            cur.execute(f"""
                SELECT c.content_id, c.title, c.description, c.genre, c.mood_tags,
                       c.rating, c.release_year, c.duration, c.director, c.actors,
                       COALESCE(AVG(cr.rating), c.rating) as avg_rating
                FROM content c
                LEFT JOIN content_reactions cr ON c.content_id = cr.content_id
                WHERE c.content_id IN ({placeholders})
                GROUP BY c.content_id, c.title, c.description, c.genre, c.mood_tags,
                         c.rating, c.release_year, c.duration, c.director, c.actors
                ORDER BY ARRAY_POSITION(ARRAY[{placeholders}], c.content_id)
            """, content_ids + content_ids)
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            recommendations = []
            for row in results:
                recommendations.append({
                    "id": row[0],
                    "content_id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "genre": row[3] if row[3] else [],
                    "mood_tags": row[4] if row[4] else [],
                    "rating": float(row[5]) if row[5] else 0,
                    "release_year": row[6],
                    "duration": row[7],
                    "director": row[8] if row[8] else [],
                    "actors": row[9] if row[9] else [],
                    "relevance_score": float(row[10]) if row[10] else 0,
                    "recommendation_reason": "AI/ML recommendation based on multiple factors"
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error getting content details: {e}")
            return []
    
    def fallback_recommendations(self, search_query, user_id):
        """Simple fallback when ML models fail"""
        try:
            conn = db.get_connection()
            cur = conn.cursor()
            
            cur.execute("""
                SELECT content_id, title, description, genre, rating
                FROM content
                WHERE title ILIKE %s OR description ILIKE %s
                ORDER BY rating DESC
                LIMIT 5
            """, (f"%{search_query}%", f"%{search_query}%"))
            
            results = cur.fetchall()
            cur.close()
            conn.close()
            
            recommendations = []
            for row in results:
                recommendations.append({
                    "id": row[0],
                    "content_id": row[0],
                    "title": row[1],
                    "description": row[2],
                    "genre": row[3] if row[3] else [],
                    "rating": float(row[4]) if row[4] else 0,
                    "recommendation_reason": "Basic text search result"
                })
            
            return {
                "recommendations": recommendations,
                "search_intent": {"original_query": search_query},
                "total_results": len(recommendations),
                "prosody_influenced": False,
                "fallback_used": True
            }
            
        except Exception as e:
            logger.error(f"Error in fallback recommendations: {e}")
            return {"recommendations": [], "error": str(e)}

# Global instance
recommendation_engine = IntelligentRecommendationEngine()