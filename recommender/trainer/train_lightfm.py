import numpy as np
import pandas as pd
from scipy import sparse
from lightfm import LightFM
import pickle
import os
from datetime import datetime, timedelta

class FireTVRecommender:
    def __init__(self):
        self.model = None
        self.user_features = None
        self.item_features = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        
    def preprocess_data(self, watch_history_df, content_reactions_df, users_df, content_df):
        """
        Preprocess data into user-item interaction matrix and feature matrices
        """
        print("Preprocessing data...")
        
        # Create user and item mappings
        unique_users = users_df['user_id'].unique()
        unique_items = content_df['content_id'].unique()
        
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        # Create user-item interaction matrix (weighted by watch completion and reactions)
        interactions = []
        weights = []
        
        # Process watch history
        for _, row in watch_history_df.iterrows():
            user_idx = self.user_mapping.get(row['user_id'])
            item_idx = self.item_mapping.get(row['content_id'])
            
            if user_idx is not None and item_idx is not None:
                # Basic interaction from watching
                interactions.append((user_idx, item_idx))
                
                # Weight based on completion percentage
                completion_weight = min(row['completion_percentage'] / 100.0, 1.0)
                weights.append(completion_weight)
        
        # Process explicit reactions (ratings, likes)
        for _, row in content_reactions_df.iterrows():
            user_idx = self.user_mapping.get(row['user_id'])
            item_idx = self.item_mapping.get(row['content_id'])
            
            if user_idx is not None and item_idx is not None:
                interactions.append((user_idx, item_idx))
                
                # Weight based on rating or reaction
                if pd.notnull(row['rating']):
                    weight = row['rating'] / 5.0  # Normalize to 0-1
                elif row['liked'] is True:
                    weight = 1.0
                else:
                    weight = 0.5  # Default weight for other reactions
                
                weights.append(weight)
        
        # Convert to sparse matrix format
        rows, cols = zip(*interactions) if interactions else ([], [])
        interaction_matrix = sparse.coo_matrix(
            (weights, (rows, cols)),
            shape=(len(self.user_mapping), len(self.item_mapping))
        ).tocsr()
        
        # Build user features
        user_features_dict = {}
        for _, user_row in users_df.iterrows():
            user_idx = self.user_mapping.get(user_row['user_id'])
            if user_idx is not None:
                features = []
                
                # Age groups
                age = user_row.get('age', 0)
                age_buckets = [(0, 18), (19, 24), (25, 34), (35, 44), (45, 54), (55, 100)]
                for i, (low, high) in enumerate(age_buckets):
                    if low <= age <= high:
                        features.append(f"age_bucket_{i}")
                
                # Gender
                gender = user_row.get('gender', 'unknown')
                features.append(f"gender_{gender}")
                
                # Location features
                location = user_row.get('location', '')
                features.append(f"location_{location}")
                
                # Store features for this user
                user_features_dict[user_idx] = features
        
        # Build item features
        item_features_dict = {}
        for _, content_row in content_df.iterrows():
            item_idx = self.item_mapping.get(content_row['content_id'])
            if item_idx is not None:
                features = []
                
                # Content type
                features.append(f"type_{content_row['type']}")
                
                # Genres
                for genre in content_row.get('genre', []):
                    features.append(f"genre_{genre}")
                
                # Release year buckets
                year = content_row.get('release_year', 2000)
                year_buckets = [(1900, 1980), (1981, 1990), (1991, 2000), 
                               (2001, 2010), (2011, 2020), (2021, 2030)]
                for i, (low, high) in enumerate(year_buckets):
                    if low <= year <= high:
                        features.append(f"year_bucket_{i}")
                
                # Duration buckets
                duration = content_row.get('duration', 90)
                if duration <= 30:
                    features.append("duration_short")
                elif duration <= 60:
                    features.append("duration_medium")
                elif duration <= 120:
                    features.append("duration_long")
                else:
                    features.append("duration_very_long")
                
                # Mood tags
                for mood in content_row.get('mood_tags', []):
                    features.append(f"mood_{mood}")
                
                # Store features for this item
                item_features_dict[item_idx] = features
        
        # Convert to LightFM feature matrices
        user_features = self._features_to_matrix(user_features_dict, len(self.user_mapping))
        item_features = self._features_to_matrix(item_features_dict, len(self.item_mapping))
        
        return interaction_matrix, user_features, item_features
    
    def _features_to_matrix(self, features_dict, num_entities):
        """Convert dictionary of features to a sparse matrix"""
        all_features = set()
        for features in features_dict.values():
            all_features.update(features)
        
        feature_mapping = {feature: idx for idx, feature in enumerate(all_features)}
        
        rows = []
        cols = []
        data = []
        
        for entity_idx, features in features_dict.items():
            for feature in features:
                feature_idx = feature_mapping.get(feature)
                if feature_idx is not None:
                    rows.append(entity_idx)
                    cols.append(feature_idx)
                    data.append(1.0)  # Binary feature presence
        
        return sparse.coo_matrix(
            (data, (rows, cols)), 
            shape=(num_entities, len(feature_mapping))
        ).tocsr()
    
    def train(self, watch_history_df, content_reactions_df, users_df, content_df,
              num_components=128, learning_rate=0.05, epochs=30):
        """
        Train the hybrid recommender model
        """
        print("Training recommendation model...")
        
        # Preprocess the data
        interaction_matrix, user_features, item_features = self.preprocess_data(
            watch_history_df, content_reactions_df, users_df, content_df
        )
        
        # Save the feature matrices for prediction
        self.user_features = user_features
        self.item_features = item_features
        
        # Initialize and train the model
        self.model = LightFM(
            no_components=num_components,
            learning_rate=learning_rate,
            loss='warp',  # Weighted Approximate-Rank Pairwise loss
            user_alpha=0.0001,  # L2 penalty on user features
            item_alpha=0.0001   # L2 penalty on item features
        )
        
        # Train the model
        self.model.fit(
            interactions=interaction_matrix,
            user_features=user_features,
            item_features=item_features,
            epochs=epochs,
            verbose=True
        )
        
        print("Model training complete")
        return self
    
    def save_model(self, path='models/firetv_recommender.pkl'):
        """Save the trained model and mappings"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping,
            'user_features': self.user_features,
            'item_features': self.item_features
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load_model(cls, path='models/firetv_recommender.pkl'):
        """Load a trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        recommender = cls()
        recommender.model = model_data['model']
        recommender.user_mapping = model_data['user_mapping']
        recommender.item_mapping = model_data['item_mapping']
        recommender.reverse_user_mapping = model_data['reverse_user_mapping']
        recommender.reverse_item_mapping = model_data['reverse_item_mapping']
        recommender.user_features = model_data['user_features']
        recommender.item_features = model_data['item_features']
        
        return recommender
    
    def predict_for_user(self, user_id, content_ids=None, top_n=10):
        """
        Generate recommendations for a user
        
        Args:
            user_id: The ID of the user
            content_ids: Optional list of content IDs to score, if None will score all items
            top_n: Number of recommendations to return
        
        Returns:
            List of (content_id, score) tuples
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        user_idx = self.user_mapping.get(user_id)
        if user_idx is None:
            raise ValueError(f"Unknown user ID: {user_id}")
        
        if content_ids is not None:
            # Score only the specified content items
            item_indices = [self.item_mapping.get(cid) for cid in content_ids if cid in self.item_mapping]
            scores = self.model.predict(
                user_ids=[user_idx] * len(item_indices),
                item_ids=item_indices,
                user_features=self.user_features,
                item_features=self.item_features
            )
            id_scores = list(zip([self.reverse_item_mapping[idx] for idx in item_indices], scores))
        else:
            # Score all items
            scores = self.model.predict(
                user_ids=user_idx,
                item_ids=list(range(len(self.item_mapping))),
                user_features=self.user_features,
                item_features=self.item_features
            )
            id_scores = [(self.reverse_item_mapping[i], score) for i, score in enumerate(scores)]
        
        # Sort by score and return top N
        return sorted(id_scores, key=lambda x: x[1], reverse=True)[:top_n]

# Example usage:
if __name__ == "__main__":
    # This would be replaced with actual database calls in production
    watch_history_df = pd.read_csv("data/watch_history.csv")
    content_reactions_df = pd.read_csv("data/content_reactions.csv")
    users_df = pd.read_csv("data/users.csv")
    content_df = pd.read_csv("data/content.csv")
    
    # Initialize and train recommender
    recommender = FireTVRecommender()
    recommender.train(watch_history_df, content_reactions_df, users_df, content_df)
    
    # Save the trained model
    recommender.save_model()