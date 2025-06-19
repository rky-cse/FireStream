FireTV AI-Driven Recommendation Engine
README for Recommendation Module
This document provides details on the recommendation system implementation, including models used, architecture, data flows, setup instructions, and testing procedures.

Table of Contents
Overview
Models Used
Key Features
Architecture
Data Flow
Setup Instructions
Testing
Requirements
Overview
The FireTV AI-Driven Recommendation Engine is a hybrid recommendation system that combines collaborative filtering, content-based filtering, and contextual reranking to provide personalized content recommendations. The system incorporates social signals, contextual information (time, weather, festivals), and group dynamics to enhance the relevance of recommendations.

Models Used
1. LightFM Hybrid Recommendation Model
Description: Core recommendation model that combines collaborative filtering with content features
Technique: Matrix factorization with user and item features
Advantages: Handles cold-start problem, incorporates content metadata, and user behavior
Loss Function: WARP (Weighted Approximate-Rank Pairwise) optimized for ranking performance
2. Contextual Boosting Model
Description: Rule-based contextual reranking system
Technique: Applies context-based multipliers to base recommendation scores
Contexts: Time of day, day of week, weather conditions, nearby festivals/holidays
3. Social Signal Processing
Description: Friend activity tracking and group recommendation logic
Technique: User similarity calculation, social proof metrics, notification generation
Outputs: Friend activity notifications, group recommendation weights
4. Feature Engineering Pipeline
Description: Creates rich user and content embeddings
Technique: Time-based patterns, genre preferences, demographic features
Usage: Input features for the LightFM model and contextual reranker
Key Features
Personalized Recommendations

Individual user recommendations based on watch history and preferences
Time-pattern recognition (e.g., sci-fi at night, news in the morning)
Genre and content-type preferences
Contextual Awareness

Time-of-day and day-of-week optimization
Weather-based recommendations (e.g., family movies on rainy days)
Festival and holiday-themed suggestions
Social Features

Friend activity notifications ("3 friends loved this movie")
Social proof integration in recommendation ranking
Friend-based trust signals
Group Recommendations

Multi-user optimization for shared viewing experiences
Dynamic weighting based on user similarity and group composition
Aggregation strategies to balance individual preferences
Recommendation Explanations

Human-readable explanations for recommendations
Social, contextual, and preference-based reasoning
Architecture
The recommender module consists of several components that work together:

Code
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Feature Engineering  │     │ Model Training       │     │ Inference Pipeline   │
│                      │     │                      │     │                      │
│ - User embeddings    │────►│ - LightFM training   │────►│ - Base predictions   │
│ - Content embeddings │     │ - Hyperparameter     │     │ - Contextual rerank  │
│ - Context processing │     │   optimization       │     │ - Group aggregation  │
└──────────────────────┘     └──────────────────────┘     └──────────┬───────────┘
                                                                     │
                                                                     ▼
┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│ Social Signal        │     │ Explanation          │     │ API Layer            │
│ Processor            │     │ Generator            │     │                      │
│                      │◄────┤ - Reason formatting  │◄────┤ - FastAPI endpoints  │
│ - Friend activity    │     │ - Context-aware      │     │ - Request handling   │
│ - Notifications      │     │   explanations       │     │ - Response formatting│
└──────────────────────┘     └──────────────────────┘     └──────────────────────┘
Data Flow
Input Data Sources

User watch history (content_id, timestamp, completion %)
Content metadata (genres, types, duration, release year)
User profiles (demographics, preferences)
Social connections (friend relationships)
Real-time context (time, date, weather, festivals)
Group composition (for multi-user sessions)
Feature Processing Flow

Code
Raw Data → Feature Extraction → Feature Normalization → Feature Combination → Embeddings
Recommendation Generation Flow

Code
User Request → Base Recommendations (LightFM) → Context Application → Social Signal Integration → Final Ranking
Group Recommendation Flow

Code
Individual Recommendations → User Weight Calculation → Score Aggregation → Group Reranking → Final Group Recommendations
Social Notification Flow

Code
Friend Activity Collection → Filtering → Notification Generation → Content Enrichment → Delivery
Setup Instructions
Prerequisites
Python 3.8+
pip or conda package manager
Installation
Clone the repository:
bash
git clone https://github.com/yourusername/firetv-recommender.git
cd firetv-recommender
Create a virtual environment:
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:
bash
pip install -r requirements.txt
Download and prepare sample data:
bash
mkdir -p data
python scripts/download_sample_data.py
Train the base model:
bash
python recommender/trainer/train_lightfm.py
Start the API server:
bash
cd api
uvicorn app.main:app --reload
The API will be available at http://127.0.0.1:8000/

Testing
Sample Test Data
Since you don't have any database or data, we'll use the MovieLens dataset as a sample:

bash
# Download and prepare MovieLens dataset
python scripts/prepare_movielens.py
This script will:

Download the MovieLens 100K dataset
Transform it into the schema expected by our system
Generate sample user profiles, social connections, and contextual data
Running Tests
Unit tests:
bash
pytest tests/unit/
Integration tests:
bash
pytest tests/integration/
Example API call:
bash
curl -X GET "http://localhost:8000/recommendations/user123?num_recommendations=10" | json_pp
Testing Different Scenarios
Time-based recommendations:
bash
python scripts/test_context.py --context-type time --time "08:00:00"
python scripts/test_context.py --context-type time --time "20:00:00"
Weather-based recommendations:
bash
python scripts/test_context.py --context-type weather --condition "rainy"
python scripts/test_context.py --context-type weather --condition "sunny"
Group recommendations:
bash
python scripts/test_group.py --users user123,user456,user789
Festival-based recommendations:
bash
python scripts/test_context.py --context-type festival --festival "Christmas"
Requirements

requirements.txt
# Core dependencies
numpy==1.24.3
pandas==2.0.3
scipy==1.10.1
scikit-learn==1.3.0
lightfm==1.17
System Requirements
CPU: 4+ cores recommended for training
RAM: 8GB minimum, 16GB+ recommended
Storage: 1GB for code and basic models, 10GB+ if using full datasets
OS: Linux, macOS, or Windows
Optional: CUDA-compatible GPU for faster model training
Sample Data Generation
To generate synthetic data for testing without a database:

bash
python scripts/generate_synthetic_data.py --users 1000 --content 5000 --watch-events 100000
This will create CSV files in the data/ directory that mimic the structure of our database tables, including:

users.csv: User profiles with demographics
content.csv: Content metadata with genres, types, etc.
watch_history.csv: Viewing history with completion percentages
content_reactions.csv: User ratings and reactions
user_connections.csv: Social connections between users
File Structure
Code
recommender/
├── embedding/
│   └── build_embeddings.py    # User and content embedding generation
├── trainer/
│   └── train_lightfm.py       # Core recommendation model training
├── reranker/
│   └── rerank.py              # Contextual reranking implementation
├── shared/
│   └── utils.py               # Shared utility functions
├── models/                    # Directory for saved model files
│   └── firetv_recommender.pkl # Trained model
└── README.md                  # This documentation
Next Steps for Development
Implement A/B testing framework for algorithm evaluation
Add personalized weight optimization based on user feedback
Enhance group recommendation algorithms with more advanced strategies
Integrate real-time sentiment analysis from voice and chat
Add video content parsing for e-commerce tie-ins