# tests/smoke_test_recommender.py
import pytest
from recommender import ContentRecommender
from intent import SearchIntent
from user_context import UserContextFetcher

# Test configuration matching your setup
TEST_DB_CONFIG = {
    'user': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': '5432',
    'dbname': 'firestream_db'
}

def test_recommender_with_existing_data():
    """Smoke test using only pre-populated database data"""
    # Initialize components
    context_fetcher = UserContextFetcher(TEST_DB_CONFIG)
    recommender = ContentRecommender(TEST_DB_CONFIG)
    
    # Test with user USR10001 from your populated data
    test_user_id = "USR10001"
    
    # Verify user exists and get context
    try:
        context = context_fetcher.fetch_context(test_user_id)
    except Exception as e:
        pytest.fail(f"Failed to fetch context for user {test_user_id}: {str(e)}")
    
    # Test mood-based recommendations (using "happy" as sample mood)
    mood_recs = recommender.recommend(
        context=context,
        intent=SearchIntent.MOOD,
        intent_entities={"mood": "happy"}
    )
    
    assert len(mood_recs) > 0, "Should return mood-based recommendations"
    print("\nMood-Based Recommendations:")
    for rec in mood_recs[:3]:
        print(f"- {rec.title} (Score: {rec.score:.2f})")
        print(f"  Reasons: {', '.join(rec.match_reasons)}")
    
    # Test genre-based recommendations (using "Comedy" from your populated data)
    genre_recs = recommender.recommend(
        context=context,
        intent=SearchIntent.GENRE,
        intent_entities={"genres": ["Comedy"]}
    )
    
    assert len(genre_recs) > 0, "Should return genre-based recommendations"
    print("\nGenre-Based Recommendations:")
    for rec in genre_recs[:3]:
        print(f"- {rec.title} (Score: {rec.score:.2f})")
        print(f"  Reasons: {', '.join(rec.match_reasons)}")
    
    # Test general recommendations
    general_recs = recommender.recommend(
        context=context,
        intent=SearchIntent.GENERAL
    )
    
    assert len(general_recs) > 0, "Should return general recommendations"
    print("\nGeneral Recommendations:")
    for rec in general_recs[:3]:
        print(f"- {rec.title} (Score: {rec.score:.2f})")
        print(f"  Reasons: {', '.join(rec.match_reasons)}")
    
    print("\nSmoke test passed successfully!")

if __name__ == "__main__":
    test_recommender_with_existing_data()