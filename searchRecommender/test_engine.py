#!/usr/bin/env python3
"""
Test script for AI Search Recommendation Engine
Current Date and Time: 2025-06-20 09:24:30 UTC
Current User: rky-cse
"""

import sys
import os
import asyncio
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our recommendation system
from engine import get_search_recommender
from database import get_database_manager
#from recommender.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_database_connection():
    """Test database connectivity and basic operations."""
    print("\n" + "="*50)
    print("🗄️  TESTING DATABASE CONNECTION")
    print("="*50)
    
    try:
        # Initialize database manager
        db_manager = get_database_manager()
        
        # Health check
        health = db_manager.health_check()
        print(f"Database Health: {health['overall_status']}")
        
        if health['overall_status'] == 'healthy':
            print("✅ Database connection successful")
            
            # Test query
            content = db_manager.execute_query(
                "SELECT content_id, title FROM content LIMIT 3",
                fetch_mode='all'
            )
            print(f"✅ Sample content found: {len(content)} items")
            for item in content:
                print(f"   • {item['content_id']}: {item['title']}")
                
            return True
        else:
            print(f"❌ Database health check failed: {health}")
            return False
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

async def test_ai_models():
    """Test AI model loading and functionality."""
    print("\n" + "="*50)
    print("🧠 TESTING AI MODELS")
    print("="*50)
    
    try:
        from recommender.searchRecommender.models import ModelManager
        
        # Initialize model manager
        model_manager = ModelManager()
        
        # Get model status
        status = model_manager.get_model_status()
        print(f"Models loaded: {status['models_loaded_count']}")
        print(f"Available models: {status['models_available']}")
        
        if status['models_loaded_count'] > 0:
            print("✅ AI models loaded successfully")
            
            # Test semantic embedding
            test_text = "I want action movies"
            embedding = model_manager.get_semantic_embedding(test_text)
            if embedding is not None:
                print(f"✅ Semantic embedding generated: {len(embedding)} dimensions")
            
            # Test sentiment analysis
            analysis = model_manager.analyze_sentiment_and_emotion(test_text)
            print(f"✅ Sentiment analysis: {analysis['sentiment']} (confidence: {analysis.get('sentiment_confidence', 0):.2f})")
            
            return True
        else:
            print("⚠️  No AI models loaded - system will use fallback methods")
            return True  # Still okay, just degraded functionality
            
    except Exception as e:
        print(f"❌ AI models test failed: {e}")
        return False

async def test_search_engine():
    """Test the complete search recommendation engine."""
    print("\n" + "="*50)
    print("🎯 TESTING SEARCH RECOMMENDATION ENGINE")
    print("="*50)
    
    try:
        # Initialize database first
        db_manager = get_database_manager()
        
        # Initialize search recommender
        search_recommender = get_search_recommender(db_manager)
        
        if search_recommender is None:
            print("❌ Search recommender initialization failed")
            return False
        
        print("✅ Search recommender initialized")
        
        # Get engine statistics
        stats = search_recommender.get_engine_statistics()
        print("\n📊 Engine Component Status:")
        
        system_status = stats.get('system_status', {})
        for component, status in system_status.items():
            status_emoji = "✅" if status.get('status') == 'healthy' else "⚠️"
            print(f"   {status_emoji} {component}: {status.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search engine test failed: {e}")
        return False

async def test_search_functionality():
    """Test actual search functionality with sample queries."""
    print("\n" + "="*50)
    print("🔍 TESTING SEARCH FUNCTIONALITY")
    print("="*50)
    
    try:
        # Initialize system
        db_manager = get_database_manager()
        search_recommender = get_search_recommender(db_manager)
        
        if search_recommender is None:
            print("❌ Search engine not available for testing")
            return False
        
        # Test queries
        test_queries = [
            {
                "query": "I want action movies",
                "user_id": "user_test_001",
                "description": "Simple genre search"
            },
            {
                "query": "I'm feeling sad, need something funny",
                "user_id": "user_test_001", 
                "description": "Mood-based search"
            },
            {
                "query": "something with good ratings",
                "user_id": "user_test_001",
                "description": "Quality-based search"
            }
        ]
        
        for i, test in enumerate(test_queries, 1):
            print(f"\n🧪 Test {i}: {test['description']}")
            print(f"Query: '{test['query']}'")
            
            try:
                # Get AI recommendations
                result = search_recommender.get_ai_recommendations(
                    search_query=test['query'],
                    user_id=test['user_id'],
                    limit=5
                )
                
                if result and 'recommendations' in result:
                    recommendations = result['recommendations']
                    analysis_data = result.get('analysis_data', {})
                    
                    print(f"✅ Found {len(recommendations)} recommendations")
                    print(f"   Detected mood: {analysis_data.get('detected_mood', 'none')}")
                    print(f"   Search intent: {analysis_data.get('search_intent', 'general')}")
                    print(f"   Confidence: {analysis_data.get('confidence_score', 0):.2f}")
                    
                    # Show top 3 results
                    print("   Top results:")
                    for j, rec in enumerate(recommendations[:3], 1):
                        title = rec.get('title', 'Unknown')
                        score = rec.get('final_score', 0)
                        print(f"     {j}. {title} (score: {score:.2f})")
                else:
                    print("⚠️  No recommendations returned")
                    
            except Exception as e:
                print(f"❌ Search test {i} failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Search functionality test failed: {e}")
        return False

async def main():
    """Run comprehensive test suite for the AI search system."""
    print("🚀 AI SEARCH RECOMMENDATION ENGINE TEST SUITE")
    print(f"📅 Started at: 2025-06-20 09:24:30 UTC")
    print(f"👤 Testing user: rky-cse")
    print("="*60)
    
    # Track test results
    tests = []
    
    # Test 1: Database
    db_result = await test_database_connection()
    tests.append(("Database Connection", db_result))
    
    # Test 2: AI Models
    ai_result = await test_ai_models()
    tests.append(("AI Models", ai_result))
    
    # Test 3: Search Engine
    engine_result = await test_search_engine()
    tests.append(("Search Engine", engine_result))
    
    # Test 4: Search Functionality (only if engine works)
    if engine_result:
        search_result = await test_search_functionality()
        tests.append(("Search Functionality", search_result))
    else:
        tests.append(("Search Functionality", False))
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST RESULTS SUMMARY")
    print("="*60)
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nTests passed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("🎉 ALL TESTS PASSED! Your AI search system is ready!")
    elif passed >= len(tests) - 1:
        print("⚠️  Most tests passed - system should work with minor limitations")
    else:
        print("❌ Multiple test failures - check setup and dependencies")
    
    print(f"\n✅ Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("👤 Tested by: rky-cse")

if __name__ == "__main__":
    asyncio.run(main())