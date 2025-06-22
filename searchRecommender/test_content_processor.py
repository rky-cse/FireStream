import time
from database import DatabaseManager
from contentProcessor import ContentProcessor

def quick_test():
    db = DatabaseManager()
    processor = ContentProcessor(db_connection=db)
    
    # 1. Verify basic genre search
    start = time.time()
    genres = processor.get_genre_recommendations(["Comedy"], limit=1)
    print(f"Genre search took {time.time()-start:.2f}s. Results: {genres}")
    
    # 2. Test with minimal data requirements
    with db.get_cursor() as (conn, cur):
        cur.execute("""
            INSERT INTO content (content_id, title, genre, rating)
            VALUES ('test123', 'Test Movie', ARRAY['Comedy'], 4.0)
            ON CONFLICT (content_id) DO NOTHING
        """)
        conn.commit()
    
    # 3. Should return instantly with the test record
    start = time.time()
    test_result = processor.get_genre_recommendations(["Comedy"], limit=1)
    print(f"Test query took {time.time()-start:.2f}s. Should return ['test123']:", test_result)

if __name__ == "__main__":
    quick_test()