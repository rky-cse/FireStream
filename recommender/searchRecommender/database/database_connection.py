import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
import logging
from contextlib import contextmanager
import json

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class DatabaseConnection:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'database': os.getenv('DB_NAME', 'video_recommender'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'port': os.getenv('DB_PORT', '5432')
        }
        
    def get_connection(self):
        """Get database connection"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            conn.set_session(autocommit=False)
            return conn
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    def get_dict_cursor(self):
        """Get connection with dictionary cursor"""
        try:
            conn = psycopg2.connect(**self.connection_params)
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            return conn, cur
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise
    
    @contextmanager
    def get_cursor(self, dict_cursor=False):
        """Context manager for database operations"""
        conn = None
        cur = None
        try:
            if dict_cursor:
                conn, cur = self.get_dict_cursor()
            else:
                conn = self.get_connection()
                cur = conn.cursor()
            
            yield conn, cur
            conn.commit()
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
    
    def execute_query(self, query, params=None, fetch_one=False, fetch_all=False):
        """Execute a query and return results"""
        with self.get_cursor() as (conn, cur):
            cur.execute(query, params)
            
            if fetch_one:
                return cur.fetchone()
            elif fetch_all:
                return cur.fetchall()
            else:
                return cur.rowcount

# Global instance
db = DatabaseConnection()