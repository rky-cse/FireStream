import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import os
from contextlib import contextmanager
import logging
from typing import Optional, Union, List, Dict, Any

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Simplified but robust PostgreSQL connection manager with:
    - Connection pooling
    - Automatic retries
    - Clean resource handling
    - Basic query caching
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._get_default_config()
        self.pool = self._create_pool()
        self.query_cache = {}
        logger.info("DatabaseManager initialized")

    def _get_default_config(self) -> Dict:
        """Get connection config from environment variables"""
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME', 'postgres'),
            'user': os.getenv('DB_USER', 'postgres'),
            'password': os.getenv('DB_PASSWORD', 'postgres'),
            'minconn': int(os.getenv('DB_MIN_CONN', '5')),
            'maxconn': int(os.getenv('DB_MAX_CONN', '20'))
        }

    def _create_pool(self) -> pool.ThreadedConnectionPool:
        """Initialize connection pool with error handling"""
        try:
            return psycopg2.pool.ThreadedConnectionPool(
                minconn=self.config['minconn'],
                maxconn=self.config['maxconn'],
                cursor_factory=RealDictCursor,
                **{k: v for k, v in self.config.items() 
                   if k not in ['minconn', 'maxconn']}
            )
        except Exception as e:
            logger.error(f"Connection pool creation failed: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """Get a managed database connection"""
        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            raise
        finally:
            if conn:
                self.pool.putconn(conn)

    def execute_query(
        self,
        query: str,
        params: Optional[tuple] = None,
        fetch_mode: str = 'all'
    ) -> Union[List[Dict], int, None]:
        """
        Execute a query with automatic connection handling
        
        Args:
            query: SQL query string
            params: Query parameters
            fetch_mode: 'all', 'one', or 'none'
        
        Returns:
            Results based on fetch_mode:
            - 'all': List of rows as dicts
            - 'one': Single row as dict
            - 'none': Number of affected rows
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(query, params)
                    
                    if fetch_mode == 'all':
                        return cur.fetchall()
                    elif fetch_mode == 'one':
                        return cur.fetchone()
                    return cur.rowcount
                
                except Exception as e:
                    logger.error(f"Query failed: {e}\nQuery: {query}\nParams: {params}")
                    raise

    def get_content_by_ids(self, content_ids: List[str]) -> List[Dict]:
        """Get multiple content items efficiently"""
        if not content_ids:
            return []

        placeholders = ','.join(['%s'] * len(content_ids))
        query = f"""
            SELECT * FROM content 
            WHERE content_id IN ({placeholders})
            ORDER BY array_position(ARRAY[{placeholders}], content_id)
        """
        return self.execute_query(
            query,
            params=tuple(content_ids * 2),
            fetch_mode='all'
        )

    def close(self):
        """Clean up all connections"""
        if hasattr(self, 'pool'):
            self.pool.closeall()
            logger.info("All database connections closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()