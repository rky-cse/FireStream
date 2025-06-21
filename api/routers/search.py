import logging
import json
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Optional
import os
import sys

# Add the project root to path (two levels up from api/routers)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from recommender.searchRecommender.engine import RecommendationEngine, SearchResult
    logger = logging.getLogger("search_router")
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise

logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

router = APIRouter()

# WebSocket connections tracking
connections: Dict[str, Dict] = {}

# Initialize engine instance
_engine = None

def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", 5432)),
            "database": os.getenv("DB_NAME", "firestream_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres")
        }
        es_host = os.getenv("ES_HOST", "http://localhost:9200")
        _engine = RecommendationEngine(db_config, es_host)
        logger.info("RecommendationEngine initialized successfully")
    return _engine

@router.websocket("/ws/{client_id}/search")
async def websocket_search_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = {
        "websocket": websocket,
        "connected_at": datetime.now()
    }
    logger.info(f"Client {client_id} connected to search endpoint")

    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                msg = json.loads(raw_data)
                
                if msg.get("type") == "search_request":
                    user_id = 'USR10001'
                    query_text = msg.get("search_content", "")
                    prosody_data = msg.get("prosody", {})

                    logger.info(f"Processing search request from {client_id} for user {user_id}")

                    engine = get_engine()
                    results = engine.get_recommendations(
                        user_id=user_id,
                        search_text=query_text,
                        prosody=prosody_data
                    )

                    response = {
                        "type": "search_result",
                        "timestamp": datetime.now().isoformat(),
                        "results": {
                            "hits": [{
                                "content_id": r.content_id,
                                "title": r.title,
                                "score": float(r.score),
                                "match_reasons": r.match_reasons,
                                "metadata": r.metadata
                            } for r in results],
                            "count": len(results)
                        }
                    }
                    await websocket.send_text(json.dumps(response, default=str))

                else:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Invalid message type"
                    }))

            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                }))

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    finally:
        connections.pop(client_id, None)

@router.get("/health")
async def health_check():
    return {"status": "healthy"}