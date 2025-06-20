import logging
import json
from datetime import datetime
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

# Assume there's a function called do_search in this file:
# from recommender.searchRecommendor.engine import do_search
# Example signature:
# def do_search(user_id: str, query_text: str, prosody_data: dict) -> dict:
#     # Implementation in engine.py
#     return {"hits": [...], "metadata": {...}}

try:
    from recommender.searchRecommendor.engine import do_search
except ImportError:
    # Fallback if engine is not actually present; just a placeholder
    def do_search(user_id: str, query_text: str, prosody_data: dict) -> dict:
        return {
            "hits": [{"title": "Sample Result", "score": 0.9}],
            "info": "Placeholder do_search result"
        }

logger = logging.getLogger("search_router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

router = APIRouter()

# Same pattern as in voice.py but for search tasks
connections = {}

@router.websocket("/ws/{client_id}/search")
async def websocket_search_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for handling search requests. It is meant to run in
    parallel with the voice WebSocket, allowing the user to confirm or edit 
    transcribed text and then hit 'enter' to initiate a search request. 
    If prosody data is available, it's included; otherwise a default or empty 
    prosody dict is used.
    """
    await websocket.accept()
    connections[client_id] = {
        "websocket": websocket,
        "connected_at": datetime.now()
    }
    logger.info(f"Client {client_id} connected to search endpoint")

    try:
        while True:
            raw_data = await websocket.receive_text()
            msg = json.loads(raw_data)

            # Check for search request
            if msg.get("type") == "search_request":
                user_id = msg.get("userId")
                query_text = msg.get("search_content", "")
                prosody_data = msg.get("prosody") or {}

                logger.info(
                    f"Received search_request from {client_id} "
                    f"(user_id: {user_id} / text: {query_text[:30]}...)"
                )

                # If no prosody provided, use default or empty structure
                if not prosody_data:
                    prosody_data = {"pitch_mean": 0.0, "energy_mean": 0.0}

                # Call the actual search engine function
                try:
                    results = do_search(user_id, query_text, prosody_data)
                except Exception as engine_err:
                    logger.error(f"Search engine error: {engine_err}")
                    results = {"error": str(engine_err)}

                # Send back the result
                response = {
                    "type": "search_result",
                    "timestamp": datetime.now().isoformat(),
                    "results": results
                }
                await websocket.send_text(json.dumps(response))

            # Could handle other message types here if needed
            else:
                logger.warning(f"Unknown message type from {client_id}: {msg}")

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected from search endpoint")
    except Exception as e:
        logger.error(f"Error on WebSocket ({client_id}): {e}")
    finally:
        connections.pop(client_id, None)
        logger.info(f"Cleaned up data for client {client_id} from search endpoint")