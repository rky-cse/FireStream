# api/routers/voice.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import whisper

from ingestion.voice.voiceProsodyExtractor import (
    transcribe_audio,
    extract_prosody,
)

# ——— new imports ———
from searchRecommender.engine import RecommendationEngine
from starlette.concurrency import run_in_threadpool
# ————————————

# Load env vars (for any OPENSMILE paths, etc.)
load_dotenv()

# Logger
logger = logging.getLogger("voice_router")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)

# Whisper model
try:
    whisper_model = whisper.load_model("base")
    logger.info("Whisper model loaded in voice router")
except Exception as e:
    whisper_model = None
    logger.error(f"Failed to load Whisper: {e}")

# Initialize recommendation engine (for search feature)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "database": os.getenv("DB_NAME", "firestream_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
try:
    engine = RecommendationEngine(DB_CONFIG, es_host=ES_HOST)
    logger.info("Recommendation engine initialized in voice router")
except Exception as e:
    engine = None
    logger.error(f"Failed to initialize recommendation engine: {e}")

router = APIRouter()

# In‑memory storage
connections: dict = {}
audio_sessions: dict = {}


@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = {
        "websocket": websocket,
        "audio_buffer": np.array([], dtype=np.float32),
        "connected_at": datetime.now()
    }
    logger.info(f"Client {client_id} connected")

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            # Handle search requests asynchronously
            if msg.get("type") == "search":
                if engine is None:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "message": "Search engine unavailable"
                    }))
                    continue

                search_text = msg.get("query", "")
                logger.info(f"Received search: {search_text!r}")

                # Immediately acknowledge search request
                await websocket.send_text(json.dumps({
                    "type": "search_ack",
                    "timestamp": datetime.now().isoformat(),
                    "message": "Search request received, processing..."
                }))

                # Run recommendation engine without prosody in threadpool
                results = await run_in_threadpool(
                    engine.get_recommendations,
                    'USR10001',
                    search_text
                )
                print("Search results:", results)
                # Send back final results
                await websocket.send_text(json.dumps({
                    "type": "search_results",
                    "timestamp": datetime.now().isoformat(),
                    "results": [
                        {
                            "content_id": r.content_id,
                            "title": r.title,
                            "score": r.score,
                            "match_reasons": r.match_reasons,
                            "metadata": r.metadata
                        } for r in results
                    ]
                }))
                continue

            # Append incoming audio
            if msg.get("type") == "audio_data":
                arr = np.array(msg["data"], dtype=np.float32)
                sr = msg.get("sampleRate", 16000)
                buf = connections[client_id]["audio_buffer"]
                connections[client_id]["audio_buffer"] = np.concatenate([buf, arr])

                # Process every 3s of audio
                if len(connections[client_id]["audio_buffer"]) >= sr * 3:
                    await process_audio_buffer(client_id, sr)

            # Final chunk
            elif msg.get("type") == "audio_end":
                sr = msg.get("sampleRate", 16000)
                await process_audio_buffer(client_id, sr, final=True)

            # Prosody analysis request
            elif msg.get("type") == "analyze_prosody":
                text = msg.get("text", "")
                session_id = msg.get("session_id", "")
                sr = msg.get("sampleRate", 16000)
                await analyze_with_prosody_by_session(client_id, text, session_id, sr)

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Error on WebSocket ({client_id}): {e}")
    finally:
        connections.pop(client_id, None)
        logger.info(f"Cleaned up client {client_id}")


async def process_audio_buffer(client_id: str, sample_rate: int, final: bool = False):
    conn = connections.get(client_id)
    if not conn:
        return

    audio = conn["audio_buffer"]
    if len(audio) < 0.5 * sample_rate:
        return

    session_id = f"{client_id}_{int(datetime.now().timestamp()*1000)}"
    audio_sessions[session_id] = {
        "audio_data": audio.copy(),
        "sample_rate": sample_rate,
        "timestamp": datetime.now().isoformat(),
        "client_id": client_id
    }

    # Transcribe
    text = transcribe_audio(audio, whisper_model, sample_rate)
    if text and not text.startswith(("Audio", "Transcription failed")):
        resp = {
            "type": "transcription",
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "sample_count": len(audio),
            "duration": len(audio)/sample_rate,
            "session_id": session_id
        }
        await conn["websocket"].send_text(json.dumps(resp))
        logger.info(f"Sent transcription to {client_id}: {text[:30]}…")
    else:
        err_msg = text
        if not err_msg.startswith("No speech detected"):
            await conn["websocket"].send_text(json.dumps({
                "type": "error",
                "message": err_msg,
                "timestamp": datetime.now().isoformat()
            }))

    # Reset or trim buffer
    if final:
        conn["audio_buffer"] = np.array([], dtype=np.float32)
    else:
        keep = sample_rate * 1
        conn["audio_buffer"] = audio[-keep:] if len(audio) > keep else audio

    # Prune old sessions
    if len(audio_sessions) > 15:
        for sid in sorted(audio_sessions)[:-15]:
            audio_sessions.pop(sid, None)


async def analyze_with_prosody_by_session(client_id: str, text: str, session_id: str, sample_rate: int):
    conn = connections.get(client_id)
    if not conn:
        return

    # Retrieve audio
    if session_id and session_id in audio_sessions:
        data = audio_sessions[session_id]
        audio = data["audio_data"]
        sr = data["sample_rate"]
    else:
        audio = conn["audio_buffer"]
        sr = sample_rate

    if len(audio) < 0.5 * sr:
        available = list(audio_sessions.keys())[-5:]
        await send_error(client_id,
            f"Not enough audio for prosody (session: {session_id}, available: {available})"
        )
        return

    feats = extract_prosody(audio, sr)
    summary = {
        "text": text,
        "word_count": len(text.split()),
        "character_count": len(text),
        "timestamp": datetime.now().isoformat(),
        "audio_duration": len(audio)/sr,
        "sample_count": len(audio),
        "session_id": session_id,
        "prosody_features": feats
    }

    if feats and not feats.get("error"):
        pitch = {k: v for k,v in feats.items() if any(t in k.lower() for t in ["pitch","f0","voicing"])}
        energy = {k: v for k,v in feats.items() if any(t in k.lower() for t in ["energy","rms","power"])}
        spectral = {k: v for k,v in feats.items() if "spectral" in k.lower()}
        temporal = {k: v for k,v in feats.items() if any(t in k.lower() for t in ["duration","rate","zero_crossing"])}
        summary["prosody_summary"] = {
            "pitch_features_count": len(pitch),
            "energy_features_count": len(energy),
            "spectral_features_count": len(spectral),
            "temporal_features_count": len(temporal),
            "total_features": len(feats),
            "has_prosody_data": True,
            "feature_extraction_success": True,
            "extraction_method": feats.get("method", "unknown")
        }

        keys = {}
        for name, val in feats.items():
            if isinstance(val, (int, float)) and any(t in name.lower() for t in ["pitch","f0","energy","rms","spectral_centroid","duration","zero_crossing","estimated"]):
                clean = name.replace("_", " ").title()
                keys[clean] = round(val, 3)
            if len(keys) >= 10:
                break
        summary["key_features"] = keys
    else:
        summary["prosody_summary"] = {
            "pitch_features_count": 0,
            "energy_features_count": 0,
            "spectral_features_count": 0,
            "temporal_features_count": 0,
            "total_features": 0,
            "has_prosody_data": False,
            "feature_extraction_success": False,
            "error_info": feats.get("error", "Unknown"),
            "extraction_method": feats.get("method", "failed")
        }
        summary["key_features"] = {}

    await conn["websocket"].send_text(json.dumps({
        "type": "prosody_analysis",
        "analysis": summary
    }))
    logger.info(f"Sent prosody analysis to {client_id} (session: {session_id})")


async def send_error(client_id: str, message: str):
    conn = connections.get(client_id)
    if conn:
        await conn["websocket"].send_text(json.dumps({
            "type": "error",
            "message": message,
            "timestamp": datetime.now().isoformat()
        }))
