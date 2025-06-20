# api/routers/health.py

from fastapi import APIRouter
from datetime import datetime
import os

from ingestion.voice.voiceProsodyExtractor import try_different_opensmile_configs
from api.routers.voice import connections, audio_sessions, whisper_model

router = APIRouter(prefix="/health")


@router.get("")
async def health():
    return {
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "opensmile_configs": try_different_opensmile_configs(),
        "active_connections": len(connections),
        "stored_sessions": len(audio_sessions),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/sessions")
async def debug_sessions():
    info = {
        sid: {
            "sample_count": len(d["audio_data"]),
            "duration": len(d["audio_data"]) / d["sample_rate"],
            "timestamp": d["timestamp"],
            "client_id": d["client_id"]
        }
        for sid, d in audio_sessions.items()
    }
    return {
        "total_sessions": len(info),
        "sessions": info
    }
