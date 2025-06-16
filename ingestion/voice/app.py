import os
import asyncio
import tempfile
import subprocess
import wave
from pathlib import Path

import numpy as np
import whisper
import aioredis
from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import Field
from pydantic_settings import BaseSettings
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect

app = FastAPI(title="Voice Prosody + Whisper Streaming API")

# Enable CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store raw streams
RECORD_DIR = Path("recordings")
RECORD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB per file

class Settings(BaseSettings):
    WHISPER_MODEL_SIZE: str = Field(..., env="WHISPER_MODEL_SIZE")
    OPENSMILE_BINARY: Path = Field(..., env="OPENSMILE_BINARY")
    OPENSMILE_CONFIG: Path = Field(..., env="OPENSMILE_CONFIG")
    REDIS_URL: str = Field(..., env="REDIS_URL")
    PROSODY_REDIS_CHANNEL: str = Field(..., env="PROSODY_REDIS_CHANNEL")
    TRANSCRIPTION_REDIS_CHANNEL: str = Field(..., env="TRANSCRIPTION_REDIS_CHANNEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
redis: aioredis.Redis = None
whisper_model = None


def log_published(channel: str, message: str):
    print(f"📤 Published to {channel}: {message}")

async def subscribe_and_log():
    pubsub = redis.pubsub()
    await pubsub.subscribe(settings.PROSODY_REDIS_CHANNEL, settings.TRANSCRIPTION_REDIS_CHANNEL)
    print(f"🔔 Subscribed to channels: {settings.PROSODY_REDIS_CHANNEL}, {settings.TRANSCRIPTION_REDIS_CHANNEL}")
    async for msg in pubsub.listen():
        if msg['type'] == 'message':
            print(f"📥 Received from {msg['channel']}: {msg['data']}")

@app.on_event("startup")
async def startup():
    global redis, whisper_model
    redis = await aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    whisper_model = whisper.load_model(settings.WHISPER_MODEL_SIZE)
    asyncio.create_task(subscribe_and_log())
    print("🚀 Startup complete: Redis & Whisper loaded")

@app.on_event("shutdown")
async def shutdown():
    if redis:
        await redis.close()
    print("👋 Shutdown: Redis connection closed")


def extract_prosody(wav_path: Path) -> str:
    try:
        proc = subprocess.run([
            str(settings.OPENSMILE_BINARY),
            "-C", str(settings.OPENSMILE_CONFIG),
            "-I", str(wav_path),
            "-csvoutput", "/dev/stdout",
            "-nologfile"
        ], capture_output=True, text=True, check=True)
        for line in proc.stdout.splitlines():
            if line and not line.startswith("#") and "," in line:
                return line
    except Exception as err:
        print(f"OpenSMILE error: {err}")
    return ""

async def wav_from_raw_pcm(raw_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_name = tmp.name
    tmp.close()
    with wave.open(tmp_name, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)
    return Path(tmp_name)

@app.websocket("/stream-prosody")
async def stream_prosody(ws: WebSocket):
    file_index = 1
    current_file = RECORD_DIR / f"stream_{file_index}.pcm"
    fw = open(current_file, 'ab')
    await ws.accept()
    try:
        while True:
            data = await ws.receive_bytes()
            fw.write(data)
            if fw.tell() >= MAX_FILE_SIZE:
                fw.close()
                file_index += 1
                current_file = RECORD_DIR / f"stream_{file_index}.pcm"
                fw = open(current_file, 'ab')

            wav_path = await wav_from_raw_pcm(data)
            prosody_line = await asyncio.get_running_loop().run_in_executor(None, extract_prosody, wav_path)
            if prosody_line:
                await redis.publish(settings.PROSODY_REDIS_CHANNEL, prosody_line)
                log_published(settings.PROSODY_REDIS_CHANNEL, prosody_line)
            wav_path.unlink(missing_ok=True)

            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            result = whisper_model.transcribe(audio, fp16=False, language="en")
            text = result.get("text", "").strip()
            if text:
                await redis.publish(settings.TRANSCRIPTION_REDIS_CHANNEL, text)
                log_published(settings.TRANSCRIPTION_REDIS_CHANNEL, text)
    except WebSocketDisconnect:
        print("🛑 Client disconnected gracefully")
    except Exception as e:
        print(f"⚠️ WebSocket error: {e}")
    finally:
        fw.close()

@app.post("/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    ctype = file.content_type or ""
    ext = Path(file.filename or "").suffix.lower()
    if ctype and not ctype.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Please upload an audio file")
    if not ctype and ext not in [".wav", ".mp3", ".flac", ".m4a"]:
        raise HTTPException(status_code=400, detail="Please upload a valid audio file")
    contents = await file.read()
    tmp = tempfile.NamedTemporaryFile(suffix=ext or ".wav", delete=False)
    tmp_name = tmp.name
    try:
        tmp.write(contents)
        tmp.flush()
        res = whisper_model.transcribe(tmp_name)
        transcript = res.get("text", "").strip()
        print(f"📤 Published to {settings.TRANSCRIPTION_REDIS_CHANNEL}: {transcript}")
        return JSONResponse({"transcript": transcript})
    except Exception as e:
        print(f"Transcribe-file error: {e}")
        raise HTTPException(status_code=500, detail="Transcription failed")
    finally:
        tmp.close()
        os.unlink(tmp_name)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        ws_max_size=20_000_000,
        ws_ping_interval=None,
        ws_ping_timeout=None,
        timeout_keep_alive=600,  # keep idle connections alive for 10 minutes
    )
