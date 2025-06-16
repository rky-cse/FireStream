import asyncio
import uuid
import logging
import json
import tempfile
from pathlib import Path
from typing import AsyncGenerator

import whisper
import aioredis
import ffmpeg
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings


# 1. Configuration
class Settings(BaseSettings):
    whisper_model_size: str = "base"
    opensmile_binary: str     # e.g. "/usr/local/bin/SMILExtract"
    opensmile_config: str     # e.g. "/app/opensmile/config/prosodyShs.conf"
    redis_url: str = "redis://localhost:6379/0"
    prosody_redis_channel: str = "voice:prosody"
    transcription_redis_channel: str = "voice:transcript"

    class Config:
        env_file = ".env"

settings = Settings()

# 2. Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("voice-ingestion")

# 3. FastAPI app
app = FastAPI(title="Advanced Voice Ingestion Service")

# 4. Load model & Redis at startup
@app.on_event("startup")
async def startup_event():
    global whisper_model, redis
    logger.info("Loading Whisper model: %s", settings.whisper_model_size)
    whisper_model = whisper.load_model(settings.whisper_model_size)
    logger.info("Connecting to Redis: %s", settings.redis_url)
    redis = await aioredis.from_url(settings.redis_url)

# 5. Response schema
class VoiceFeatures(BaseModel):
    transcript: str
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float

# 6. Utility: normalize & convert any audio to WAV 16kHz mono
def convert_to_wav16k(input_path: Path, output_path: Path):
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), format="wav", acodec="pcm_s16le", ac=1, ar="16k")
        .overwrite_output()
        .run(quiet=True)
    )

# 7. Batch analysis endpoint
@app.post("/analyze", response_model=VoiceFeatures)
async def analyze_voice(file: UploadFile = File(...)):
    tmp_id = uuid.uuid4().hex
    raw_path = Path(tempfile.gettempdir()) / f"{tmp_id}{Path(file.filename).suffix}"
    wav_path = raw_path.with_suffix(".wav")
    csv_path = raw_path.with_suffix(".csv")

    # 7.1 Save upload
    contents = await file.read()
    raw_path.write_bytes(contents)
    logger.info("Saved upload to %s", raw_path)

    # 7.2 Convert to WAV 16k mono
    try:
        convert_to_wav16k(raw_path, wav_path)
        raw_path.unlink(missing_ok=True)
    except Exception as e:
        logger.error("FFmpeg conversion failed: %s", e)
        raise HTTPException(status_code=400, detail="Invalid audio format")

    # 7.3 Whisper transcription (sync, CPU/GPU)
    result = whisper_model.transcribe(str(wav_path))
    transcript = result["text"].strip()
    logger.info("Transcription complete (%d chars)", len(transcript))

    # 7.4 Prosody extraction via async subprocess
    proc = await asyncio.create_subprocess_exec(
        settings.opensmile_binary,
        "-C", settings.opensmile_config,
        "-I", str(wav_path),
        "-O", str(csv_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        logger.error("OpenSMILE failed: %s", stderr.decode())
        raise HTTPException(status_code=500, detail="Prosody extraction error")

    # 7.5 Parse prosody CSV
    data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
    pitch = data.get("F0semitoneFrom27.5Hz_sma3nz", np.array([]))
    energy = data.get("pcm_LOGenergy_sma3", np.array([]))
    features = VoiceFeatures(
        transcript=transcript,
        pitch_mean=float(np.nanmean(pitch)),
        pitch_std=float(np.nanstd(pitch)),
        energy_mean=float(np.nanmean(energy)),
        energy_std=float(np.nanstd(energy)),
    )
    logger.info("Prosody features computed")

    # 7.6 Push to Redis (raw JSON) for downstream consumers
    await redis.publish(
        settings.transcription_redis_channel,
        json.dumps({"session": tmp_id, "transcript": transcript})
    )
    await redis.publish(
        settings.prosody_redis_channel,
        json.dumps({"session": tmp_id, **features.dict()})
    )

    # 7.7 Cleanup
    await asyncio.sleep(0)  # yield
    wav_path.unlink(missing_ok=True)
    csv_path.unlink(missing_ok=True)

    return features

# 8. WebSocket streaming endpoint (prosody only)
# … (keep your imports and startup code as-is) …

@app.websocket("/stream-prosody")
async def ws_prosody(websocket: WebSocket):
    """
    Client streams raw 1-second WAV chunks;
    server returns per-chunk pitch & energy stats.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted: %s", websocket.client)

    try:
        async for chunk in websocket.iter_bytes():
            # 1) Create a real temp .wav file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
                wav_path = Path(wav_tmp.name)
                wav_tmp.write(chunk)

            # 2) Create corresponding temp .csv path
            csv_path = wav_path.with_suffix(".csv")

            # 3) Run OpenSMILE on that file
            proc = await asyncio.create_subprocess_exec(
                settings.opensmile_binary,
                "-C", settings.opensmile_config,
                "-I", str(wav_path),
                "-O", str(csv_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning("Chunk prosody failed: %s", stderr.decode(errors="ignore"))
                # Clean up and continue
                wav_path.unlink(missing_ok=True)
                continue

            # 4) Parse the CSV
            data = np.genfromtxt(str(csv_path), delimiter=",", names=True)
            pitch = data.get("F0semitoneFrom27.5Hz_sma3nz", np.array([]))
            energy = data.get("pcm_LOGenergy_sma3", np.array([]))
            resp = {
                "pitch_mean": float(np.nanmean(pitch)),
                "pitch_std":   float(np.nanstd(pitch)),
                "energy_mean": float(np.nanmean(energy)),
                "energy_std":  float(np.nanstd(energy)),
            }

            # 5) Send the JSON response and publish
            await websocket.send_json(resp)
            await redis.publish(settings.prosody_redis_channel, json.dumps(resp))

            # 6) Cleanup temp files
            wav_path.unlink(missing_ok=True)
            csv_path.unlink(missing_ok=True)

    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        await websocket.close()
        logger.info("WebSocket connection closed")


# 9. Healthcheck
@app.get("/health")
async def health():
    return JSONResponse({"status": "ok"})
