import asyncio
import sys
import wave
import numpy as np
import requests
import aioredis
import websockets
import sounddevice as sd
import tempfile
import scipy.io.wavfile as wavfile
import subprocess
import platform

# Usage:
#   python test_client.py [--record duration]
#   python test_client.py /path/to/sample.wav
# Ensure your FastAPI app is running on localhost:8000 and ffmpeg is installed

REDIS_URL = "redis://localhost:6379/0"
WS_URL = "ws://localhost:8000/stream-prosody"
HTTP_URL = "http://localhost:8000/transcribe-file"


def record_to_wav(duration: float, fs: int = 16000, channels: int = 1) -> str:
    """
    Record audio via sounddevice or fallback to ffmpeg, save to temp WAV file.
    Returns the filename.
    """
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    filename = tmp.name
    tmp.close()
    # Try using sounddevice
    try:
        print(f"Recording via sounddevice for {duration} seconds...")
        data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype='int16')
        sd.wait()
        wavfile.write(filename, fs, data)
        print(f"Saved recording to '{filename}' (sounddevice)")
        return filename
    except Exception as e:
        print(f"sounddevice recording failed ({e}), falling back to ffmpeg...")

    # Fallback: use ffmpeg CLI
    system = platform.system()
    if system == 'Windows':
        fmt = 'dshow'
        device = 'audio="default"'
    elif system == 'Linux':
        fmt = 'alsa'
        device = 'default'
    else:
        fmt = 'alsa'
        device = 'default'

    cmd = [
        'ffmpeg', '-y',
        '-f', fmt,
        '-i', device,
        '-t', str(duration),
        '-ar', str(fs),
        '-ac', str(channels),
        '-sample_fmt', 's16',
        filename
    ]
    print("Running ffmpeg command:", ' '.join(cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Saved recording to '{filename}' (ffmpeg)")
        return filename
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg recording failed: {e}")
        sys.exit(1)

async def test_http(wav_path):
    print("\n[HTTP TEST] Uploading file for transcription...")
    with open(wav_path, 'rb') as f:
        files = {'file': f}
        resp = requests.post(HTTP_URL, files=files)
    if resp.status_code == 200:
        print("Transcript:", resp.json()['transcript'])
    else:
        print("HTTP error:", resp.status_code, resp.text)

async def test_stream(wav_path):
    print("\n[STREAM TEST] Subscribing to Redis channels... (prosody & transcript)")
    redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe('voice:prosody', 'voice:transcript')
    
    async def reader():
        async for msg in pubsub.listen():
            if msg['type'] == 'message':
                print(f"[REDIS:{msg['channel']}] {msg['data']}")

    async def sender():
        print("Connecting to WebSocket...")
        async with websockets.connect(WS_URL, max_size=10_000_000) as ws:
            with wave.open(wav_path, 'rb') as wf:
                sr = wf.getframerate()
                chunk_frames = sr
                while True:
                    frames = wf.readframes(chunk_frames)
                    if not frames:
                        break
                    await ws.send(frames)
                    await asyncio.sleep(1)
        print("Finished sending audio chunks.")

    await asyncio.gather(reader(), sender())

async def main():
    # Determine source: record or file
    if len(sys.argv) == 3 and sys.argv[1] == '--record':
        try:
            duration = float(sys.argv[2])
        except ValueError:
            print("Invalid duration. Must be a number.")
            sys.exit(1)
        wav = record_to_wav(duration)
    elif len(sys.argv) == 2:
        wav = sys.argv[1]
    else:
        print("Usage:")
        print("  python test_client.py --record <duration_secs>")
        print("  python test_client.py <path_to_wav>")
        sys.exit(1)

    await test_http(wav)
    await test_stream(wav)

if __name__ == '__main__':
    asyncio.run(main())
