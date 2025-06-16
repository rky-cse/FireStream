import asyncio
import sounddevice as sd
import websockets
import json
import time

WS_URL      = "ws://localhost:8000/stream-prosody"
SAMPLE_RATE = 16000      # 16 kHz
CHANNELS    = 1          # mono
CHUNK_SEC   = 1          # 1 second per chunk

async def stream_prosody():
    # Open the microphone once, reuse the RawInputStream across reconnects
    stream = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='int16'
    )
    with stream:
        while True:
            try:
                async with websockets.connect(WS_URL) as ws:
                    print(f"[{time.strftime('%H:%M:%S')}] Connected to server")
                    # Continuously read & send until server closes
                    while True:
                        frames, _ = stream.read(CHUNK_SEC * SAMPLE_RATE)
                        await ws.send(bytes(frames))     # send binary PCM
                        resp = await ws.recv()           # may raise ConnectionClosedOK
                        data = json.loads(resp)
                        print("Prosody chunk:", data)
            except websockets.exceptions.ConnectionClosedOK:
                # Server closed with code 1000 (normal closure)
                print(f"[{time.strftime('%H:%M:%S')}] Connection closed by server, reconnecting…")
                await asyncio.sleep(0.5)  # slight back‑off before reconnect
            except Exception as e:
                # Other errors (network, parsing, etc.)
                print("Error:", e)
                await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(stream_prosody())
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down.")
