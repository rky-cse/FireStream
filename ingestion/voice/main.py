from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import tempfile
import subprocess
import os
import wave
import struct
from datetime import datetime
from dotenv import load_dotenv
import base64
import numpy as np

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENSMILE_BINARY = os.getenv("OPENSMILE_BINARY")
OPENSMILE_CONFIG = os.getenv("OPENSMILE_CONFIG")

# Simple connection storage
connections = {}

# Load Whisper
try:
    import whisper
    whisper_model = whisper.load_model("base")
    logger.info("Whisper loaded successfully")
except Exception as e:
    logger.error(f"Whisper failed to load: {e}")
    whisper_model = None

def create_wav_from_float32_array(float32_array, sample_rate=16000):
    """Create WAV file from float32 audio array"""
    try:
        # Convert float32 to int16
        int16_array = (float32_array * 32767).astype(np.int16)
        
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name
        
        # Write WAV file
        with wave.open(wav_path, 'wb') as wav:
            wav.setnchannels(1)  # Mono
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(int16_array.tobytes())
        
        logger.debug(f"Created WAV file: {wav_path}, samples: {len(int16_array)}")
        return wav_path
        
    except Exception as e:
        logger.error(f"Error creating WAV file: {e}")
        return None

def transcribe_audio(audio_data, sample_rate=16000):
    """Transcribe audio from float32 array"""
    if not whisper_model:
        return "Whisper not available"
    
    if len(audio_data) < 1600:  # Less than 0.1 seconds at 16kHz
        return "Audio too short"
    
    try:
        # Create WAV file from float32 array
        wav_path = create_wav_from_float32_array(audio_data, sample_rate)
        if not wav_path:
            return "Failed to create audio file"
        
        # Check if WAV file exists and has content
        if not os.path.exists(wav_path) or os.path.getsize(wav_path) < 100:
            return "Invalid audio file"
        
        # Transcribe
        result = whisper_model.transcribe(wav_path)
        
        # Clean up
        os.unlink(wav_path)
        
        text = result["text"].strip()
        return text if text else "No speech detected"
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Transcription failed: {str(e)}"

def extract_prosody(audio_data, sample_rate=16000):
    """Extract prosody features from float32 audio array"""
    if not OPENSMILE_BINARY or not os.path.exists(OPENSMILE_BINARY):
        return {"error": "OpenSMILE not available"}
    
    try:
        # Create WAV file from float32 array
        wav_path = create_wav_from_float32_array(audio_data, sample_rate)
        if not wav_path:
            return {"error": "Failed to create audio file"}
        
        # Create output file for OpenSMILE
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as output_file:
            output_path = output_file.name
        
        # Run OpenSMILE
        cmd = [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-O", output_path]
        result = subprocess.run(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE, 
            timeout=15,
            text=True
        )
        
        features = {}
        if result.returncode == 0 and os.path.exists(output_path):
            with open(output_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    headers = lines[0].strip().split(',')
                    values = lines[1].strip().split(',')
                    
                    # Create feature dictionary, skip first column
                    for i, (header, value) in enumerate(zip(headers[1:], values[1:])):
                        try:
                            features[header] = float(value) if value and value != '?' else 0.0
                        except ValueError:
                            features[header] = value
        else:
            logger.error(f"OpenSMILE failed: {result.stderr}")
        
        # Clean up files
        os.unlink(wav_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
        
        return features
        
    except Exception as e:
        logger.error(f"Prosody extraction error: {e}")
        return {"error": str(e)}

@app.websocket("/ws/{client_id}")
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
            message = json.loads(data)
            
            if message["type"] == "audio_data":
                # Receive raw float32 audio data
                audio_array = np.array(message["data"], dtype=np.float32)
                sample_rate = message.get("sampleRate", 16000)
                
                # Append to buffer
                connections[client_id]["audio_buffer"] = np.concatenate([
                    connections[client_id]["audio_buffer"], 
                    audio_array
                ])
                
                # Process if buffer is large enough (3 seconds worth)
                buffer_size = len(connections[client_id]["audio_buffer"])
                if buffer_size >= sample_rate * 3:  # 3 seconds
                    await process_audio_buffer(client_id, sample_rate)
            
            elif message["type"] == "audio_end":
                sample_rate = message.get("sampleRate", 16000)
                await process_audio_buffer(client_id, sample_rate, final=True)
                
            elif message["type"] == "analyze_prosody":
                text = message.get("text", "")
                sample_rate = message.get("sampleRate", 16000)
                await analyze_with_prosody(client_id, text, sample_rate)
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        if client_id in connections:
            del connections[client_id]
        logger.info(f"Client {client_id} cleaned up")

async def process_audio_buffer(client_id, sample_rate=16000, final=False):
    """Process accumulated audio buffer"""
    if client_id not in connections:
        return
    
    connection = connections[client_id]
    audio_buffer = connection["audio_buffer"]
    
    if len(audio_buffer) < sample_rate * 0.5:  # Less than 0.5 seconds
        return
    
    try:
        logger.info(f"Processing {len(audio_buffer)} audio samples for {client_id} (duration: {len(audio_buffer)/sample_rate:.1f}s)")
        
        # Transcribe the audio
        text = transcribe_audio(audio_buffer, sample_rate)
        
        if text and not text.startswith("Audio") and not text.startswith("Transcription failed"):
            # Send transcription back
            response = {
                "type": "transcription",
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "sample_count": len(audio_buffer),
                "duration": len(audio_buffer) / sample_rate
            }
            
            await connection["websocket"].send_text(json.dumps(response))
            logger.info(f"Sent transcription to {client_id}: {text[:50]}...")
        else:
            logger.warning(f"Transcription failed for {client_id}: {text}")
            if not text.startswith("No speech detected"):
                # Send error only if it's not just silence
                error_response = {
                    "type": "error",
                    "message": text,
                    "timestamp": datetime.now().isoformat()
                }
                await connection["websocket"].send_text(json.dumps(error_response))
        
        # Keep last 1 second for continuity
        if final:
            connection["audio_buffer"] = np.array([], dtype=np.float32)
        else:
            keep_samples = sample_rate * 1  # 1 second
            connection["audio_buffer"] = audio_buffer[-keep_samples:] if len(audio_buffer) > keep_samples else audio_buffer
            
    except Exception as e:
        logger.error(f"Error processing audio for {client_id}: {e}")
        try:
            error_response = {
                "type": "error",
                "message": f"Processing error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            await connection["websocket"].send_text(json.dumps(error_response))
        except:
            pass

async def analyze_with_prosody(client_id, text, sample_rate=16000):
    """Analyze text with prosody features from recent audio"""
    if client_id not in connections:
        return
    
    connection = connections[client_id]
    audio_buffer = connection["audio_buffer"]
    
    if len(audio_buffer) < sample_rate * 0.5:  # Less than 0.5 seconds
        await send_error(client_id, "Not enough audio data for prosody analysis")
        return
    
    try:
        # Extract prosody features
        prosody_features = extract_prosody(audio_buffer, sample_rate)
        
        # Calculate basic metrics
        analysis = {
            "text": text,
            "word_count": len(text.split()),
            "character_count": len(text),
            "timestamp": datetime.now().isoformat(),
            "audio_duration": len(audio_buffer) / sample_rate,
            "sample_count": len(audio_buffer),
            "prosody_features": prosody_features
        }
        
        # Add simplified prosody metrics if features available
        if prosody_features and "error" not in prosody_features:
            # Look for common prosody features
            pitch_features = {k: v for k, v in prosody_features.items() if 'f0' in k.lower() or 'pitch' in k.lower()}
            energy_features = {k: v for k, v in prosody_features.items() if 'energy' in k.lower() or 'loudness' in k.lower()}
            
            analysis["prosody_summary"] = {
                "pitch_features_count": len(pitch_features),
                "energy_features_count": len(energy_features),
                "total_features": len(prosody_features),
                "has_prosody_data": len(prosody_features) > 0
            }
        
        response = {
            "type": "prosody_analysis",
            "analysis": analysis
        }
        
        await connection["websocket"].send_text(json.dumps(response))
        logger.info(f"Sent prosody analysis to {client_id} for text: {text[:30]}...")
        
    except Exception as e:
        logger.error(f"Error in prosody analysis for {client_id}: {e}")
        await send_error(client_id, f"Prosody analysis failed: {str(e)}")

async def send_error(client_id, error_message):
    """Send error message to client"""
    if client_id in connections:
        try:
            response = {
                "type": "error",
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            }
            await connections[client_id]["websocket"].send_text(json.dumps(response))
        except Exception as e:
            logger.error(f"Failed to send error to {client_id}: {e}")

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "whisper_loaded": whisper_model is not None,
        "opensmile_available": OPENSMILE_BINARY and os.path.exists(OPENSMILE_BINARY),
        "active_connections": len(connections),
        "audio_method": "float32_pcm",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)