from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import logging
import tempfile
import subprocess
import os
import wave
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import uuid
from typing import Dict, Any, Optional

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Search API", version="1.0.0")

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

# Storage - using same pattern as working prosody code
connections = {}
voice_search_sessions = {}

# Load Whisper - same as working code
try:
    import whisper
    whisper_model = whisper.load_model("base")
    logger.info("Whisper loaded successfully for voice search")
except Exception as e:
    logger.error(f"Whisper failed to load: {e}")
    whisper_model = None

def create_wav_from_float32_array(float32_array, sample_rate=16000):
    """Create WAV file from float32 audio array - same as working code"""
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
    """Transcribe audio from float32 array - same as working code"""
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

def extract_simple_prosody(audio_data, sample_rate=16000):
    """Extract simple prosody features - based on working code"""
    try:
        # Basic audio features
        duration = len(audio_data) / sample_rate
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2
        
        # Simple pitch estimation using autocorrelation
        def estimate_pitch(signal, sr):
            signal = signal - np.mean(signal)
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            min_period = int(sr / 500)  # 500 Hz max
            max_period = int(sr / 50)   # 50 Hz min
            
            if len(autocorr) > max_period:
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                estimated_f0 = sr / peak_idx if peak_idx > 0 else 0
                return estimated_f0
            return 0
        
        estimated_pitch = estimate_pitch(audio_data, sample_rate)
        
        # Return prosody features
        return {
            "duration": float(duration),
            "rms_energy": float(rms_energy),
            "zero_crossing_rate": float(zero_crossing_rate),
            "estimated_pitch_hz": float(estimated_pitch),
            "energy_db": float(20 * np.log10(rms_energy + 1e-10)),
            "sample_count": len(audio_data),
            "sample_rate": sample_rate,
            "method": "simple_python_analysis",
            "intensity": "high" if rms_energy > 0.1 else "medium" if rms_energy > 0.05 else "low",
            "tempo": "fast" if zero_crossing_rate > 50 else "medium" if zero_crossing_rate > 20 else "slow",
            "pitch_level": "high" if estimated_pitch > 200 else "medium" if estimated_pitch > 100 else "low"
        }
        
    except Exception as e:
        logger.error(f"Error in prosody extraction: {e}")
        return {"error": f"Prosody analysis failed: {e}"}

def get_default_prosody():
    """Get default prosody features for text-only search"""
    return {
        "duration": 0.0,
        "rms_energy": 0.0,
        "zero_crossing_rate": 0.0,
        "estimated_pitch_hz": 0.0,
        "energy_db": 0.0,
        "sample_count": 0,
        "sample_rate": 0,
        "method": "text_only",
        "intensity": "none",
        "tempo": "none",
        "pitch_level": "none"
    }

def dummy_search_service(query: str, prosody_features: Dict[str, Any]) -> Dict[str, Any]:
    """Dummy search service that simulates search with prosody context"""
    
    # Sample search results
    sample_results = [
        {
            "title": f"Understanding {query} - Complete Guide",
            "url": f"https://example.com/guide/{query.replace(' ', '-').lower()}",
            "snippet": f"Comprehensive guide about {query}. Learn everything you need to know with examples and practical applications."
        },
        {
            "title": f"{query} - Latest News and Updates",
            "url": f"https://news.example.com/{len(query)}",
            "snippet": f"Stay updated with the latest news and developments related to {query}. Recent articles and expert opinions."
        },
        {
            "title": f"How to {query} - Step by Step Tutorial",
            "url": f"https://tutorial.com/{hash(query) % 1000}",
            "snippet": f"Learn {query} with our step-by-step tutorial. Beginner-friendly instructions with practical examples."
        },
        {
            "title": f"{query} FAQ - Frequently Asked Questions",
            "url": f"https://faq.com/{query[:5]}",
            "snippet": f"Find answers to common questions about {query}. Expert-verified responses and helpful tips."
        },
        {
            "title": f"Advanced {query} Techniques and Tips",
            "url": f"https://advanced.com/{len(query.split())}",
            "snippet": f"Master advanced techniques for {query}. Professional tips and best practices from experts."
        }
    ]
    
    # Adjust results based on prosody
    intensity = prosody_features.get("intensity", "none")
    tempo = prosody_features.get("tempo", "none")
    method = prosody_features.get("method", "text_only")
    
    # Determine number of results and context based on prosody
    if intensity == "high" or tempo == "fast":
        # High intensity/fast speech - user wants quick answers
        results = sample_results[:3]
        search_context = "urgent_query"
    elif intensity == "low" or tempo == "slow":
        # Low intensity/slow speech - user wants detailed information
        results = sample_results[:5]
        search_context = "detailed_query"
    else:
        # Normal search
        results = sample_results[:4]
        search_context = "standard_query"
    
    return {
        "query": query,
        "results": results,
        "total_results": len(results),
        "search_context": search_context,
        "prosody_influenced": method != "text_only",
        "search_metadata": {
            "timestamp": datetime.now().isoformat(),
            "prosody_summary": {
                "method": method,
                "intensity": intensity,
                "tempo": tempo,
                "pitch": prosody_features.get("pitch_level", "none"),
                "duration": prosody_features.get("duration", 0.0),
                "energy": prosody_features.get("rms_energy", 0.0)
            }
        }
    }

@app.websocket("/ws/voice-search/{client_id}")
async def voice_search_websocket(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connections[client_id] = {
        "websocket": websocket,
        "audio_buffer": np.array([], dtype=np.float32),
        "connected_at": datetime.now(),
        "recording": False
    }
    logger.info(f"Voice search client {client_id} connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "start_voice_recording":
                # Start recording audio for voice search
                connections[client_id]["recording"] = True
                connections[client_id]["audio_buffer"] = np.array([], dtype=np.float32)
                
                await websocket.send_text(json.dumps({
                    "type": "recording_started",
                    "status": "recording"
                }))
                logger.info(f"Started voice recording for {client_id}")
                
            elif message["type"] == "audio_data":
                # Receive audio data - same pattern as working code
                if connections[client_id]["recording"]:
                    audio_array = np.array(message["data"], dtype=np.float32)
                    sample_rate = message.get("sampleRate", 16000)
                    
                    # Append to buffer
                    connections[client_id]["audio_buffer"] = np.concatenate([
                        connections[client_id]["audio_buffer"], 
                        audio_array
                    ])
                    
                    logger.debug(f"Received {len(audio_array)} audio samples for {client_id}")
                
            elif message["type"] == "stop_voice_recording":
                # Stop recording and process
                connections[client_id]["recording"] = False
                await process_voice_search(client_id)
                
            elif message["type"] == "text_search":
                # Handle text search
                query = message.get("query", "").strip()
                if query:
                    await process_text_search(client_id, query)
    
    except WebSocketDisconnect:
        logger.info(f"Voice search client {client_id} disconnected")
    except Exception as e:
        logger.error(f"Voice search WebSocket error for {client_id}: {e}")
    finally:
        if client_id in connections:
            del connections[client_id]
        logger.info(f"Voice search client {client_id} cleaned up")

async def process_voice_search(client_id: str):
    """Process voice search - same pattern as working prosody code"""
    if client_id not in connections:
        return
    
    connection = connections[client_id]
    audio_buffer = connection["audio_buffer"]
    
    if len(audio_buffer) < 8000:  # Less than 0.5 seconds at 16kHz
        await send_error(client_id, "Audio too short for voice search")
        return
    
    try:
        logger.info(f"Processing voice search for {client_id}: {len(audio_buffer)} samples ({len(audio_buffer)/16000:.1f}s)")
        
        # Create session ID
        session_id = f"voice_search_{client_id}_{int(datetime.now().timestamp() * 1000)}"
        
        # Transcribe audio
        text = transcribe_audio(audio_buffer, 16000)
        
        if text and not text.startswith("Audio") and not text.startswith("Transcription failed") and not text.startswith("No speech"):
            # Extract prosody
            prosody_features = extract_simple_prosody(audio_buffer, 16000)
            
            # Store session
            voice_search_sessions[session_id] = {
                "client_id": client_id,
                "transcription": text,
                "prosody": prosody_features,
                "timestamp": datetime.now().isoformat(),
                "audio_duration": len(audio_buffer) / 16000
            }
            
            # Send transcription back
            response = {
                "type": "voice_transcription",
                "session_id": session_id,
                "text": text,
                "confidence": 0.9,  # Dummy confidence
                "prosody_available": "error" not in prosody_features,
                "audio_duration": len(audio_buffer) / 16000
            }
            
            await connection["websocket"].send_text(json.dumps(response))
            logger.info(f"Sent voice transcription to {client_id}: {text[:50]}...")
            
        else:
            await send_error(client_id, f"Voice transcription failed: {text}")
            
        # Clear buffer
        connection["audio_buffer"] = np.array([], dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Error processing voice search for {client_id}: {e}")
        await send_error(client_id, f"Voice processing failed: {str(e)}")

async def process_text_search(client_id: str, query: str):
    """Process text search"""
    if client_id not in connections:
        return
    
    try:
        logger.info(f"Processing text search for {client_id}: {query}")
        
        # Use default prosody for text search
        prosody_features = get_default_prosody()
        
        # Call search service
        search_results = dummy_search_service(query, prosody_features)
        
        # Send results back
        await connections[client_id]["websocket"].send_text(json.dumps({
            "type": "search_results",
            "results": search_results
        }))
        
        logger.info(f"Sent text search results to {client_id}: {search_results['total_results']} results")
        
    except Exception as e:
        logger.error(f"Error processing text search: {e}")
        await send_error(client_id, f"Text search failed: {str(e)}")

async def send_error(client_id: str, error_message: str):
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

@app.post("/api/search")
async def search_api(request: Dict[str, Any]):
    """HTTP API for search"""
    try:
        query = request.get("query", "").strip()
        session_id = request.get("session_id")
        search_type = request.get("type", "text")
        
        if not query:
            return {"error": "Query is required"}
        
        logger.info(f"API search request: {query} (type: {search_type}, session: {session_id})")
        
        # Get prosody features
        prosody_features = get_default_prosody()
        
        if search_type == "voice" and session_id and session_id in voice_search_sessions:
            # Use voice session prosody
            session = voice_search_sessions[session_id]
            if session.get("prosody"):
                prosody_features = session["prosody"]
        
        # Call search service
        search_results = dummy_search_service(query, prosody_features)
        
        logger.info(f"Search completed: {search_results['total_results']} results")
        return search_results
        
    except Exception as e:
        logger.error(f"Search API error: {e}")
        return {"error": f"Search failed: {str(e)}"}

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "service": "voice_search",
        "whisper_loaded": whisper_model is not None,
        "opensmile_available": OPENSMILE_BINARY and os.path.exists(OPENSMILE_BINARY),
        "active_connections": len(connections),
        "active_sessions": len(voice_search_sessions),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/debug/sessions")
async def debug_sessions():
    """Debug endpoint for voice search sessions"""
    session_info = {}
    for session_id, session_data in voice_search_sessions.items():
        session_info[session_id] = {
            "transcription": session_data.get("transcription", ""),
            "audio_duration": session_data.get("audio_duration", 0),
            "timestamp": session_data.get("timestamp", ""),
            "client_id": session_data.get("client_id", ""),
            "prosody_method": session_data.get("prosody", {}).get("method", "unknown")
        }
    return {
        "total_sessions": len(voice_search_sessions),
        "sessions": session_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)