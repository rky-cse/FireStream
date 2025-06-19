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
import csv
import io

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

# Connection and session storage
connections = {}
audio_sessions = {}

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

def debug_csv_file(file_path):
    """Debug CSV file content"""
    try:
        file_size = os.path.getsize(file_path)
        logger.debug(f"CSV file size: {file_size} bytes")
        
        if file_size == 0:
            return "File is empty"
        
        # Read first 500 bytes as hex and text
        with open(file_path, 'rb') as f:
            raw_content = f.read(500)
        
        hex_content = raw_content.hex()
        logger.debug(f"First 100 hex bytes: {hex_content[:200]}")
        
        # Try to decode as text
        try:
            text_content = raw_content.decode('utf-8', errors='replace')
            logger.debug(f"First 200 chars as text: {repr(text_content[:200])}")
            return f"File readable, starts with: {text_content[:100]}"
        except:
            return f"Binary file, hex: {hex_content[:50]}"
            
    except Exception as e:
        return f"Debug error: {e}"

def try_different_opensmile_configs():
    """Try to find working OpenSMILE configurations"""
    if not OPENSMILE_BINARY or not os.path.exists(OPENSMILE_BINARY):
        return []
    
    # Common OpenSMILE config files
    possible_configs = [
        OPENSMILE_CONFIG,  # User specified
        "prosodyShs.conf",
        "emobase.conf", 
        "gemaps.conf",
        "egemaps.conf",
        "IS09_emotion.conf",
        "IS10_paraling.conf"
    ]
    
    working_configs = []
    
    for config in possible_configs:
        if not config:
            continue
            
        try:
            # Test with a simple command
            result = subprocess.run([
                OPENSMILE_BINARY, 
                "-h"  # Help command
            ], capture_output=True, timeout=5, text=True)
            
            if result.returncode == 0:
                working_configs.append(config)
                logger.debug(f"OpenSMILE binary works, testing config: {config}")
                
                # Try to find config file
                if os.path.exists(config):
                    working_configs.append(config)
                    logger.debug(f"Found config file: {config}")
                elif config.startswith("/"):
                    continue  # Absolute path that doesn't exist
                else:
                    # Try to find in common locations
                    common_paths = [
                        f"/usr/share/opensmile/config/{config}",
                        f"/usr/local/share/opensmile/config/{config}",
                        f"./config/{config}",
                        config
                    ]
                    
                    for path in common_paths:
                        if os.path.exists(path):
                            working_configs.append(path)
                            logger.debug(f"Found config at: {path}")
                            break
                break
        except Exception as e:
            logger.debug(f"Error testing OpenSMILE: {e}")
            continue
    
    return working_configs

def extract_prosody_simple(audio_data, sample_rate=16000):
    """Simple prosody extraction with basic audio features"""
    try:
        # Calculate basic audio statistics as fallback
        duration = len(audio_data) / sample_rate
        
        # Basic audio features
        rms_energy = np.sqrt(np.mean(audio_data ** 2))
        zero_crossing_rate = np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2
        
        # Simple pitch estimation using autocorrelation
        def estimate_pitch(signal, sr):
            # Autocorrelation-based pitch estimation
            signal = signal - np.mean(signal)
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find peaks
            min_period = int(sr / 500)  # 500 Hz max
            max_period = int(sr / 50)   # 50 Hz min
            
            if len(autocorr) > max_period:
                peak_idx = np.argmax(autocorr[min_period:max_period]) + min_period
                estimated_f0 = sr / peak_idx if peak_idx > 0 else 0
                return estimated_f0
            return 0
        
        estimated_pitch = estimate_pitch(audio_data, sample_rate)
        
        # Spectral features
        fft = np.fft.fft(audio_data)
        magnitude = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(fft)//2]
        
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
        
        # Return basic features
        return {
            "duration": duration,
            "rms_energy": float(rms_energy),
            "zero_crossing_rate": float(zero_crossing_rate),
            "estimated_pitch_hz": float(estimated_pitch),
            "spectral_centroid_hz": float(spectral_centroid),
            "energy_db": float(20 * np.log10(rms_energy + 1e-10)),
            "sample_count": len(audio_data),
            "sample_rate": sample_rate,
            "method": "simple_python_analysis"
        }
        
    except Exception as e:
        logger.error(f"Error in simple prosody extraction: {e}")
        return {"error": f"Simple analysis failed: {e}"}

def extract_prosody(audio_data, sample_rate=16000):
    """Extract prosody features from float32 audio array"""
    if not OPENSMILE_BINARY or not os.path.exists(OPENSMILE_BINARY):
        logger.warning("OpenSMILE not available, using simple analysis")
        return extract_prosody_simple(audio_data, sample_rate)
    
    if not OPENSMILE_CONFIG or not os.path.exists(OPENSMILE_CONFIG):
        logger.warning(f"OpenSMILE config not found: {OPENSMILE_CONFIG}, using simple analysis")
        return extract_prosody_simple(audio_data, sample_rate)
    
    try:
        # Create WAV file from float32 array
        wav_path = create_wav_from_float32_array(audio_data, sample_rate)
        if not wav_path:
            return extract_prosody_simple(audio_data, sample_rate)
        
        # Create output file for OpenSMILE
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as output_file:
            output_path = output_file.name
        
        # Try OpenSMILE with different approaches
        commands_to_try = [
            # Standard command
            [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-O", output_path, "-l", "1"],
            # Without log level
            [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-O", output_path],
            # With different output format
            [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-csvoutput", output_path],
            # With explicit format
            [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-O", output_path, "-outputformat", "csv"]
        ]
        
        opensmile_success = False
        last_error = ""
        
        for i, cmd in enumerate(commands_to_try):
            try:
                logger.debug(f"Trying OpenSMILE command {i+1}: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, 
                    timeout=30,
                    text=True
                )
                
                logger.debug(f"OpenSMILE return code: {result.returncode}")
                
                if result.stdout:
                    logger.debug(f"OpenSMILE stdout: {result.stdout[:200]}")
                if result.stderr:
                    logger.debug(f"OpenSMILE stderr: {result.stderr[:200]}")
                
                if result.returncode == 0:
                    # Check if output file was created
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        logger.debug(f"OpenSMILE output file size: {file_size} bytes")
                        
                        if file_size > 0:
                            # Debug the CSV content
                            debug_info = debug_csv_file(output_path)
                            logger.debug(f"CSV debug info: {debug_info}")
                            
                            # Try to read and parse
                            try:
                                with open(output_path, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    
                                if content.strip():
                                    lines = content.strip().split('\n')
                                    logger.debug(f"CSV has {len(lines)} lines")
                                    
                                    if len(lines) >= 2:
                                        # Try to parse
                                        features = {}
                                        header_line = lines[0]
                                        data_line = lines[1]
                                        
                                        # Simple CSV parsing
                                        headers = header_line.split(',') if ',' in header_line else header_line.split(';')
                                        values = data_line.split(',') if ',' in data_line else data_line.split(';')
                                        
                                        logger.debug(f"Headers: {len(headers)}, Values: {len(values)}")
                                        
                                        for j in range(1, min(len(headers), len(values))):
                                            try:
                                                header = headers[j].strip().strip('"')
                                                value_str = values[j].strip().strip('"')
                                                
                                                if value_str and value_str != '?':
                                                    features[header] = float(value_str)
                                            except (ValueError, IndexError):
                                                continue
                                        
                                        if features:
                                            logger.info(f"OpenSMILE extracted {len(features)} features successfully")
                                            features["method"] = "opensmile"
                                            opensmile_success = True
                                            
                                            # Clean up
                                            os.unlink(wav_path)
                                            os.unlink(output_path)
                                            
                                            return features
                                        else:
                                            logger.warning("No features extracted from CSV")
                                    else:
                                        logger.warning("CSV has insufficient lines")
                                else:
                                    logger.warning("CSV file is empty")
                                    
                            except Exception as parse_error:
                                logger.warning(f"Error parsing CSV: {parse_error}")
                        else:
                            logger.warning("OpenSMILE output file is empty")
                    else:
                        logger.warning("OpenSMILE output file was not created")
                else:
                    last_error = result.stderr
                    logger.warning(f"OpenSMILE failed with code {result.returncode}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.warning("OpenSMILE timed out")
                last_error = "Process timed out"
            except Exception as e:
                logger.warning(f"Error running OpenSMILE: {e}")
                last_error = str(e)
        
        # Clean up files
        try:
            os.unlink(wav_path)
            if os.path.exists(output_path):
                os.unlink(output_path)
        except:
            pass
        
        # If OpenSMILE failed, fall back to simple analysis
        logger.info("OpenSMILE failed, falling back to simple Python analysis")
        simple_features = extract_prosody_simple(audio_data, sample_rate)
        simple_features["opensmile_error"] = last_error
        simple_features["opensmile_attempted"] = True
        return simple_features
        
    except Exception as e:
        logger.error(f"Prosody extraction error: {e}")
        # Fall back to simple analysis
        simple_features = extract_prosody_simple(audio_data, sample_rate)
        simple_features["extraction_error"] = str(e)
        return simple_features

# [Rest of the WebSocket and other handlers remain the same...]
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
                audio_array = np.array(message["data"], dtype=np.float32)
                sample_rate = message.get("sampleRate", 16000)
                
                connections[client_id]["audio_buffer"] = np.concatenate([
                    connections[client_id]["audio_buffer"], 
                    audio_array
                ])
                
                buffer_size = len(connections[client_id]["audio_buffer"])
                if buffer_size >= sample_rate * 3:
                    await process_audio_buffer(client_id, sample_rate)
            
            elif message["type"] == "audio_end":
                sample_rate = message.get("sampleRate", 16000)
                await process_audio_buffer(client_id, sample_rate, final=True)
                
            elif message["type"] == "analyze_prosody":
                text = message.get("text", "")
                session_id = message.get("session_id", "")
                sample_rate = message.get("sampleRate", 16000)
                await analyze_with_prosody_by_session(client_id, text, session_id, sample_rate)
    
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
    
    if len(audio_buffer) < sample_rate * 0.5:
        return
    
    try:
        logger.info(f"Processing {len(audio_buffer)} audio samples for {client_id} (duration: {len(audio_buffer)/sample_rate:.1f}s)")
        
        session_id = f"{client_id}_{int(datetime.now().timestamp() * 1000)}"
        
        audio_sessions[session_id] = {
            "audio_data": audio_buffer.copy(),
            "sample_rate": sample_rate,
            "timestamp": datetime.now().isoformat(),
            "client_id": client_id
        }
        
        text = transcribe_audio(audio_buffer, sample_rate)
        
        if text and not text.startswith("Audio") and not text.startswith("Transcription failed"):
            response = {
                "type": "transcription",
                "text": text,
                "timestamp": datetime.now().isoformat(),
                "sample_count": len(audio_buffer),
                "duration": len(audio_buffer) / sample_rate,
                "session_id": session_id
            }
            
            await connection["websocket"].send_text(json.dumps(response))
            logger.info(f"Sent transcription to {client_id}: {text[:50]}... (session: {session_id})")
        else:
            logger.warning(f"Transcription failed for {client_id}: {text}")
            if not text.startswith("No speech detected"):
                error_response = {
                    "type": "error",
                    "message": text,
                    "timestamp": datetime.now().isoformat()
                }
                await connection["websocket"].send_text(json.dumps(error_response))
        
        if final:
            connection["audio_buffer"] = np.array([], dtype=np.float32)
        else:
            keep_samples = sample_rate * 1
            connection["audio_buffer"] = audio_buffer[-keep_samples:] if len(audio_buffer) > keep_samples else audio_buffer
        
        if len(audio_sessions) > 15:
            oldest_sessions = sorted(audio_sessions.keys())[:len(audio_sessions)-15]
            for old_session in oldest_sessions:
                del audio_sessions[old_session]
            
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

async def analyze_with_prosody_by_session(client_id, text, session_id, sample_rate=16000):
    """Analyze text with prosody features using stored session audio"""
    if client_id not in connections:
        return
    
    audio_data = None
    actual_sample_rate = sample_rate
    
    if session_id and session_id in audio_sessions:
        session_data = audio_sessions[session_id]
        audio_data = session_data["audio_data"]
        actual_sample_rate = session_data["sample_rate"]
        logger.info(f"Using stored audio session {session_id} for prosody analysis")
    else:
        connection = connections[client_id]
        audio_data = connection["audio_buffer"]
        logger.info(f"Session {session_id} not found, using current buffer for prosody analysis")
    
    if audio_data is None or len(audio_data) < actual_sample_rate * 0.5:
        available_sessions = list(audio_sessions.keys())[-5:]
        await send_error(client_id, f"Not enough audio data for prosody analysis (session: {session_id}, available: {available_sessions})")
        return
    
    try:
        logger.info(f"Analyzing prosody for session {session_id}: {len(audio_data)} samples ({len(audio_data)/actual_sample_rate:.1f}s)")
        
        prosody_features = extract_prosody(audio_data, actual_sample_rate)
        
        analysis = {
            "text": text,
            "word_count": len(text.split()),
            "character_count": len(text),
            "timestamp": datetime.now().isoformat(),
            "audio_duration": len(audio_data) / actual_sample_rate,
            "sample_count": len(audio_data),
            "session_id": session_id,
            "prosody_features": prosody_features
        }
        
        # Enhanced prosody summary
        if prosody_features and "error" not in prosody_features and len(prosody_features) > 0:
            # Categorize features by type
            pitch_features = {k: v for k, v in prosody_features.items() if any(term in k.lower() for term in ['f0', 'pitch', 'voicing'])}
            energy_features = {k: v for k, v in prosody_features.items() if any(term in k.lower() for term in ['energy', 'loudness', 'rms', 'power'])}
            spectral_features = {k: v for k, v in prosody_features.items() if any(term in k.lower() for term in ['mfcc', 'spectral', 'formant', 'centroid'])}
            temporal_features = {k: v for k, v in prosody_features.items() if any(term in k.lower() for term in ['duration', 'rate', 'zero_crossing'])}
            
            analysis["prosody_summary"] = {
                "pitch_features_count": len(pitch_features),
                "energy_features_count": len(energy_features),
                "spectral_features_count": len(spectral_features),
                "temporal_features_count": len(temporal_features),
                "total_features": len(prosody_features),
                "has_prosody_data": True,
                "feature_extraction_success": True,
                "extraction_method": prosody_features.get("method", "unknown")
            }
            
            # Extract key meaningful features
            key_features = {}
            for feature_name, value in prosody_features.items():
                if isinstance(value, (int, float)):
                    # Key features we want to highlight
                    if any(term in feature_name.lower() for term in [
                        'pitch', 'f0', 'energy', 'rms', 'spectral_centroid', 
                        'duration', 'zero_crossing', 'estimated'
                    ]):
                        clean_name = feature_name.replace('_', ' ').title()
                        key_features[clean_name] = round(value, 3)
                        
                if len(key_features) >= 10:
                    break
            
            analysis["key_features"] = key_features
            
        else:
            analysis["prosody_summary"] = {
                "pitch_features_count": 0,
                "energy_features_count": 0,
                "spectral_features_count": 0,
                "temporal_features_count": 0,
                "total_features": 0,
                "has_prosody_data": False,
                "feature_extraction_success": False,
                "error_info": prosody_features.get("error", "Unknown error") if isinstance(prosody_features, dict) else "No features extracted",
                "extraction_method": "failed"
            }
            analysis["key_features"] = {}
        
        response = {
            "type": "prosody_analysis",
            "analysis": analysis
        }
        
        await connections[client_id]["websocket"].send_text(json.dumps(response))
        logger.info(f"Sent prosody analysis to {client_id} for session {session_id}: {text[:30]}...")
        
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
        "opensmile_config": OPENSMILE_CONFIG,
        "active_connections": len(connections),
        "stored_sessions": len(audio_sessions),
        "audio_method": "float32_pcm_with_fallback",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/opensmile")
async def debug_opensmile():
    """Debug OpenSMILE configuration and availability"""
    debug_info = {
        "opensmile_binary": OPENSMILE_BINARY,
        "opensmile_config": OPENSMILE_CONFIG,
        "binary_exists": os.path.exists(OPENSMILE_BINARY) if OPENSMILE_BINARY else False,
        "config_exists": os.path.exists(OPENSMILE_CONFIG) if OPENSMILE_CONFIG else False,
    }
    
    # Test OpenSMILE binary
    if debug_info["binary_exists"]:
        try:
            result = subprocess.run([OPENSMILE_BINARY, "-h"], 
                                  capture_output=True, timeout=5, text=True)
            debug_info["binary_works"] = result.returncode == 0
            debug_info["binary_version"] = result.stdout[:200] if result.stdout else "No output"
        except Exception as e:
            debug_info["binary_works"] = False
            debug_info["binary_error"] = str(e)
    
    # Try simple prosody extraction
    try:
        test_audio = np.array([0.1, -0.1, 0.2, -0.2] * 4000, dtype=np.float32)  # 1 second at 16kHz
        result = extract_prosody(test_audio, 16000)
        debug_info["prosody_test"] = {
            "success": "error" not in result,
            "feature_count": len(result) if isinstance(result, dict) else 0,
            "method": result.get("method", "unknown") if isinstance(result, dict) else "unknown",
            "sample_features": dict(list(result.items())[:3]) if isinstance(result, dict) else {}
        }
    except Exception as e:
        debug_info["prosody_test"] = {"error": str(e)}
    
    return debug_info

@app.get("/debug/sessions")
async def debug_sessions():
    """Debug endpoint to see stored audio sessions"""
    session_info = {}
    for session_id, session_data in audio_sessions.items():
        session_info[session_id] = {
            "sample_count": len(session_data["audio_data"]),
            "duration": len(session_data["audio_data"]) / session_data["sample_rate"],
            "timestamp": session_data["timestamp"],
            "client_id": session_data["client_id"]
        }
    return {
        "total_sessions": len(audio_sessions),
        "sessions": session_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)