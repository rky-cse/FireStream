import os
import wave
import tempfile
import subprocess
import logging
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENSMILE_BINARY = os.getenv("OPENSMILE_BINARY")
OPENSMILE_CONFIG = os.getenv("OPENSMILE_CONFIG")

# Setup logging
logger = logging.getLogger("voiceProsodyExtractor")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(ch)


def create_wav_from_float32_array(float32_array: np.ndarray, sample_rate: int = 16000) -> str:
    """Create WAV file from float32 audio array."""
    try:
        # Convert float32 to int16
        int16_array = (float32_array * 32767).astype(np.int16)

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_file:
            wav_path = wav_file.name

        # Write WAV
        with wave.open(wav_path, 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(int16_array.tobytes())

        logger.debug(f"Created WAV file: {wav_path}, samples: {len(int16_array)}")
        return wav_path
    except Exception as e:
        logger.error(f"Error creating WAV file: {e}")
        return None


def transcribe_audio(audio_data: np.ndarray, whisper_model, sample_rate: int = 16000) -> str:
    """Transcribe audio from float32 array using Whisper."""
    if not whisper_model:
        return "Whisper not available"
    if len(audio_data) < 0.1 * sample_rate:
        return "Audio too short"

    wav_path = create_wav_from_float32_array(audio_data, sample_rate)
    if not wav_path:
        return "Failed to create audio file"

    try:
        result = whisper_model.transcribe(wav_path)
        text = result.get("text", "").strip()
        return text if text else "No speech detected"
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return f"Transcription failed: {e}"
    finally:
        try:
            os.unlink(wav_path)
        except:
            pass


def debug_csv_file(file_path: str) -> str:
    """Debug CSV file content."""
    try:
        size = os.path.getsize(file_path)
        logger.debug(f"CSV file size: {size} bytes")
        if size == 0:
            return "File is empty"

        with open(file_path, 'rb') as f:
            raw = f.read(500)

        hex_content = raw.hex()
        try:
            text_content = raw.decode('utf-8', errors='replace')
            return f"File readable, starts with: {text_content[:100]!r}"
        except:
            return f"Binary file, hex: {hex_content[:50]}"
    except Exception as e:
        return f"Debug error: {e}"


def extract_prosody_simple(audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
    """Simple prosody extraction with basic audio features."""
    try:
        duration = len(audio_data) / sample_rate
        rms_energy = float(np.sqrt(np.mean(audio_data ** 2)))
        zero_crossing_rate = float(np.mean(np.abs(np.diff(np.sign(audio_data)))) / 2)

        def estimate_pitch(signal):
            signal = signal - np.mean(signal)
            autocorr = np.correlate(signal, signal, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            min_period = int(sample_rate / 500)
            max_period = int(sample_rate / 50)
            if len(autocorr) > max_period:
                peak = np.argmax(autocorr[min_period:max_period]) + min_period
                return float(sample_rate / peak) if peak > 0 else 0.0
            return 0.0

        estimated_pitch = estimate_pitch(audio_data)
        fft = np.fft.fft(audio_data)
        mag = np.abs(fft[:len(fft)//2])
        freqs = np.fft.fftfreq(len(audio_data), 1/sample_rate)[:len(mag)]
        spectral_centroid = float(np.sum(freqs * mag) / (np.sum(mag) + 1e-12))

        return {
            "method": "simple_python_analysis",
            "duration": duration,
            "rms_energy": rms_energy,
            "zero_crossing_rate": zero_crossing_rate,
            "estimated_pitch_hz": estimated_pitch,
            "spectral_centroid_hz": spectral_centroid,
            "energy_db": float(20 * np.log10(rms_energy + 1e-10)),
            "sample_count": len(audio_data),
            "sample_rate": sample_rate
        }
    except Exception as e:
        logger.error(f"Error in simple prosody extraction: {e}")
        return {"error": str(e)}


def extract_prosody(audio_data: np.ndarray, sample_rate: int = 16000) -> dict:
    """Extract prosody features via OpenSMILE, falling back to simple if needed."""
    # If OpenSMILE not configured, fallback
    if not OPENSMILE_BINARY or not os.path.exists(OPENSMILE_BINARY):
        logger.warning("OpenSMILE binary not available, using simple analysis")
        return extract_prosody_simple(audio_data, sample_rate)
    if not OPENSMILE_CONFIG or not os.path.exists(OPENSMILE_CONFIG):
        logger.warning(f"OpenSMILE config not found: {OPENSMILE_CONFIG}, using simple analysis")
        return extract_prosody_simple(audio_data, sample_rate)

    wav_path = create_wav_from_float32_array(audio_data, sample_rate)
    if not wav_path:
        return extract_prosody_simple(audio_data, sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as out:
        csv_path = out.name

    commands = [
        [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-O", csv_path, "-l", "1"],
        [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-O", csv_path],
        [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-csvoutput", csv_path],
        [OPENSMILE_BINARY, "-C", OPENSMILE_CONFIG, "-I", wav_path, "-O", csv_path, "-outputformat", "csv"],
    ]

    last_error = ""
    for cmd in commands:
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=30, text=True)
            if proc.returncode == 0 and os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
                debug_info = debug_csv_file(csv_path)
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().splitlines()
                if len(lines) >= 2:
                    hdr = lines[0].replace('"', '').split(',')
                    vals = lines[1].replace('"', '').split(',')
                    features = {}
                    for i in range(1, min(len(hdr), len(vals))):
                        try:
                            if vals[i] and vals[i] != '?':
                                features[hdr[i]] = float(vals[i])
                        except:
                            continue
                    if features:
                        features["method"] = "opensmile"
                        # cleanup
                        os.unlink(wav_path)
                        os.unlink(csv_path)
                        logger.info(f"OpenSMILE extracted {len(features)} features")
                        return features
            else:
                last_error = proc.stderr or proc.stdout
        except Exception as e:
            last_error = str(e)

    # cleanup
    for p in (wav_path, csv_path):
        try: os.unlink(p)
        except: pass

    logger.info("OpenSMILE failed, falling back to simple analysis")
    simple = extract_prosody_simple(audio_data, sample_rate)
    simple["opensmile_error"] = last_error
    simple["opensmile_attempted"] = True
    return simple


def try_different_opensmile_configs() -> list:
    """Return list of OpenSMILE configs that exist and whose binary responds."""
    configs = []
    if not OPENSMILE_BINARY or not os.path.exists(OPENSMILE_BINARY):
        return configs

    common = [
        OPENSMILE_CONFIG,
        "prosodyShs.conf",
        "emobase.conf",
        "gemaps.conf",
        "egemaps.conf",
        "IS09_emotion.conf",
        "IS10_paraling.conf",
    ]
    for cfg in common:
        if not cfg:
            continue
        try:
            res = subprocess.run([OPENSMILE_BINARY, "-h"], capture_output=True, timeout=5, text=True)
            if res.returncode == 0:
                # look for config file
                paths = [cfg, f"/usr/share/opensmile/config/{cfg}", f"/usr/local/share/opensmile/config/{cfg}", f"./config/{cfg}"]
                for p in paths:
                    if os.path.exists(p):
                        configs.append(p)
                        break
        except:
            continue

    return configs
