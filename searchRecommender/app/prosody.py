from typing import Dict, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from scipy import stats
import json
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ProsodyFeatures:
    pitch_mean: float
    pitch_std: float
    intensity_mean: float
    intensity_std: float
    speaking_rate: float
    voice_breaks: int
    hnr: float  # Harmonics-to-noise ratio

@dataclass
class ProsodyResult:
    mood: Optional[str]
    confidence: float
    features: ProsodyFeatures
    raw_analysis: Dict

class ProsodyAnalyzer:
    """
    Analyzes audio prosodic features to infer speaker mood and characteristics.
    Uses pitch, intensity, speaking rate, and other acoustic features.
    """
    
    # Mood thresholds based on empirical studies
    MOOD_THRESHOLDS = {
        'happy': {
            'pitch_mean': (180, 400),  # Hz (higher pitch)
            'pitch_std': (30, 80),     # More variation
            'intensity': (65, 80),     # dB (louder)
            'speaking_rate': (4.5, 6.5) # syllables/sec (faster)
        },
        'sad': {
            'pitch_mean': (80, 180),   # Hz (lower pitch)
            'pitch_std': (10, 30),      # Less variation
            'intensity': (50, 65),      # dB (softer)
            'speaking_rate': (2.5, 4.0) # syllables/sec (slower)
        },
        'angry': {
            'pitch_mean': (150, 300),   # Hz (higher)
            'pitch_std': (50, 100),     # More variation
            'intensity': (70, 90),      # dB (louder)
            'speaking_rate': (4.0, 6.0) # syllables/sec (faster)
        },
        'neutral': {
            'pitch_mean': (100, 200),   # Hz
            'pitch_std': (20, 50),      # Moderate variation
            'intensity': (55, 70),      # dB
            'speaking_rate': (3.5, 5.0) # syllables/sec
        }
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize with optional config file path.
        Config can override default thresholds and weights.
        """
        self.config = self._load_config(config_path)
        if self.config:
            logger.info("Loaded custom prosody configuration")
            self.MOOD_THRESHOLDS = self.config.get('mood_thresholds', self.MOOD_THRESHOLDS)

    def _load_config(self, path: Optional[str]) -> Dict:
        """Load configuration from JSON file if provided"""
        if not path:
            return {}
        
        config_path = Path(path)
        try:
            if config_path.exists():
                with open(config_path) as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.warning(f"Failed to load config: {str(e)}")
            return {}

    def _extract_features(self, audio_data: Dict) -> ProsodyFeatures:
        """
        Extract prosodic features from audio data dictionary.
        Expected audio_data structure:
        {
            "pitch": [array of pitch values in Hz],
            "intensity": [array of intensity values in dB],
            "voiced_frames": [bool array of voiced frames],
            "duration_sec": float,
            "syllable_count": int,
            "hnr": float
        }
        """
        pitch = np.array(audio_data.get("pitch", []))
        intensity = np.array(audio_data.get("intensity", []))
        voiced = np.array(audio_data.get("voiced_frames", []))
        
        # Calculate basic statistics
        pitch_mean = np.mean(pitch[voiced]) if np.any(voiced) else 0
        pitch_std = np.std(pitch[voiced]) if np.any(voiced) else 0
        intensity_mean = np.mean(intensity) if len(intensity) > 0 else 0
        intensity_std = np.std(intensity) if len(intensity) > 0 else 0
        
        # Calculate speaking rate (syllables per second)
        duration = audio_data.get("duration_sec", 1)
        syllable_count = audio_data.get("syllable_count", 0)
        speaking_rate = syllable_count / duration if duration > 0 else 0
        
        # Count voice breaks (unvoiced segments between voiced segments)
        voice_breaks = 0
        if len(voiced) > 1:
            voice_breaks = np.sum(~voiced[1:] & voiced[:-1])
        
        return ProsodyFeatures(
            pitch_mean=float(pitch_mean),
            pitch_std=float(pitch_std),
            intensity_mean=float(intensity_mean),
            intensity_std=float(intensity_std),
            speaking_rate=float(speaking_rate),
            voice_breaks=int(voice_breaks),
            hnr=float(audio_data.get("hnr", 0))
        )

    def _classify_mood(self, features: ProsodyFeatures) -> Tuple[Optional[str], float]:
        """
        Classify mood based on extracted features.
        Returns (mood, confidence) tuple.
        """
        scores = {}
        
        for mood, thresholds in self.MOOD_THRESHOLDS.items():
            # Calculate how many features fall within expected ranges
            match_count = 0
            
            # Pitch mean check
            if thresholds['pitch_mean'][0] <= features.pitch_mean <= thresholds['pitch_mean'][1]:
                match_count += 1
            
            # Pitch std check
            if thresholds['pitch_std'][0] <= features.pitch_std <= thresholds['pitch_std'][1]:
                match_count += 1
                
            # Intensity check
            if thresholds['intensity'][0] <= features.intensity_mean <= thresholds['intensity'][1]:
                match_count += 1
                
            # Speaking rate check
            if thresholds['speaking_rate'][0] <= features.speaking_rate <= thresholds['speaking_rate'][1]:
                match_count += 1
            
            # Score is percentage of matching features
            scores[mood] = match_count / 4
        
        if not scores:
            return None, 0.0
        
        # Get mood with highest score
        best_mood = max(scores.items(), key=lambda x: x[1])
        
        # Only return mood if confidence > 0.5
        return (best_mood[0], best_mood[1]) if best_mood[1] > 0.5 else (None, 0.0)

    def analyze_prosody(self, audio_data: Dict) -> ProsodyResult:
        """
        Main analysis function that processes audio features.
        
        Args:
            audio_data: Dictionary containing audio features (see _extract_features)
            
        Returns:
            ProsodyResult with mood classification and features
        """
        if not audio_data:
            return ProsodyResult(None, 0.0, None, {})
            
        try:
            # Step 1: Extract features
            features = self._extract_features(audio_data)
            
            # Step 2: Classify mood
            mood, confidence = self._classify_mood(features)
            
            # Prepare raw analysis data
            raw_analysis = {
                "pitch_stats": {
                    "mean": features.pitch_mean,
                    "std": features.pitch_std,
                    "min": np.min(audio_data.get("pitch", [])),
                    "max": np.max(audio_data.get("pitch", []))
                },
                "intensity_stats": {
                    "mean": features.intensity_mean,
                    "std": features.intensity_std
                },
                "speaking_rate": features.speaking_rate,
                "voice_breaks": features.voice_breaks,
                "hnr": features.hnr
            }
            
            return ProsodyResult(
                mood=mood,
                confidence=confidence,
                features=features,
                raw_analysis=raw_analysis
            )
            
        except Exception as e:
            logger.error(f"Prosody analysis failed: {str(e)}")
            return ProsodyResult(None, 0.0, None, {})

if __name__ == "__main__":
    # Test cases with synthetic data
    analyzer = ProsodyAnalyzer()
    
    # Happy voice example
    happy_voice = {
        "pitch": np.random.normal(220, 30, 100).tolist(),
        "intensity": np.random.normal(70, 5, 100).tolist(),
        "voiced_frames": [True] * 100,
        "duration_sec": 5.0,
        "syllable_count": 30,
        "hnr": 25.0
    }
    
    # Sad voice example
    sad_voice = {
        "pitch": np.random.normal(120, 15, 100).tolist(),
        "intensity": np.random.normal(60, 3, 100).tolist(),
        "voiced_frames": [True] * 100,
        "duration_sec": 5.0,
        "syllable_count": 15,
        "hnr": 20.0
    }
    
    # Angry voice example
    angry_voice = {
        "pitch": np.random.normal(250, 60, 100).tolist(),
        "intensity": np.random.normal(80, 8, 100).tolist(),
        "voiced_frames": [True] * 100,
        "duration_sec": 5.0,
        "syllable_count": 28,
        "hnr": 18.0
    }
    
    print("=== Testing Prosody Analyzer ===")
    
    for name, data in [("Happy", happy_voice), ("Sad", sad_voice), ("Angry", angry_voice)]:
        print(f"\nAnalyzing {name} voice sample...")
        result = analyzer.analyze_prosody(data)
        
        print(f"Detected Mood: {result.mood} (confidence: {result.confidence:.2f})")
        print("Features:")
        print(f"- Pitch: {result.features.pitch_mean:.1f}±{result.features.pitch_std:.1f} Hz")
        print(f"- Intensity: {result.features.intensity_mean:.1f}±{result.features.intensity_std:.1f} dB")
        print(f"- Speaking Rate: {result.features.speaking_rate:.1f} syllables/sec")
        print(f"- Voice Breaks: {result.features.voice_breaks}")
        print(f"- HNR: {result.features.hnr:.1f} dB")