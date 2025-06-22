import logging
import numpy as np
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ProsodyAnalyzer:
    """
    Advanced prosody analysis for voice-based mood detection and recommendation enhancement.
    
    Features:
    - Voice characteristic analysis (intensity, tempo, pitch, energy)
    - Emotion detection from prosodic features
    - Mood mapping for content recommendations
    - Voice quality assessment
    - Multi-method prosody analysis support
    - Performance optimization with caching
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_expiry = 300  # 5 minutes for prosody analysis
        
        # Prosody analysis statistics
        self.stats = {
            "analyses_performed": 0,
            "cache_hits": 0,
            "mood_detections": 0,
            "high_confidence_detections": 0
        }
        
        # Prosody to mood mapping based on voice characteristics
        self.prosody_mood_mapping = {
            "high_intensity_fast_tempo": {"mood": "excited", "confidence": 0.85, "boost_genres": ["Action", "Adventure", "Comedy"]},
            "high_intensity_slow_tempo": {"mood": "angry", "confidence": 0.80, "boost_genres": ["Drama", "Thriller", "Action"]},
            "low_intensity_fast_tempo": {"mood": "anxious", "confidence": 0.75, "boost_genres": ["Comedy", "Feel-good", "Romance"]},
            "low_intensity_slow_tempo": {"mood": "sad", "confidence": 0.90, "boost_genres": ["Comedy", "Feel-good", "Uplifting"]},
            "medium_intensity_medium_tempo": {"mood": "calm", "confidence": 0.70, "boost_genres": ["Drama", "Documentary", "Romance"]},
            "variable_intensity": {"mood": "stressed", "confidence": 0.65, "boost_genres": ["Comedy", "Feel-good", "Relaxing"]}
        }
        
        # Voice quality thresholds
        self.quality_thresholds = {
            "min_duration": 1.0,        # Minimum 1 second
            "min_energy": 0.01,         # Minimum RMS energy
            "max_duration": 30.0,       # Maximum 30 seconds
            "confidence_threshold": 0.6  # Minimum confidence for mood detection
        }
        
        logger.info("ProsodyAnalyzer initialized")
        logger.info(f"📅 Initialized at: 2025-06-20 08:38:13 UTC by user: rky-cse")
    
    def analyze_mood_from_prosody(self, prosody_features):
        """
        Analyze mood from prosodic features with confidence scoring.
        
        Args:
            prosody_features: Dictionary containing prosodic analysis results from ingestion/voice
            
        Returns:
            Dictionary with detected mood, confidence, and recommendations
        """
        self.stats["analyses_performed"] += 1
        
        try:
            if not prosody_features:
                logger.warning("No prosody features provided for analysis")
                return {"detected_mood": None, "confidence": 0.0, "error": "No prosody features"}
            
            # Check cache first
            cache_key = self._generate_prosody_cache_key(prosody_features)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self.stats["cache_hits"] += 1
                return cached_result
            
            logger.debug("🎤 Analyzing mood from voice prosody features")
            
            # Validate prosody features quality
            quality_check = self._assess_prosody_quality(prosody_features)
            if not quality_check["is_valid"]:
                logger.warning(f"Poor prosody quality: {quality_check['issues']}")
                return {"detected_mood": None, "confidence": 0.0, "quality_issues": quality_check["issues"]}
            
            # Extract and normalize prosodic characteristics
            characteristics = self._extract_prosodic_characteristics(prosody_features)
            
            # Detect mood using multiple methods
            mood_analysis = self._detect_mood_multi_method(characteristics, prosody_features)
            
            # Enhance with contextual analysis
            enhanced_analysis = self._enhance_with_context(mood_analysis, prosody_features)
            
            # Generate content recommendations based on detected mood
            recommendations = self._generate_prosody_recommendations(enhanced_analysis)
            
            # Compile final result
            result = {
                "detected_mood": enhanced_analysis.get("primary_mood"),
                "confidence": enhanced_analysis.get("confidence", 0.0),
                "alternative_moods": enhanced_analysis.get("alternative_moods", []),
                "prosody_characteristics": characteristics,
                "voice_quality": quality_check,
                "content_recommendations": recommendations,
                "analysis_method": enhanced_analysis.get("method", "multi_method"),
                "timestamp": datetime.now().isoformat(),
                "processing_notes": enhanced_analysis.get("notes", [])
            }
            
            # Cache the result
            self._add_to_cache(cache_key, result)
            
            # Update statistics
            if result["detected_mood"]:
                self.stats["mood_detections"] += 1
                if result["confidence"] > 0.8:
                    self.stats["high_confidence_detections"] += 1
            
            logger.debug(f"✅ Prosody analysis complete: mood='{result['detected_mood']}', confidence={result['confidence']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in prosody mood analysis: {e}", exc_info=True)
            return {"detected_mood": None, "confidence": 0.0, "error": str(e)}
    
    def _assess_prosody_quality(self, prosody_features):
        """Assess the quality of prosodic features for reliable analysis."""
        try:
            issues = []
            quality_score = 1.0
            
            # Check duration
            duration = prosody_features.get("duration", 0)
            if duration < self.quality_thresholds["min_duration"]:
                issues.append(f"Audio too short ({duration:.1f}s, min: {self.quality_thresholds['min_duration']}s)")
                quality_score *= 0.5
            elif duration > self.quality_thresholds["max_duration"]:
                issues.append(f"Audio too long ({duration:.1f}s, max: {self.quality_thresholds['max_duration']}s)")
                quality_score *= 0.8
            
            # Check energy level
            rms_energy = prosody_features.get("rms_energy", 0)
            if rms_energy < self.quality_thresholds["min_energy"]:
                issues.append(f"Low audio energy ({rms_energy:.3f}, min: {self.quality_thresholds['min_energy']})")
                quality_score *= 0.6
            
            # Check if required features are present
            required_features = ["intensity", "tempo", "estimated_pitch_hz"]
            missing_features = [f for f in required_features if f not in prosody_features]
            if missing_features:
                issues.append(f"Missing features: {missing_features}")
                quality_score *= 0.7
            
            # Check analysis method quality
            method = prosody_features.get("method", "unknown")
            method_quality = {
                "opensmile": 1.0,
                "enhanced_analysis": 0.9,
                "enhanced_python_analysis": 0.8,
                "simple_analysis": 0.6
            }
            quality_score *= method_quality.get(method, 0.5)
            
            return {
                "is_valid": len(issues) == 0 or quality_score > 0.4,
                "quality_score": round(quality_score, 2),
                "issues": issues,
                "method_quality": method_quality.get(method, 0.5)
            }
            
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            return {"is_valid": False, "quality_score": 0.0, "issues": ["Quality assessment failed"]}
    
    def _extract_prosodic_characteristics(self, prosody_features):
        """Extract and normalize prosodic characteristics."""
        try:
            # Extract raw features
            intensity = prosody_features.get("intensity", "medium")
            tempo = prosody_features.get("tempo", "medium")
            pitch_hz = prosody_features.get("estimated_pitch_hz", 150)
            energy = prosody_features.get("rms_energy", 0.1)
            duration = prosody_features.get("duration", 2.0)
            
            # Normalize features to 0-1 scale
            characteristics = {
                "intensity_level": intensity,
                "tempo_level": tempo,
                "pitch_hz": pitch_hz,
                "energy_raw": energy,
                "duration": duration,
                
                # Normalized scores
                "intensity_score": self._normalize_intensity(intensity),
                "tempo_score": self._normalize_tempo(tempo),
                "pitch_score": self._normalize_pitch(pitch_hz),
                "energy_score": min(energy * 10, 1.0),  # Normalize energy
                "duration_score": min(duration / 5.0, 1.0),  # Normalize duration (5s = 1.0)
                
                # Derived characteristics
                "voice_stability": self._calculate_voice_stability(prosody_features),
                "emotional_intensity": self._calculate_emotional_intensity(intensity, energy, pitch_hz),
                "speech_rate_factor": self._calculate_speech_rate_factor(tempo, duration)
            }
            
            return characteristics
            
        except Exception as e:
            logger.warning(f"Characteristic extraction failed: {e}")
            return {"intensity_score": 0.5, "tempo_score": 0.5, "pitch_score": 0.5, "energy_score": 0.5}
    
    def _normalize_intensity(self, intensity):
        """Normalize intensity to 0-1 score."""
        intensity_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
        return intensity_map.get(intensity, 0.5)
    
    def _normalize_tempo(self, tempo):
        """Normalize tempo to 0-1 score."""
        tempo_map = {"slow": 0.2, "medium": 0.5, "fast": 0.8}
        return tempo_map.get(tempo, 0.5)
    
    def _normalize_pitch(self, pitch_hz):
        """Normalize pitch to 0-1 score (typical range: 80-300 Hz)."""
        try:
            # Typical human speech: 80-300 Hz
            normalized = (pitch_hz - 80) / (300 - 80)
            return max(0.0, min(1.0, normalized))
        except:
            return 0.5
    
    def _calculate_voice_stability(self, prosody_features):
        """Calculate voice stability indicator."""
        try:
            # Use energy consistency as a proxy for stability
            energy = prosody_features.get("rms_energy", 0.1)
            intensity = prosody_features.get("intensity", "medium")
            
            # Higher energy + consistent intensity = more stable
            if intensity in ["medium", "high"] and energy > 0.05:
                return 0.8
            elif intensity == "low" and energy < 0.03:
                return 0.3  # Unstable (too quiet)
            else:
                return 0.6  # Moderate stability
                
        except:
            return 0.5
    
    def _calculate_emotional_intensity(self, intensity, energy, pitch_hz):
        """Calculate overall emotional intensity from voice features."""
        try:
            intensity_score = self._normalize_intensity(intensity)
            energy_score = min(energy * 10, 1.0)
            
            # Higher pitch often indicates higher emotional intensity
            pitch_intensity = 1.0 if pitch_hz > 200 else 0.7 if pitch_hz > 150 else 0.4
            
            # Combined emotional intensity
            emotional_intensity = (intensity_score * 0.4 + energy_score * 0.4 + pitch_intensity * 0.2)
            return round(emotional_intensity, 2)
            
        except:
            return 0.5
    
    def _calculate_speech_rate_factor(self, tempo, duration):
        """Calculate speech rate factor."""
        try:
            tempo_score = self._normalize_tempo(tempo)
            
            # Adjust for duration (longer speech might indicate different tempo patterns)
            duration_factor = 1.0 if duration < 5 else 0.9 if duration < 10 else 0.8
            
            return round(tempo_score * duration_factor, 2)
            
        except:
            return 0.5
    
    def _detect_mood_multi_method(self, characteristics, prosody_features):
        """Detect mood using multiple analysis methods."""
        try:
            mood_candidates = []
            
            # Method 1: Pattern-based detection
            pattern_mood = self._detect_mood_by_patterns(characteristics)
            if pattern_mood:
                mood_candidates.append(pattern_mood)
            
            # Method 2: Threshold-based detection
            threshold_mood = self._detect_mood_by_thresholds(characteristics)
            if threshold_mood:
                mood_candidates.append(threshold_mood)
            
            # Method 3: Energy-pitch analysis
            energy_pitch_mood = self._detect_mood_by_energy_pitch(characteristics)
            if energy_pitch_mood:
                mood_candidates.append(energy_pitch_mood)
            
            # Combine and rank mood candidates
            if not mood_candidates:
                return {"primary_mood": None, "confidence": 0.0, "method": "no_detection"}
            
            # Find consensus or highest confidence mood
            mood_scores = {}
            for candidate in mood_candidates:
                mood = candidate["mood"]
                confidence = candidate["confidence"]
                if mood in mood_scores:
                    mood_scores[mood] = max(mood_scores[mood], confidence)
                else:
                    mood_scores[mood] = confidence
            
            # Select primary mood (highest score)
            primary_mood = max(mood_scores.items(), key=lambda x: x[1])
            alternative_moods = [{"mood": mood, "confidence": conf} 
                               for mood, conf in mood_scores.items() 
                               if mood != primary_mood[0]]
            
            return {
                "primary_mood": primary_mood[0],
                "confidence": round(primary_mood[1], 2),
                "alternative_moods": sorted(alternative_moods, key=lambda x: x["confidence"], reverse=True)[:2],
                "method": "multi_method_consensus",
                "candidate_count": len(mood_candidates)
            }
            
        except Exception as e:
            logger.warning(f"Multi-method mood detection failed: {e}")
            return {"primary_mood": None, "confidence": 0.0, "method": "detection_failed"}
    
    def _detect_mood_by_patterns(self, characteristics):
        """Detect mood using prosodic patterns."""
        try:
            intensity = characteristics.get("intensity_level", "medium")
            tempo = characteristics.get("tempo_level", "medium")
            
            # Create pattern key
            pattern_key = f"{intensity}_intensity_{tempo}_tempo"
            
            # Check for exact pattern match
            if pattern_key in self.prosody_mood_mapping:
                pattern_data = self.prosody_mood_mapping[pattern_key]
                return {
                    "mood": pattern_data["mood"],
                    "confidence": pattern_data["confidence"],
                    "method": "pattern_match"
                }
            
            # Check for partial pattern matches
            for pattern, data in self.prosody_mood_mapping.items():
                if intensity in pattern or tempo in pattern:
                    return {
                        "mood": data["mood"],
                        "confidence": data["confidence"] * 0.7,  # Reduce confidence for partial match
                        "method": "partial_pattern_match"
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Pattern-based detection failed: {e}")
            return None
    
    def _detect_mood_by_thresholds(self, characteristics):
        """Detect mood using threshold-based analysis."""
        try:
            intensity_score = characteristics.get("intensity_score", 0.5)
            tempo_score = characteristics.get("tempo_score", 0.5)
            energy_score = characteristics.get("energy_score", 0.5)
            emotional_intensity = characteristics.get("emotional_intensity", 0.5)
            
            # Threshold-based mood detection
            if intensity_score < 0.3 and tempo_score < 0.4 and energy_score < 0.3:
                return {"mood": "sad", "confidence": 0.85, "method": "threshold_low_energy"}
            
            elif intensity_score > 0.7 and tempo_score > 0.7 and emotional_intensity > 0.7:
                return {"mood": "excited", "confidence": 0.80, "method": "threshold_high_energy"}
            
            elif intensity_score > 0.6 and tempo_score < 0.4:
                return {"mood": "angry", "confidence": 0.75, "method": "threshold_intense_slow"}
            
            elif intensity_score < 0.4 and tempo_score > 0.6:
                return {"mood": "anxious", "confidence": 0.70, "method": "threshold_quiet_fast"}
            
            elif 0.4 <= intensity_score <= 0.6 and 0.4 <= tempo_score <= 0.6:
                return {"mood": "calm", "confidence": 0.65, "method": "threshold_balanced"}
            
            return None
            
        except Exception as e:
            logger.warning(f"Threshold-based detection failed: {e}")
            return None
    
    def _detect_mood_by_energy_pitch(self, characteristics):
        """Detect mood using energy and pitch analysis."""
        try:
            energy_score = characteristics.get("energy_score", 0.5)
            pitch_score = characteristics.get("pitch_score", 0.5)
            voice_stability = characteristics.get("voice_stability", 0.5)
            
            # Energy-pitch based mood detection
            if energy_score < 0.3 and pitch_score < 0.4:
                return {"mood": "tired", "confidence": 0.70, "method": "energy_pitch_low"}
            
            elif energy_score > 0.7 and pitch_score > 0.6:
                return {"mood": "happy", "confidence": 0.75, "method": "energy_pitch_high"}
            
            elif voice_stability < 0.4 and energy_score > 0.5:
                return {"mood": "stressed", "confidence": 0.65, "method": "energy_pitch_unstable"}
            
            elif voice_stability > 0.7 and energy_score > 0.4:
                return {"mood": "confident", "confidence": 0.60, "method": "energy_pitch_stable"}
            
            return None
            
        except Exception as e:
            logger.warning(f"Energy-pitch detection failed: {e}")
            return None
    
    def _enhance_with_context(self, mood_analysis, prosody_features):
        """Enhance mood analysis with contextual information."""
        try:
            if not mood_analysis.get("primary_mood"):
                return mood_analysis
            
            enhanced = mood_analysis.copy()
            notes = []
            
            # Adjust confidence based on voice quality
            method = prosody_features.get("method", "unknown")
            if method == "opensmile":
                enhanced["confidence"] = min(enhanced["confidence"] * 1.1, 1.0)
                notes.append("High-quality analysis method")
            elif method in ["simple_analysis", "basic"]:
                enhanced["confidence"] = enhanced["confidence"] * 0.8
                notes.append("Basic analysis method - reduced confidence")
            
            # Adjust based on duration
            duration = prosody_features.get("duration", 0)
            if duration > 3:
                enhanced["confidence"] = min(enhanced["confidence"] * 1.05, 1.0)
                notes.append("Longer audio increases confidence")
            elif duration < 1.5:
                enhanced["confidence"] = enhanced["confidence"] * 0.9
                notes.append("Short audio reduces confidence")
            
            # Check for edge cases
            energy = prosody_features.get("rms_energy", 0.1)
            if energy < 0.02:
                notes.append("Very low energy - may indicate whisper or poor audio")
                enhanced["confidence"] = enhanced["confidence"] * 0.7
            
            enhanced["notes"] = notes
            enhanced["confidence"] = round(enhanced["confidence"], 2)
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Context enhancement failed: {e}")
            return mood_analysis
    
    def _generate_prosody_recommendations(self, mood_analysis):
        """Generate content recommendations based on prosody analysis."""
        try:
            mood = mood_analysis.get("primary_mood")
            confidence = mood_analysis.get("confidence", 0.0)
            
            if not mood or confidence < self.quality_thresholds["confidence_threshold"]:
                return {"recommendation_strength": "low", "note": "Insufficient confidence for specific recommendations"}
            
            # Mood-based recommendations
            mood_recommendations = {
                "sad": {
                    "boost_genres": ["Comedy", "Feel-good", "Uplifting", "Family"],
                    "boost_moods": ["cheerful", "heartwarming", "inspiring", "uplifting"],
                    "avoid_moods": ["melancholic", "dark", "depressing"],
                    "reasoning": "Comedy and uplifting content to improve mood"
                },
                "excited": {
                    "boost_genres": ["Action", "Adventure", "Comedy", "Thriller"],
                    "boost_moods": ["thrilling", "energetic", "fun", "exciting"],
                    "avoid_moods": ["slow", "boring", "dull"],
                    "reasoning": "High-energy content matching excited state"
                },
                "angry": {
                    "boost_genres": ["Action", "Drama", "Thriller"],
                    "boost_moods": ["intense", "powerful", "cathartic"],
                    "avoid_moods": ["slow", "romantic", "gentle"],
                    "reasoning": "Intense content for emotional release"
                },
                "anxious": {
                    "boost_genres": ["Comedy", "Feel-good", "Romance", "Family"],
                    "boost_moods": ["calming", "reassuring", "gentle", "peaceful"],
                    "avoid_moods": ["intense", "scary", "stressful"],
                    "reasoning": "Calming content to reduce anxiety"
                },
                "calm": {
                    "boost_genres": ["Drama", "Documentary", "Romance"],
                    "boost_moods": ["thoughtful", "peaceful", "engaging"],
                    "avoid_moods": ["chaotic", "overwhelming"],
                    "reasoning": "Balanced content for relaxed state"
                },
                "tired": {
                    "boost_genres": ["Comedy", "Feel-good", "Light-Drama"],
                    "boost_moods": ["gentle", "easy-watching", "comforting"],
                    "avoid_moods": ["complex", "intense", "demanding"],
                    "reasoning": "Easy-to-watch content when energy is low"
                },
                "stressed": {
                    "boost_genres": ["Comedy", "Feel-good", "Romance"],
                    "boost_moods": ["relaxing", "soothing", "stress-relief"],
                    "avoid_moods": ["stressful", "intense", "chaotic"],
                    "reasoning": "Stress-relieving content for relaxation"
                }
            }
            
            recommendation = mood_recommendations.get(mood, {
                "boost_genres": ["Comedy", "Feel-good"],
                "boost_moods": ["entertaining"],
                "reasoning": "General mood-appropriate content"
            })
            
            # Add confidence-based adjustments
            recommendation["recommendation_strength"] = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            recommendation["confidence_level"] = confidence
            recommendation["voice_detected_mood"] = mood
            
            return recommendation
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            return {"recommendation_strength": "low", "note": "Failed to generate recommendations"}
    
    def get_prosody_summary(self, prosody_features):
        """Get human-readable summary of prosody analysis."""
        try:
            if not prosody_features:
                return "No voice data available"
            
            intensity = prosody_features.get("intensity", "medium")
            tempo = prosody_features.get("tempo", "medium")
            duration = prosody_features.get("duration", 0)
            method = prosody_features.get("method", "unknown")
            
            summary_parts = []
            
            # Voice characteristics
            if intensity == "high":
                summary_parts.append("strong voice")
            elif intensity == "low":
                summary_parts.append("quiet voice")
            else:
                summary_parts.append("moderate voice")
            
            if tempo == "fast":
                summary_parts.append("quick speech")
            elif tempo == "slow":
                summary_parts.append("slow speech")
            else:
                summary_parts.append("normal pace")
            
            # Duration note
            if duration > 5:
                summary_parts.append("extended speech")
            elif duration < 2:
                summary_parts.append("brief input")
            
            # Quality note
            if method in ["opensmile", "enhanced_analysis"]:
                summary_parts.append("high-quality analysis")
            
            return ", ".join(summary_parts)
            
        except Exception as e:
            logger.warning(f"Prosody summary generation failed: {e}")
            return "Voice analysis available"
    
    def _generate_prosody_cache_key(self, prosody_features):
        """Generate cache key for prosody analysis."""
        try:
            # Use key features for cache key
            key_features = [
                prosody_features.get("intensity", "medium"),
                prosody_features.get("tempo", "medium"),
                str(round(prosody_features.get("rms_energy", 0.1), 3)),
                str(round(prosody_features.get("duration", 2.0), 1))
            ]
            return "|".join(key_features)
        except:
            return "default_prosody_key"
    
    def _get_from_cache(self, cache_key):
        """Get prosody analysis from cache if available and fresh."""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            age_seconds = (datetime.now() - entry['timestamp']).total_seconds()
            if age_seconds < self.cache_expiry:
                return entry['analysis']
        return None
    
    def _add_to_cache(self, cache_key, analysis):
        """Add prosody analysis to cache."""
        self.cache[cache_key] = {'timestamp': datetime.now(), 'analysis': analysis}
        
        # Manage cache size
        if len(self.cache) > 50:
            # Remove oldest 10 entries
            sorted_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            for old_key in sorted_keys[:10]:
                del self.cache[old_key]
    
    def get_prosody_statistics(self):
        """Get prosody analysis statistics."""
        cache_hit_rate = (self.stats["cache_hits"] / max(self.stats["analyses_performed"], 1)) * 100
        high_confidence_rate = (self.stats["high_confidence_detections"] / max(self.stats["mood_detections"], 1)) * 100
        
        return {
            "analyses_performed": self.stats["analyses_performed"],
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "mood_detections": self.stats["mood_detections"],
            "high_confidence_detections": self.stats["high_confidence_detections"],
            "high_confidence_rate_percent": round(high_confidence_rate, 2),
            "cache_size": len(self.cache),
            "supported_moods": list(set(data["mood"] for data in self.prosody_mood_mapping.values()))
        }
    
    def clear_cache(self):
        """Clear prosody analysis cache."""
        cache_size = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared prosody cache ({cache_size} entries)")
        return cache_size