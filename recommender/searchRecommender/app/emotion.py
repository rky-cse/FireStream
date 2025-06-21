from typing import Optional, Dict, List
from transformers import pipeline
from dataclasses import dataclass
import numpy as np
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings('ignore')

@dataclass
class EmotionResult:
    label: str
    score: float
    mood: str  # Our simplified mood category

class EmotionDetector:
    """
    Advanced emotion detection using Hugging Face's transformer models.
    Maps detailed emotions to broader mood categories suitable for recommendations.
    """
    
    # Map model outputs to our mood categories
    MOOD_MAPPING = {
        'sadness': 'sad',
        'joy': 'happy',
        'anger': 'angry',
        'fear': 'stressed',
        'surprise': 'excited',
        'disgust': 'angry',
        'neutral': 'neutral',
        'excitement': 'excited',
        'frustration': 'stressed',
        'loneliness': 'lonely',
        'tiredness': 'tired'
    }

    def __init__(self, model_name: str = "SamLowe/roberta-base-go_emotions"):
        """
        Initialize with a pre-trained emotion model.
        Available models:
        - "SamLowe/roberta-base-go_emotions" (28 emotions)
        - "bert-base-uncased-emotion" (6 basic emotions)
        """
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            top_k=None,
            device=-1  # Use CPU (-1) or GPU (0)
        )
    
    def detect_emotion(self, text: str, top_n: int = 3) -> List[EmotionResult]:
        """
        Detect emotions from text with confidence scores.
        Returns top N emotions mapped to our mood categories.
        """
        if not text.strip():
            return []
            
        # Get raw predictions
        predictions = self.classifier(text)[0]
        
        # Process results
        results = []
        for pred in predictions[:top_n]:
            results.append(EmotionResult(
                label=pred['label'],
                score=float(pred['score']),
                mood=self.MOOD_MAPPING.get(
                    pred['label'].lower(), 
                    'neutral'
                )
            ))
        
        return results
    
    def get_dominant_mood(self, text: str) -> Optional[str]:
        """
        Get the single most relevant mood category for recommendation purposes.
        Applies thresholding to avoid weak classifications.
        """
        emotions = self.detect_emotion(text)
        if not emotions:
            return None
            
        # Filter by confidence threshold
        valid_emotions = [e for e in emotions if e.score >= 0.3]
        if not valid_emotions:
            return None
            
        # Get mood with highest aggregate score
        mood_scores = {}
        for e in valid_emotions:
            mood_scores[e.mood] = mood_scores.get(e.mood, 0) + e.score
        
        return max(mood_scores.items(), key=lambda x: x[1])[0]

# Smoke Tests
if __name__ == "__main__":
    import time
    
    def run_test_case(detector: EmotionDetector, text: str):
        start = time.time()
        emotions = detector.detect_emotion(text)
        dominant = detector.get_dominant_mood(text)
        elapsed = (time.time() - start) * 1000
        
        print(f"\nInput: '{text}'")
        print(f"Dominant Mood: {dominant}")
        print("Top Emotions:")
        for e in emotions:
            print(f"- {e.label} ({e.score:.2f}) → {e.mood}")
        print(f"Time: {elapsed:.1f}ms")
    
    print("Initializing emotion detector...")
    detector = EmotionDetector()
    
    test_cases = [
        "I'm feeling really sad and lonely today",
        "Wow! This is amazing! I'm so happy!",
        "I'm so angry at what happened",
        "The constant pressure at work is overwhelming",
        "Just a normal day with nothing special",
        ""
    ]
    
    for case in test_cases:
        run_test_case(detector, case)
    
    # Performance test
    long_text = """
    I've been feeling extremely anxious lately. The combination of work deadlines 
    and personal issues has left me exhausted. Yesterday I broke down crying 
    after another stressful meeting. My friends say I should take a vacation, 
    but I'm worried about falling behind.
    """
    run_test_case(detector, long_text.strip())