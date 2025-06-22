# app/intent.py
from enum import Enum, auto
import spacy
from typing import Optional, Dict, Tuple
import re
from dataclasses import dataclass
import json
from pathlib import Path

nlp = spacy.load("en_core_web_sm")

class SearchIntent(Enum):
    MOOD = auto()
    GENRE = auto()
    ACTOR = auto()
    DIRECTOR = auto()
    TITLE = auto()
    YEAR = auto()
    GENERAL = auto()

@dataclass
class IntentResult:
    intent: SearchIntent
    confidence: float
    entities: Dict[str, str] = None

class IntentClassifier:
    def __init__(self):
        self.genre_terms = self._load_genre_terms()
        self.mood_phrases = {
            'cheer up': 'happy',
            'feel better': 'happy',
            'lift my mood': 'happy',
            'make me happy': 'happy',
            'comfort me': 'happy',
            'I\'m sad': 'sad',
            'I\'m depressed': 'sad',
            'I\'m lonely': 'lonely',
            'I\'m anxious': 'stressed'
        }
        
        self.patterns = [
            (r"(movie|film) (called|named|titled) (.+)", SearchIntent.TITLE),
            (r"\"(.+)\"", SearchIntent.TITLE),
            (r"(starring|with|actor|actress) (.+)", SearchIntent.ACTOR),
            (r"(directed|director) (.+)", SearchIntent.DIRECTOR),
            (r"(from|of|year) (19\d{2}|20\d{2})", SearchIntent.YEAR),
        ]
        
        self._emotion_detector = None  # Make it private with underscore

    def _load_genre_terms(self) -> set:
        """Load genre terms from JSON file with proper error handling"""
        try:
            with open(Path(__file__).parent / "genre_terms.json") as f:
                return set(json.load(f).get("genres", []))
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load genre terms: {e}")
            return {
                'comedy', 'drama', 'action', 'horror', 'romance',
                'sci-fi', 'thriller', 'documentary', 'animation'
            }

    @property
    def emotion_detector(self):
        """Lazy-load emotion detector with proper import"""
        if self._emotion_detector is None:
            from searchRecommender.app.emotion import EmotionDetector  # Absolute import
            self._emotion_detector = EmotionDetector()
        return self._emotion_detector

    def _check_mood_phrases(self, text: str) -> Optional[str]:
        """Check for explicit mood phrases in text"""
        text_lower = text.lower()
        for phrase, mood in self.mood_phrases.items():
            if phrase in text_lower:
                return mood
        return None

    def _extract_with_patterns(self, text: str) -> Optional[Tuple[SearchIntent, Dict]]:
        """Extract intent using regex patterns"""
        doc = nlp(text.lower())
        lemmatized = " ".join([token.lemma_ for token in doc])
        
        for pattern, intent in self.patterns:
            if match := re.search(pattern, lemmatized):
                entities = {}
                if intent == SearchIntent.TITLE:
                    entities["title"] = match.group(3) if len(match.groups()) > 2 else match.group(1)
                elif intent in (SearchIntent.ACTOR, SearchIntent.DIRECTOR):
                    entities["person"] = match.group(2)
                elif intent == SearchIntent.YEAR:
                    entities["year"] = match.group(2)
                return intent, entities
        return None

    def classify_intent(self, text: str) -> IntentResult:
        """Classify user intent from text with proper error handling"""
        if not text or not text.strip():
            return IntentResult(SearchIntent.GENERAL, 1.0)
        
        try:
            # Check explicit mood phrases first
            if mood := self._check_mood_phrases(text):
                return IntentResult(
                    intent=SearchIntent.MOOD,
                    confidence=0.95,
                    entities={"mood": mood}
                )
            
            # Then check structural patterns
            if pattern_result := self._extract_with_patterns(text):
                intent, entities = pattern_result
                return IntentResult(intent, 0.9, entities)
            
            # Check for genres
            doc = nlp(text.lower())
            genres_found = {
                chunk.text for chunk in doc.noun_chunks 
                if chunk.text in self.genre_terms
            }
            genres_found.update(
                token.text for token in doc 
                if token.text in self.genre_terms and token.pos_ in ("NOUN", "ADJ")
            )
            
            if genres_found:
                return IntentResult(
                    SearchIntent.GENRE,
                    0.8,
                    {"genres": list(genres_found)}
                )
            
            # Fallback to emotion detection
            try:
                if mood := self.emotion_detector.get_dominant_mood(text):
                    if mood != 'neutral':
                        return IntentResult(
                            intent=SearchIntent.MOOD,
                            confidence=0.85,
                            entities={"mood": mood}
                        )
            except Exception as e:
                logger.warning(f"Emotion detection failed: {e}")
            
            return IntentResult(SearchIntent.GENERAL, 0.7)
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return IntentResult(SearchIntent.GENERAL, 0.5)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    classifier = IntentClassifier()
    
    test_cases = [
        ("I want something to cheer me up", SearchIntent.MOOD),
        ("Show me something to feel better", SearchIntent.MOOD),
        ("Movies with Tom Hanks", SearchIntent.ACTOR),
        ("horror films", SearchIntent.GENRE),
        ("The Shawshank Redemption", SearchIntent.TITLE),
        ("Movies from 1999", SearchIntent.YEAR),
        ("What's popular?", SearchIntent.GENERAL),
        ("Films directed by Spielberg", SearchIntent.DIRECTOR),
        ("I'm feeling really sad today", SearchIntent.MOOD)
    ]
    
    print("Running intent classification tests...")
    failures = 0
    for text, expected_intent in test_cases:
        result = classifier.classify_intent(text)
        print(f"\nInput: '{text}'")
        print(f"Detected: {result.intent.name} (Expected: {expected_intent.name})")
        print(f"Confidence: {result.confidence:.2f}")
        if result.entities:
            print(f"Entities: {result.entities}")
        
        if result.intent != expected_intent:
            print("❌ Test failed")
            failures += 1
        else:
            print("✅ Test passed")
    
    print(f"\nTest results: {len(test_cases)-failures}/{len(test_cases)} passed")
    if failures > 0:
        print("Some tests failed. Check the implementation.")
        exit(1)
    print("All tests passed successfully!")