# tests/test_emotion.py

import pytest
from ..app import emotion

def test_detect_emotion_text_only(monkeypatch):
    # Mock the pipeline to always return 'sadness'
    monkeypatch.setattr(
        emotion, 
        '_emotion_pipeline', 
        lambda text: [{"label": "sadness"}]
    )
    
    label = emotion.detect_emotion("I feel terrible", prosody_hints=None)
    assert label == "sadness"


def test_detect_emotion_neutral_with_low_pitch(monkeypatch):
    # Pipeline says 'neutral', but pitch is low—should stay 'neutral'
    monkeypatch.setattr(
        emotion, 
        '_emotion_pipeline', 
        lambda text: [{"label": "neutral"}]
    )
    
    prosody = {"pitch_mean": 150}
    label = emotion.detect_emotion("I am okay", prosody_hints=prosody)
    assert label == "neutral"


def test_detect_emotion_neutral_with_high_pitch(monkeypatch):
    # Pipeline says 'neutral', but high pitch triggers override to 'joy'
    monkeypatch.setattr(
        emotion, 
        '_emotion_pipeline', 
        lambda text: [{"label": "neutral"}]
    )
    
    prosody = {"pitch_mean": 250}
    label = emotion.detect_emotion("I am okay", prosody_hints=prosody)
    assert label == "joy"
