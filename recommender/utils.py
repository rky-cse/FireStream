# utils.py
import yaml
import numpy as np
import datetime

def load_config(path='config.yaml'):
    return yaml.safe_load(open(path))

def get_profile_text(user_id):
    # query DB for user watch history summary
    return "User history summary text"

def get_closeness(u, v):
    return 0.8  # placeholder; compute from DB

def time_of_day_weight(hour):
    return 1.0 if 18 <= hour <= 23 else 0.5

def festival_weight(date_str):
    # compare date_str to festival list
    return 1.2 if date_str in ['2025-06-07', '2025-10-20'] else 1.0

def global_popularity(content_id):
    return 0.05

def load_mood_vector(path):
    return np.load(path)