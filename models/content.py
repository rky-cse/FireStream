from sqlalchemy import Column, String, Integer, Float, Text, ARRAY
from database import Base

class Content(Base):
    __tablename__ = "content"

    content_id = Column(String(50), primary_key=True, index=True)
    title = Column(String(200), nullable=False)
    type = Column(String(50))  # movie, show, episode, etc.
    genre = Column(ARRAY(String(100)))
    release_year = Column(Integer)
    duration = Column(Integer)  # in minutes
    director = Column(ARRAY(String(100)))
    actors = Column(ARRAY(String(100)))
    description = Column(Text)
    rating = Column(Float)
    mood_tags = Column(ARRAY(String(50)))  # e.g., funny, suspenseful, heartwarming