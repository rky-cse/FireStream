# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# Get frontend origin from environment or default to dev server
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],  # Specific origin instead of wildcard
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Existing routers
from api.routers.voice import router as voice_router
from api.routers.health import router as health_router

# New routers
from api.routers.video import router as video_router
from api.routers.search import router as search_router
from api.routers.websocket import router as websocket_router

# Include all routers
app.include_router(voice_router)
app.include_router(health_router)
app.include_router(video_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(websocket_router)  # WebSocket router without /api prefix for direct WS connections