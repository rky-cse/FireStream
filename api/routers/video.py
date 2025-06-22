from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
import os
import logging
from models.content import Content
from database import get_db
from sqlalchemy.orm import Session

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/video",  # This will make endpoints /api/video/... (due to main.py prefix)
    tags=["video"]
)

VIDEO_DIR = "videos"

@router.get("/stream/{content_id}")
async def stream_video(content_id: str, db: Session = Depends(get_db)):
    logger.info(f"Received request to stream video with content_id: {content_id}")
    
    try:
        # Get content metadata
        content = db.query(Content).filter(Content.content_id == content_id).first()
        
        if not content:
            logger.error(f"Content not found for content_id: {content_id}")
            raise HTTPException(status_code=404, detail="Content not found")
        
        video_path = os.path.join(VIDEO_DIR, "sample1.mp4")
        
        if not os.path.exists(video_path):
            logger.error(f"Video file not found at path: {video_path}")
            raise HTTPException(status_code=404, detail="Video file not found")
        
        def iterfile():
            with open(video_path, mode="rb") as file_like:
                yield from file_like
        
        # Create response with CORS headers
        response = StreamingResponse(
            iterfile(),
            media_type="video/mp4",
            headers={
                "Access-Control-Allow-Origin": "*",
                "Content-Disposition": f"inline; filename={content_id}.mp4"
            }
        )
        
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/metadata/{content_id}")
async def get_video_metadata(content_id: str, db: Session = Depends(get_db)):
    logger.info(f"Fetching metadata for content_id: {content_id}")
    
    try:
        content = db.query(Content).filter(Content.content_id == content_id).first()
        
        if not content:
            raise HTTPException(status_code=404, detail="Content not found")
        
        return {
            "content_id": content.content_id,
            "title": content.title,
            "type": content.type,
            "genre": content.genre,
            "release_year": content.release_year,
            "duration": content.duration,
            "director": content.director,
            "actors": content.actors,
            "description": content.description,
            "rating": content.rating,
            "mood_tags": content.mood_tags
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Metadata error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Test endpoint to verify CORS
@router.get("/test-cors")
async def test_cors():
    return {
        "message": "CORS test successful",
        "note": "If you see this, CORS is working properly"
    }