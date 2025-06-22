import logging
from typing import List, Tuple
from pathlib import Path
import cv2
import numpy as np

# PySceneDetect imports with fallback
try:
    from scenedetect import SceneManager, open_video
    from scenedetect.detectors import ContentDetector
    from scenedetect.frame_timecode import FrameTimecode
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    logging.warning("PySceneDetect not available, using basic scene detection")

logger = logging.getLogger(__name__)

class SceneDetector:
    def __init__(self, threshold: float = 30.0, min_scene_len: int = 15):
        """
        Initialize the SceneDetector with configuration.
        
        Args:
            threshold: Threshold for scene detection (lower = more sensitive)
            min_scene_len: Minimum length of a scene (in frames)
        """
        self.threshold = threshold
        self.min_scene_len = min_scene_len

    def detect_scenes(self, video_path: str) -> List[Tuple[int, int]]:
        """
        Detect scenes in a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of tuples containing (start_frame, end_frame) for each scene
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if SCENEDETECT_AVAILABLE:
            return self._detect_with_scenedetect(video_path)
        else:
            return self._detect_basic(video_path)

    def _detect_with_scenedetect(self, video_path: str) -> List[Tuple[int, int]]:
        """Scene detection using PySceneDetect library (modern API)."""
        try:
            # Create SceneManager and add detector
            scene_manager = SceneManager()
            scene_manager.add_detector(
                ContentDetector(
                    threshold=self.threshold,
                    min_scene_len=self.min_scene_len
                )
            )

            # Open video and detect scenes
            video = open_video(video_path)
            scene_manager.detect_scenes(video=video)
            scene_list = scene_manager.get_scene_list()
            
            # Convert to frame numbers
            scene_boundaries = []
            for scene in scene_list:
                start_frame = scene[0].get_frames()
                end_frame = scene[1].get_frames() - 1  # inclusive end frame
                scene_boundaries.append((start_frame, end_frame))
            
            logger.info(f"Detected {len(scene_boundaries)} scenes using PySceneDetect")
            return scene_boundaries

        except Exception as e:
            logger.error(f"PySceneDetect error: {str(e)}")
            logger.info("Falling back to basic detection")
            return self._detect_basic(video_path)

    def _detect_basic(self, video_path: str) -> List[Tuple[int, int]]:
        """Basic scene detection using OpenCV (fallback)."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Simple fixed-length scene splitting
        scene_length = int(fps * 5)  # 5-second scenes
        scenes = []
        
        for start in range(0, total_frames, scene_length):
            end = min(start + scene_length - 1, total_frames - 1)
            scenes.append((start, end))
        
        cap.release()
        logger.info(f"Created {len(scenes)} basic scenes (fallback mode)")
        return scenes

if __name__ == "__main__":
    """Standalone smoke test for SceneDetector using videos/sample.mp4"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Path to your sample video
    test_video_path = "videos/sample1.mp4"
    
    if not Path(test_video_path).exists():
        logger.error(f"Test video not found at {test_video_path}")
        logger.error("Please ensure the video exists at the specified path")
        exit(1)
    
    logger.info(f"Running SceneDetector smoke test on {test_video_path}")
    
    # Initialize detector with more sensitive settings
    detector = SceneDetector(threshold=27.0, min_scene_len=10)
    
    try:
        # Test detection
        scenes = detector.detect_scenes(test_video_path)
        
        # Get video info
        cap = cv2.VideoCapture(test_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Print results
        logger.info("\n=== Test Results ===")
        logger.info(f"Video Info: {total_frames} frames @ {fps:.1f} FPS")
        
        if not scenes:
            logger.warning("No scenes detected! Possible issues:")
            logger.warning("- Video may be too short or have no scene changes")
            logger.warning("- Try lowering threshold (currently: 27.0)")
            logger.warning("- Try reducing min_scene_len (currently: 10 frames)")
        else:
            logger.info(f"Detected {len(scenes)} scenes:")
            for i, (start_frame, end_frame) in enumerate(scenes):
                start_time = start_frame / fps
                end_time = end_frame / fps
                logger.info(
                    f"Scene {i+1}: Frames {start_frame}-{end_frame} "
                    f"(Time: {start_time:.1f}s - {end_time:.1f}s)"
                )
        
        logger.info("Smoke test completed")
        
    except Exception as e:
        logger.error(f"Smoke test failed: {str(e)}")