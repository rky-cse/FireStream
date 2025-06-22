import cv2
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from detectors.scene_detector import SceneDetector
from detectors.object_detector import ObjectDetector

logger = logging.getLogger(__name__)

@dataclass
class VideoScene:
    start_frame: int
    end_frame: int
    key_frame: int
    objects: List[Dict]
    frame_width: int
    frame_height: int
    fps: float

class VideoProcessor:
    def __init__(self, 
                 scene_detector: SceneDetector, 
                 object_detector: ObjectDetector,
                 process_every_n_frames: int = 10):
        """
        Initialize the VideoProcessor with detectors and configuration.
        
        Args:
            scene_detector: Configured SceneDetector instance
            object_detector: Configured ObjectDetector instance
            process_every_n_frames: Frame sampling rate for object detection
        """
        self.scene_detector = scene_detector
        self.object_detector = object_detector
        self.process_every_n_frames = process_every_n_frames

    def process_video(self, video_path: str) -> List[VideoScene]:
        """
        Main processing pipeline for a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of processed VideoScene objects
        """
        logger.info(f"Starting video processing: {video_path}")
        
        # Validate video file
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        try:
            # Get video metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

            # Step 1: Detect scenes
            scene_boundaries = self.scene_detector.detect_scenes(video_path)
            logger.info(f"Detected {len(scene_boundaries)} scenes")

            scenes = []
            for i, (start_frame, end_frame) in enumerate(scene_boundaries):
                # Step 2: Process each scene
                scene_objects = self._process_scene(cap, start_frame, end_frame)
                
                # Create scene object
                scenes.append(VideoScene(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    key_frame=start_frame + (end_frame - start_frame) // 2,
                    objects=scene_objects,
                    frame_width=width,
                    frame_height=height,
                    fps=fps
                ))
                
                logger.info(f"Processed scene {i+1}/{len(scene_boundaries)}: "
                          f"frames {start_frame}-{end_frame}, "
                          f"found {len(scene_objects)} objects")

            return scenes

        finally:
            cap.release()

    def _process_scene(self, 
                      cap: cv2.VideoCapture, 
                      start_frame: int, 
                      end_frame: int) -> List[Dict]:
        """
        Process a single scene to detect objects.
        
        Args:
            cap: OpenCV VideoCapture instance
            start_frame: Scene start frame
            end_frame: Scene end frame
            
        Returns:
            List of detected objects with metadata
        """
        objects = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            # Only process every N frames for performance
            if current_frame % self.process_every_n_frames == 0:
                detected_objects = self.object_detector.detect_objects(frame)
                for obj in detected_objects:
                    obj['frame_number'] = current_frame
                    objects.append(obj)

            current_frame += 1

        return objects

    def get_key_frames(self, video_path: str) -> List[int]:
        """Convenience method to get key frames only."""
        scenes = self.process_video(video_path)
        return [scene.key_frame for scene in scenes]

if __name__ == "__main__":
    """Standalone smoke test for VideoProcessor using videos/sample.mp4"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Path to your test video
    test_video_path = "videos/sample1.mp4"
    
    if not Path(test_video_path).exists():
        logger.error(f"Test video not found at {test_video_path}")
        logger.error("Please ensure the video exists at the specified path")
        exit(1)
    
    logger.info(f"Running VideoProcessor smoke test on {test_video_path}")
    
    # Initialize detectors
    from detectors.scene_detector import SceneDetector
    from detectors.object_detector import ObjectDetector
    
    scene_detector = SceneDetector(threshold=27.0, min_scene_len=10)
    object_detector = ObjectDetector(confidence_threshold=0.4)
    
    # Initialize processor with conservative settings
    processor = VideoProcessor(
        scene_detector=scene_detector,
        object_detector=object_detector,
        process_every_n_frames=15  # Process every 15 frames for faster testing
    )
    
    try:
        # Process the video
        scenes = processor.process_video(test_video_path)
        
        # Get video info for reporting
        cap = cv2.VideoCapture(test_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        # Print comprehensive results
        logger.info("\n=== Test Results ===")
        logger.info(f"Video: {test_video_path}")
        logger.info(f"Duration: {total_frames/fps:.1f}s ({total_frames} frames @ {fps:.1f} FPS)")
        logger.info(f"Detected {len(scenes)} scenes:")
        
        if not scenes:
            logger.warning("No scenes detected! Possible issues:")
            logger.warning("- Video may be too short or have no scene changes")
            logger.warning("- Try lowering threshold (currently: 27.0)")
            logger.warning("- Try reducing min_scene_len (currently: 10 frames)")
        else:
            for i, scene in enumerate(scenes):
                # Calculate time in seconds
                start_time = scene.start_frame / fps
                end_time = scene.end_frame / fps
                
                logger.info(f"\nScene {i+1}:")
                logger.info(f"Frames: {scene.start_frame}-{scene.end_frame}")
                logger.info(f"Time: {start_time:.1f}s - {end_time:.1f}s")
                logger.info(f"Key frame: {scene.key_frame}")
                logger.info(f"Objects detected: {len(scene.objects)}")
                
                # Log all detected objects if verbose
                for obj in scene.objects[:3]:  # Show first 3 objects max
                    logger.info(f"- {obj['label']} (conf: {obj['confidence']:.2f})")
                if len(scene.objects) > 3:
                    logger.info(f"- ... and {len(scene.objects)-3} more objects")
        
        logger.info("\nSmoke test completed successfully!")
        
    except Exception as e:
        logger.error(f"Smoke test failed: {str(e)}")
        exit(1)