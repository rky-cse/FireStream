# processors/video_processing_service.py
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np

from detectors.scene_detector import SceneDetector
from detectors.object_detector import ObjectDetector
from processors.product_matcher import ProductMatcher
from processors.video_processors import VideoScene

logger = logging.getLogger(__name__)

@dataclass
class ProductMatchResult:
    product_name: str
    image_path: str
    match_confidence: float
    frame_number: int
    scene_number: int
    object_label: str
    object_confidence: float

class VideoProcessingService:
    def __init__(self, 
                 products_dir: str = "amazon_products",
                 scene_threshold: float = 27.0,
                 min_scene_len: int = 10,
                 object_confidence: float = 0.4,
                 match_threshold: float = 0.65,
                 process_every_n_frames: int = 15):
        """
        Initialize the video processing service with all components.
        
        Args:
            products_dir: Directory containing product images
            scene_threshold: Scene detection threshold (lower = more sensitive)
            min_scene_len: Minimum scene length in frames
            object_confidence: Minimum confidence for object detection
            match_threshold: Minimum similarity score for product matching
            process_every_n_frames: Frame sampling rate for processing
        """
        # Initialize all components
        self.scene_detector = SceneDetector(
            threshold=scene_threshold,
            min_scene_len=min_scene_len
        )
        
        self.object_detector = ObjectDetector(
            confidence_threshold=object_confidence
        )
        
        self.product_matcher = ProductMatcher(
            products_dir=products_dir,
            match_threshold=match_threshold
        )
        
        self.process_every_n_frames = process_every_n_frames

    def process_video_and_match_products(self, video_path: str) -> List[ProductMatchResult]:
        """
        Main processing pipeline that takes a video path and returns matched products.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of ProductMatchResult objects with matching products
        """
        logger.info(f"Starting video processing pipeline for: {video_path}")
        
        # Validate video file
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Initialize VideoProcessor
        from processors.video_processors import VideoProcessor
        video_processor = VideoProcessor(
            scene_detector=self.scene_detector,
            object_detector=self.object_detector,
            process_every_n_frames=self.process_every_n_frames
        )
        
        # Process the video to get scenes with objects
        scenes = video_processor.process_video(video_path)
        if not scenes:
            logger.warning("No scenes detected in the video")
            return []
        
        # Process each scene to find product matches
        matched_products = []
        for scene_idx, scene in enumerate(scenes):
            cap = cv2.VideoCapture(video_path)
            try:
                for obj in scene.objects:
                    # Skip non-product objects (optional)
                    if obj['label'] not in ['bottle', 'book', 'cell phone', 'clock', 
                                          'vase', 'remote', 'scissors', 'teddy bear', 
                                          'hair drier', 'toothbrush']:
                        continue
                    
                    # Get the object ROI from the frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, obj['frame_number'])
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    x1, y1, x2, y2 = obj['bbox']
                    object_roi = frame[y1:y2, x1:x2]
                    
                    if object_roi.size == 0:
                        continue
                    
                    # Match the object to products
                    match_result = self.product_matcher.match_to_products(object_roi, obj)
                    if match_result:
                        matched_products.append(ProductMatchResult(
                            product_name=match_result['product_name'],
                            image_path=match_result['image_path'],
                            match_confidence=match_result['match_confidence'],
                            frame_number=obj['frame_number'],
                            scene_number=scene_idx + 1,
                            object_label=obj['label'],
                            object_confidence=obj['confidence']
                        ))
            finally:
                cap.release()
        
        # Remove duplicate matches (same product in multiple frames)
        unique_matches = self._remove_duplicate_matches(matched_products)
        
        logger.info(f"Found {len(unique_matches)} unique product matches")
        return unique_matches

    def _remove_duplicate_matches(self, matches: List[ProductMatchResult]) -> List[ProductMatchResult]:
        """
        Remove duplicate matches for the same product, keeping the highest confidence one.
        """
        unique_matches = {}
        for match in matches:
            if match.product_name not in unique_matches:
                unique_matches[match.product_name] = match
            elif match.match_confidence > unique_matches[match.product_name].match_confidence:
                unique_matches[match.product_name] = match
        return list(unique_matches.values())

if __name__ == "__main__":
    """Standalone smoke test with videos/sample1.mp4"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Path configuration
    test_video = "videos/sample1.mp4"
    products_dir = "amazon_products"
    
    if not Path(test_video).exists():
        logger.error(f"Test video not found at {test_video}")
        exit(1)
    
    if not Path(products_dir).exists():
        logger.error(f"Products directory not found at {products_dir}")
        exit(1)
    
    logger.info("Running VideoProcessingService smoke test...")
    
    try:
        # Initialize service with default parameters
        service = VideoProcessingService(
            products_dir=products_dir,
            scene_threshold=27.0,
            min_scene_len=10,
            object_confidence=0.4,
            match_threshold=0.4,
            process_every_n_frames=15
        )
        
        # Process the video
        results = service.process_video_and_match_products(test_video)
        
        # Display results
        logger.info("\n=== Processing Results ===")
        logger.info(f"Video processed: {test_video}")
        logger.info(f"Products directory: {products_dir}")
        logger.info(f"Found {len(results)} unique product matches:")
        
        for idx, match in enumerate(results, 1):
            logger.info(f"\nMatch {idx}:")
            logger.info(f"Product: {match.product_name}")
            logger.info(f"Image: {match.image_path}")
            logger.info(f"Scene: {match.scene_number}")
            logger.info(f"Frame: {match.frame_number}")
            logger.info(f"Object: {match.object_label} (confidence: {match.object_confidence:.2f})")
            logger.info(f"Match confidence: {match.match_confidence:.2f}")
        
        if not results:
            logger.warning("\nNo products matched. Possible reasons:")
            logger.warning("- No recognizable products in the video")
            logger.warning("- Try lowering match_threshold parameter")
            logger.warning("- Add more product images to amazon_products directory")
        
        logger.info("\nSmoke test completed!")
        
    except Exception as e:
        logger.error(f"Smoke test failed: {str(e)}")
        exit(1)