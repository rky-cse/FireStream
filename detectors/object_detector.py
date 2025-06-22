import logging
import cv2
import numpy as np
from typing import List, Dict
from pathlib import Path
import random

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_name: str = 'yolov5s', confidence_threshold: float = 0.5):
        """
        Initialize the YOLOv5 object detector.
        
        Args:
            model_name: YOLOv5 model variant (n, s, m, l, x)
            confidence_threshold: Minimum confidence score for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = self._load_model(model_name)

    def _load_model(self, model_name: str):
        """Load YOLOv5 model from torch hub."""
        try:
            import torch
            model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            logger.info(f"Loaded YOLOv5 model: {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error loading YOLOv5 model: {str(e)}")
            raise

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame using YOLOv5.
        
        Args:
            frame: Input frame in BGR format (OpenCV default)
            
        Returns:
            List of detected objects with labels, confidence, and bounding boxes
        """
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(rgb_frame)
            
            # Parse results
            detections = []
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf >= self.confidence_threshold:
                    detections.append({
                        'label': results.names[int(cls)],
                        'confidence': float(conf),
                        'bbox': [int(x) for x in xyxy]  # [x1, y1, x2, y2]
                    })
            
            logger.debug(f"Detected {len(detections)} objects in frame")
            return detections

        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            raise

if __name__ == "__main__":
    """Standalone smoke test using amazon_products images"""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize detector
    logger.info("Running ObjectDetector smoke test with amazon_products...")
    detector = ObjectDetector(model_name='yolov5s', confidence_threshold=0.4)
    
    # Path to your product images
    products_dir = Path("amazon_products")
    if not products_dir.exists():
        logger.error(f"Directory not found: {products_dir}")
        exit(1)
    
    # Get all image files
    image_files = list(products_dir.glob("*.*"))
    if not image_files:
        logger.error(f"No images found in {products_dir}")
        exit(1)
    
    # Select 3 random test images
    test_images = random.sample(image_files, min(3, len(image_files)))
    
    for img_path in test_images:
        try:
            # Load image
            logger.info(f"\nTesting with {img_path.name}...")
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Could not read {img_path}")
                continue
            
            # Run detection
            objects = detector.detect_objects(img)
            
            # Show results
            logger.info(f"Detected {len(objects)} objects:")
            for obj in objects:
                logger.info(f"- {obj['label']} (confidence: {obj['confidence']:.2f})")
                
                # Draw bounding box on image for visualization
                x1, y1, x2, y2 = obj['bbox']
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{obj['label']} {obj['confidence']:.2f}", 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # Display results (press any key to continue)
            cv2.imshow(f"Detections: {img_path.name}", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {str(e)}")
    
    logger.info("Smoke test completed")