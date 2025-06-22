import cv2
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)

class ProductMatcher:
    def __init__(self, products_dir: str = "amazon_products", 
                 model_name: str = 'clip-ViT-B-32',
                 match_threshold: float = 0.7):
        """
        Enhanced ProductMatcher using deep learning embeddings.
        
        Args:
            products_dir: Directory with product images
            model_name: Pretrained model from sentence-transformers
                       Options: 'clip-ViT-B-32', 'clip-ViT-L-14', etc.
            match_threshold: Similarity threshold (0-1)
        """
        self.products_dir = Path(products_dir)
        self.match_threshold = match_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load pretrained CLIP model
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Precompute product embeddings
        self.product_db = self._load_products()
        
        logger.info(f"Initialized ProductMatcher with {len(self.product_db)} products "
                   f"(Device: {self.device}, Model: {model_name})")

    def _load_products(self) -> Dict[str, Dict]:
        """Load product images and precompute embeddings."""
        product_db = {}
        
        if not self.products_dir.exists():
            raise FileNotFoundError(f"Directory not found: {self.products_dir}")

        for img_path in self.products_dir.glob("*.*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                continue
                
            try:
                # Load image with PIL (required by CLIP)
                img = Image.open(img_path)
                product_name = img_path.stem.replace("_", " ").title()
                
                # Compute embedding
                embedding = self.model.encode(img, convert_to_tensor=True)
                
                product_db[product_name] = {
                    'image_path': str(img_path),
                    'embedding': embedding,
                    'aspect_ratio': img.width / img.height
                }
                logger.debug(f"Loaded product: {product_name}")
                
            except Exception as e:
                logger.error(f"Error loading {img_path.name}: {str(e)}")

        logger.info(f"Loaded {len(product_db)} products")
        return product_db

    def match_to_products(self, query_img: np.ndarray, object_data: Dict) -> Optional[Dict]:
        """
        Match object to products using deep learning embeddings.
        
        Args:
            query_img: Detected object ROI (OpenCV BGR format)
            object_data: {
                'label': object class,
                'confidence': detection score,
                'bbox': [x1,y1,x2,y2]
            }
            
        Returns:
            {
                'product_name': str,
                'image_path': str,
                'similarity': float,
                'match_confidence': float (combined score)
            } or None
        """
        if not self.product_db:
            logger.error("No products available for matching")
            return None

        try:
            # Convert OpenCV BGR to PIL RGB
            query_pil = Image.fromarray(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
            
            # Compute query embedding
            query_embedding = self.model.encode(query_pil, convert_to_tensor=True)
            
            # Find most similar product
            best_match = None
            best_score = self.match_threshold
            
            for product_name, product_data in self.product_db.items():
                # Compute cosine similarity
                cos_score = util.cos_sim(query_embedding, product_data['embedding'])[0][0].item()
                
                # Combine with detection confidence
                combined_score = (cos_score * 0.7) + (object_data['confidence'] * 0.3)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = {
                        'product_name': product_name,
                        'image_path': product_data['image_path'],
                        'similarity': cos_score,
                        'match_confidence': combined_score
                    }
            
            return best_match

        except Exception as e:
            logger.error(f"Matching failed: {str(e)}")
            return None


# Smoke Test
# Updated Smoke Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Enhanced Product Matcher Test ===")
    
    # Initialize
    matcher = ProductMatcher(
        products_dir="amazon_products",
        model_name='clip-ViT-B-32',
        match_threshold=0.6
    )
    
    # Check if we have enough products for testing
    if len(matcher.product_db) < 2:
        print("Error: Need at least 2 products in amazon_products directory for testing")
        exit(1)
        
    # Get test products
    product_items = list(matcher.product_db.items())
    product1_name, product1_data = product_items[3]
    product2_name, product2_data = product_items[5]
    
    # Load test images
    def load_test_image(product_data):
        img = cv2.imread(product_data['image_path'])
        if img is None:
            print(f"Error loading test image: {product_data['image_path']}")
            exit(1)
        return img
    
    product1_img = load_test_image(product1_data)
    product2_img = load_test_image(product2_data)
    
    # Test 1: Self-matching (should match with high confidence)
    print("\nTest 1: Self-matching...")
    self_match_result = matcher.match_to_products(
        product1_img,
        {'label': 'product', 'confidence': 0.9, 'bbox': [0, 0, product1_img.shape[1], product1_img.shape[0]]}
    )
    
    if self_match_result:
        print("✅ Self-match successful:")
        print(f"Expected: {product1_name}")
        print(f"Matched: {self_match_result['product_name']}")
        print(f"Similarity: {self_match_result['similarity']:.3f}")
        print(f"Confidence: {self_match_result['match_confidence']:.3f}")
        
        if self_match_result['product_name'] != product1_name:
            print("⚠️ Warning: Matched different product than expected")
    else:
        print("❌ Self-match failed")
    
    # Test 2: Cross-matching (should either not match or match with lower confidence)
    print("\nTest 2: Cross-matching different products...")
    cross_match_result = matcher.match_to_products(
        product1_img,
        {'label': 'product', 'confidence': 0.9, 'bbox': [0, 0, product2_img.shape[1], product2_img.shape[0]]}
    )
    
    if cross_match_result:
        print(f"Cross-match result (might be correct if products are similar):")
        print(f"Query: {product1_name}")
        print(f"Matched: {cross_match_result['product_name']}")
        print(f"Similarity: {cross_match_result['similarity']:.3f}")
        print(f"Confidence: {cross_match_result['match_confidence']:.3f}")
        
        if cross_match_result['product_name'] == product1_name:
            print("✅ Correctly matched to original product")
        elif cross_match_result['similarity'] < 0.8:
            print("⚠️ Low similarity match - may be correct for dissimilar products")
        else:
            print("⚠️ High similarity match - products may be visually similar")
    else:
        print("✅ No match found (expected for dissimilar products)")
    
    # Test 3: Try matching with a partial/cropped image
    print("\nTest 3: Matching with partial/cropped image...")
    cropped_img = product1_img[product1_img.shape[0]//4:3*product1_img.shape[0]//4,
                              product1_img.shape[1]//4:3*product1_img.shape[1]//4]
    
    partial_match_result = matcher.match_to_products(
        cropped_img,
        {'label': 'product', 'confidence': 0.8, 'bbox': [0, 0, cropped_img.shape[1], cropped_img.shape[0]]}
    )
    
    if partial_match_result:
        print("Partial image match result:")
        print(f"Expected: {product1_name}")
        print(f"Matched: {partial_match_result['product_name']}")
        print(f"Similarity: {partial_match_result['similarity']:.3f}")
        
        if partial_match_result['product_name'] == product1_name:
            print("✅ Correctly matched partial image to original product")
        else:
            print("⚠️ Matched to different product - may need higher threshold")
    else:
        print("❌ No match found for partial image (may need lower threshold)")