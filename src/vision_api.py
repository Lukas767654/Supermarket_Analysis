"""
Google Vision API integration for hybrid crop creation and analysis.
"""
import logging
import os
import requests
import base64
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

from .config import config
from .models import CropContext

logger = logging.getLogger(__name__)

class VisionClient:
    """Google Vision API client using HTTP API for hybrid approach."""
    
    def __init__(self):
        """Initialize Vision API client."""
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            logger.error("No GOOGLE_API_KEY found in environment")
            self.client = None
        else:
            self.client = True
            logger.info("Vision API HTTP client initialized successfully")
    
    def detect_objects(self, image_bytes: bytes, 
                      min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Create hybrid crops using labels + grid + text detection.
        This method is called by the pipeline but returns empty list
        since we create crops directly in create_hybrid_crops().
        """
        logger.info("Using hybrid crop detection approach")
        return []  # Hybrid crops are created in create_hybrid_crops
    
    def detect_text(self, image_bytes: bytes, 
                   mode: str = 'TEXT_DETECTION',
                   language_hints: List[str] = None) -> Dict[str, Any]:
        """
        Detect text in image using Vision API OCR via HTTP.
        """
        if not self.client:
            logger.error("Vision client not initialized")
            return {'text': '', 'confidence': 0.0, 'words': []}
        
        try:
            # Encode image
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # API request
            url = "https://vision.googleapis.com/v1/images:annotate"
            payload = {
                "requests": [{
                    "image": {"content": image_b64},
                    "features": [{"type": mode, "maxResults": 200}]
                }]
            }
            
            response = requests.post(f"{url}?key={self.api_key}", json=payload, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Vision API OCR error: {response.status_code} - {response.text}")
                return {'text': '', 'confidence': 0.0, 'words': []}
            
            result = response.json()
            if 'responses' not in result:
                return {'text': '', 'confidence': 0.0, 'words': []}
            
            response_data = result['responses'][0]
            
            # Extract full text
            full_text = ""
            if 'textAnnotations' in response_data and response_data['textAnnotations']:
                full_text = response_data['textAnnotations'][0]['description']
            
            # Extract words
            words = []
            if 'textAnnotations' in response_data:
                for annotation in response_data['textAnnotations'][1:]:  # Skip first (full text)
                    if 'boundingPoly' in annotation:
                        vertices = annotation['boundingPoly']['vertices']
                        bbox = [
                            min(v.get('x', 0) for v in vertices),
                            min(v.get('y', 0) for v in vertices),
                            max(v.get('x', 0) for v in vertices),
                            max(v.get('y', 0) for v in vertices)
                        ]
                        
                        words.append({
                            'text': annotation['description'],
                            'confidence': 0.8,
                            'bbox': bbox
                        })
            
            return {
                'text': full_text,
                'confidence': 0.8 if words else 0.0,
                'words': words
            }
            
        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return {'text': '', 'confidence': 0.0, 'words': []}
    
    def detect_logos(self, image_bytes: bytes,
                    min_confidence: float = None) -> List[Dict[str, Any]]:
        """
        Detect logos in image using Vision API via HTTP.
        """
        if not self.client:
            logger.error("Vision client not initialized")
            return []
        
        if min_confidence is None:
            min_confidence = config.LOGO_MIN_CONF
        
        try:
            # Encode image
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # API request
            url = "https://vision.googleapis.com/v1/images:annotate"
            payload = {
                "requests": [{
                    "image": {"content": image_b64},
                    "features": [{"type": "LOGO_DETECTION", "maxResults": 50}]
                }]
            }
            
            response = requests.post(f"{url}?key={self.api_key}", json=payload, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Vision API logo error: {response.status_code} - {response.text}")
                return []
            
            result = response.json()
            if 'responses' not in result:
                return []
            
            response_data = result['responses'][0]
            
            logos = []
            if 'logoAnnotations' in response_data:
                for logo in response_data['logoAnnotations']:
                    if logo.get('score', 0) >= min_confidence:
                        vertices = logo['boundingPoly']['vertices']
                        bbox = [
                            min(v.get('x', 0) for v in vertices),
                            min(v.get('y', 0) for v in vertices),
                            max(v.get('x', 0) for v in vertices),
                            max(v.get('y', 0) for v in vertices)
                        ]
                        
                        logos.append({
                            'description': logo['description'],
                            'confidence': logo['score'],
                            'bbox': bbox,
                            'mid': logo.get('mid')
                        })
            
            logger.debug(f"Detected {len(logos)} logos")
            return logos
            
        except Exception as e:
            logger.error(f"Logo detection failed: {e}")
            return []
    
    def get_labels_and_text(self, image_bytes: bytes) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Get both labels and text regions from Vision API for hybrid crop creation.
        """
        if not self.client:
            return [], []
        
        try:
            # Encode image
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # API request for both labels and text
            url = "https://vision.googleapis.com/v1/images:annotate"
            payload = {
                "requests": [{
                    "image": {"content": image_b64},
                    "features": [
                        {"type": "LABEL_DETECTION", "maxResults": 50},
                        {"type": "TEXT_DETECTION", "maxResults": 200}
                    ]
                }]
            }
            
            response = requests.post(f"{url}?key={self.api_key}", json=payload, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return [], []
            
            result = response.json()
            if 'responses' not in result:
                return [], []
            
            response_data = result['responses'][0]
            
            # Extract labels
            labels = []
            if 'labelAnnotations' in response_data:
                for label in response_data['labelAnnotations']:
                    labels.append({
                        'name': label['description'],
                        'confidence': label['score'],
                        'mid': label.get('mid', '')
                    })
            
            # Extract text regions
            text_regions = []
            if 'textAnnotations' in response_data:
                for annotation in response_data['textAnnotations'][1:]:  # Skip first (full text)
                    if 'boundingPoly' in annotation:
                        vertices = annotation['boundingPoly']['vertices']
                        x_coords = [v.get('x', 0) for v in vertices]
                        y_coords = [v.get('y', 0) for v in vertices]
                        
                        if x_coords and y_coords:
                            text_regions.append({
                                'text': annotation['description'],
                                'bbox': [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                            })
            
            return labels, text_regions
            
        except Exception as e:
            logger.error(f"Labels and text detection failed: {e}")
            return [], []

class CropProcessor:
    """Processes crops using hybrid approach."""
    
    def __init__(self, vision_client: VisionClient):
        self.vision_client = vision_client
    
    def create_hybrid_crops(self, image: Image.Image, image_id: str) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """
        Create hybrid crops using multiple strategies.
        """
        try:
            # Convert image to bytes
            import io
            image_bytes = io.BytesIO()
            image.save(image_bytes, format='JPEG', quality=95)
            image_bytes = image_bytes.getvalue()
            
            # Get labels and text from Vision API
            labels, text_regions = self.vision_client.get_labels_and_text(image_bytes)
            
            logger.info(f"🏷️  Labels detected: {len(labels)}")
            logger.info(f"🔤 Text regions: {len(text_regions)}")
            
            # Create intelligent crops
            crops = self._create_intelligent_crops(image, labels, text_regions, image_id)
            
            logger.info(f"📦 Total hybrid crops created: {len(crops)}")
            return crops
            
        except Exception as e:
            logger.error(f"Hybrid crop creation failed: {e}")
            return []
    
    def _create_intelligent_crops(self, image: Image.Image, labels: List[Dict], 
                                 text_regions: List[Dict], image_id: str) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """Create intelligent crops using multiple strategies."""
        
        width, height = image.size
        crops = []
        
        # Strategy 1: Dense overlapping grid (comprehensive coverage)
        grid_crops = self._create_overlapping_grid_crops(image, image_id, grid_size=8, overlap=0.3)
        crops.extend(grid_crops)
        
        # Strategy 2: Text-region based crops (around detected text)
        text_crops = self._create_text_region_crops(image, text_regions, image_id)
        crops.extend(text_crops)
        
        # Strategy 3: Content-aware crops (retail-focused areas)
        content_crops = self._create_content_aware_crops(image, labels, image_id)
        crops.extend(content_crops)
        
        # Strategy 4: Multi-scale crops (different sizes)
        scale_crops = self._create_multi_scale_crops(image, image_id)
        crops.extend(scale_crops)
        
        logger.info(f"   Grid: {len(grid_crops)}, Text: {len(text_crops)}, Content: {len(content_crops)}, Scale: {len(scale_crops)}")
        
        return crops
    
    def _create_overlapping_grid_crops(self, image: Image.Image, image_id: str, 
                                     grid_size: int = 8, overlap: float = 0.3) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """Create overlapping grid crops for maximum coverage."""
        
        width, height = image.size
        crops = []
        
        step_x = int((width / grid_size) * (1 - overlap))
        step_y = int((height / grid_size) * (1 - overlap))
        crop_w = width // grid_size
        crop_h = height // grid_size
        
        crop_index = 0
        for y in range(0, height - crop_h + 1, step_y):
            for x in range(0, width - crop_w + 1, step_x):
                bbox = [x, y, x + crop_w, y + crop_h]
                crop_image = image.crop(bbox)
                
                crop_info = {
                    'crop_id': f"{image_id}_grid_{crop_index:03d}",
                    'bbox_pixel': bbox,
                    'bbox_normalized': [x/width, y/height, (x+crop_w)/width, (y+crop_h)/height],
                    'type': 'grid_regular',
                    'detector_confidence': 0.6,
                    'detector_class': 'grid_crop'
                }
                
                crops.append((crop_image, crop_info))
                crop_index += 1
        
        return crops
    
    def _create_text_region_crops(self, image: Image.Image, text_regions: List[Dict], 
                                image_id: str) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """Create crops around text regions with padding."""
        
        width, height = image.size
        crops = []
        
        for i, text_region in enumerate(text_regions):
            bbox = text_region['bbox']
            text = text_region['text']
            
            # Add padding around text
            padding = 50
            x1 = max(0, bbox[0] - padding)
            y1 = max(0, bbox[1] - padding)
            x2 = min(width, bbox[2] + padding)
            y2 = min(height, bbox[3] + padding)
            
            # Only create crop if it's reasonably sized
            if (x2 - x1) >= 80 and (y2 - y1) >= 80:
                crop_bbox = [x1, y1, x2, y2]
                crop_image = image.crop(crop_bbox)
                
                # Classify crop type based on text
                crop_type = self._classify_text_crop(text)
                
                crop_info = {
                    'crop_id': f"{image_id}_text_{crop_type}_{i:03d}",
                    'bbox_pixel': crop_bbox,
                    'bbox_normalized': [x1/width, y1/height, x2/width, y2/height],
                    'type': f'text_{crop_type}',
                    'detector_confidence': 0.8,
                    'detector_class': f'text_{crop_type}',
                    'text_content': text
                }
                
                crops.append((crop_image, crop_info))
        
        return crops
    
    def _create_content_aware_crops(self, image: Image.Image, labels: List[Dict], 
                                  image_id: str) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """Create crops based on detected labels (retail-focused)."""
        
        width, height = image.size
        crops = []
        
        # Check if we have retail-related labels
        retail_labels = [
            'shelf', 'product', 'bottle', 'package', 'food', 'drink',
            'supermarket', 'store', 'retail', 'goods', 'container'
        ]
        
        has_retail_content = any(
            any(keyword in label['name'].lower() for keyword in retail_labels)
            for label in labels
        )
        
        if has_retail_content:
            # Create strategic shelf area crops
            shelf_areas = [
                ('upper_shelf', [0, 0, width, height // 3], 0.7),
                ('middle_shelf', [0, height // 3, width, 2 * height // 3], 0.8),
                ('lower_shelf', [0, 2 * height // 3, width, height], 0.8),
                ('left_products', [0, 0, width // 2, height], 0.7),
                ('right_products', [width // 2, 0, width, height], 0.7),
            ]
            
            for area_name, bbox, confidence in shelf_areas:
                crop_image = image.crop(bbox)
                
                crop_info = {
                    'crop_id': f"{image_id}_content_{area_name}",
                    'bbox_pixel': bbox,
                    'bbox_normalized': [bbox[0]/width, bbox[1]/height, bbox[2]/width, bbox[3]/height],
                    'type': f'content_{area_name}',
                    'detector_confidence': confidence,
                    'detector_class': 'retail_area'
                }
                
                crops.append((crop_image, crop_info))
        
        return crops
    
    def _create_multi_scale_crops(self, image: Image.Image, image_id: str) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """Create crops at different scales."""
        
        width, height = image.size
        crops = []
        
        # Large crops (1/4 of image)
        crop_w = width // 2
        crop_h = height // 2
        
        large_positions = [(0, 0), (0, height - crop_h), (width - crop_w, 0), (width - crop_w, height - crop_h)]
        
        for i, (x, y) in enumerate(large_positions):
            bbox = [x, y, x + crop_w, y + crop_h]
            crop_image = image.crop(bbox)
            
            crop_info = {
                'crop_id': f"{image_id}_scale_large_{i}",
                'bbox_pixel': bbox,
                'bbox_normalized': [x/width, y/height, (x+crop_w)/width, (y+crop_h)/height],
                'type': 'scale_large',
                'detector_confidence': 0.6,
                'detector_class': 'large_scale'
            }
            
            crops.append((crop_image, crop_info))
        
        # Medium crops (1/9 of image)
        crop_w = width // 3
        crop_h = height // 3
        
        medium_index = 0
        for y in range(0, height - crop_h + 1, crop_h):
            for x in range(0, width - crop_w + 1, crop_w):
                bbox = [x, y, x + crop_w, y + crop_h]
                crop_image = image.crop(bbox)
                
                crop_info = {
                    'crop_id': f"{image_id}_scale_medium_{medium_index}",
                    'bbox_pixel': bbox,
                    'bbox_normalized': [x/width, y/height, (x+crop_w)/width, (y+crop_h)/height],
                    'type': 'scale_medium',
                    'detector_confidence': 0.7,
                    'detector_class': 'medium_scale'
                }
                
                crops.append((crop_image, crop_info))
                medium_index += 1
        
        return crops
    
    def _classify_text_crop(self, text: str) -> str:
        """Classify crop based on text content."""
        text_lower = text.lower().strip()
        
        # Price patterns
        if any(c in text for c in ['$', '฿', '€', '£']) or text.replace('.', '').replace(',', '').isdigit():
            return 'price'
        
        # Thai text
        if any('\u0e00' <= c <= '\u0e7f' for c in text):
            return 'thai_name'
        
        # Size info
        if any(unit in text_lower for unit in ['ml', 'g', 'kg', 'l', 'oz', 'pack']):
            return 'size'
        
        # Brand names (mostly English uppercase)
        if text.isupper() and len(text) > 2:
            return 'brand'
        
        return 'general'
    
    def create_crops_from_objects(self, image: Image.Image, image_id: str,
                                 objects: List[Dict[str, Any]],
                                 min_size: int = 50) -> List[Tuple[Image.Image, Dict[str, Any]]]:
        """
        Create crops - now uses hybrid approach instead of object detection.
        """
        logger.info("Using hybrid crop creation approach")
        return self.create_hybrid_crops(image, image_id)
    
    def process_crop(self, crop_image: Image.Image, crop_info: Dict[str, Any],
                    image_id: str) -> Optional[CropContext]:
        """
        Process individual crop with OCR and logo detection.
        """
        try:
            # Convert crop to bytes
            import io
            crop_bytes = io.BytesIO()
            crop_image.save(crop_bytes, format='JPEG', quality=95)
            crop_bytes = crop_bytes.getvalue()
            
            # Run OCR on crop
            ocr_mode = 'DOCUMENT_TEXT_DETECTION' if 'signage' in crop_info['type'] else 'TEXT_DETECTION'
            ocr_result = self.vision_client.detect_text(
                crop_bytes, 
                mode=ocr_mode,
                language_hints=config.OCR_LANGUAGE_HINTS
            )
            
            # Run logo detection on crop
            logo_result = self.vision_client.detect_logos(crop_bytes)
            
            # Extract tokens from OCR text
            ocr_tokens = []
            if ocr_result['text']:
                # Simple tokenization
                ocr_tokens = [token.strip() for token in ocr_result['text'].split() if token.strip()]
            
            # Create crop context
            context = CropContext(
                image_id=image_id,
                crop_id=crop_info['crop_id'],
                bbox=crop_info['bbox_normalized'],
                type=crop_info['type'],
                ocr_raw=ocr_result['text'],
                ocr_tokens=ocr_tokens,
                logo_candidates=[
                    {'brand': logo['description'], 'score': logo['confidence']}
                    for logo in logo_result
                ],
                scores={
                    'ocr_conf': ocr_result['confidence'],
                    'logo_conf': logo_result[0]['confidence'] if logo_result else 0.0,
                    'det_conf': crop_info['detector_confidence']
                }
            )
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to process crop {crop_info.get('crop_id', 'unknown')}: {e}")
            return None

# Global instances
vision_client = VisionClient()
crop_processor = CropProcessor(vision_client)