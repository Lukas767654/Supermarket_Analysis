"""
Image processing utilities for the retail audit pipeline.
Handles loading, quality assessment, HEIC conversion, and fingerprinting.
"""
import os
import io
import logging
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import pillow_heif
import imagehash
from skimage.metrics import structural_similarity as ssim
import cv2

from .config import config
from .models import ImageMetadata

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Main class for image processing operations."""
    
    def __init__(self):
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.heic', '.heif'}
    
    def load_images(self, assets_path: str) -> List[str]:
        """
        Load all supported image files from assets directory.
        
        Args:
            assets_path: Path to assets directory
            
        Returns:
            List of image file paths
        """
        image_paths = []
        assets_dir = Path(assets_path)
        
        if not assets_dir.exists():
            logger.error(f"Assets directory not found: {assets_path}")
            return []
        
        for file_path in assets_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                image_paths.append(str(file_path))
        
        logger.info(f"Found {len(image_paths)} supported image files")
        return sorted(image_paths)
    
    def load_image_as_pil(self, image_path: str) -> Optional[Image.Image]:
        """
        Load image as PIL Image, handling HEIC conversion.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image or None if failed
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                return img.copy()
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            return None
    
    def load_image_as_bytes(self, image_path: str) -> Optional[bytes]:
        """
        Load image as bytes for API calls.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image bytes or None if failed
        """
        try:
            # For HEIC files, convert to JPEG first
            if Path(image_path).suffix.lower() in {'.heic', '.heif'}:
                pil_image = self.load_image_as_pil(image_path)
                if pil_image is None:
                    return None
                
                byte_io = io.BytesIO()
                pil_image.save(byte_io, format='JPEG', quality=95)
                return byte_io.getvalue()
            else:
                # Read directly for JPG/PNG
                with open(image_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Failed to load image bytes {image_path}: {e}")
            return None
    
    def compute_quality_metrics(self, image: Image.Image) -> Dict[str, float]:
        """
        Compute quality metrics for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Dictionary with quality metrics
        """
        try:
            # Convert to grayscale for sharpness calculation
            gray = image.convert('L')
            gray_array = np.array(gray)
            
            # Sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray_array, cv2.CV_64F).var()
            
            # Brightness statistics
            brightness_mean = np.mean(gray_array)
            brightness_std = np.std(gray_array)
            
            # Overall quality score (normalized)
            quality_score = min(1.0, (laplacian_var / 1000.0) * (brightness_std / 64.0))
            
            return {
                'sharpness': float(laplacian_var),
                'brightness_mean': float(brightness_mean),
                'brightness_std': float(brightness_std),
                'quality_score': float(quality_score)
            }
        except Exception as e:
            logger.error(f"Failed to compute quality metrics: {e}")
            return {
                'sharpness': 0.0,
                'brightness_mean': 128.0,
                'brightness_std': 32.0,
                'quality_score': 0.5
            }
    
    def compute_phash(self, image: Image.Image) -> str:
        """
        Compute perceptual hash of image.
        
        Args:
            image: PIL Image
            
        Returns:
            Hexadecimal string representation of pHash
        """
        try:
            phash = imagehash.phash(image)
            return str(phash)
        except Exception as e:
            logger.error(f"Failed to compute pHash: {e}")
            return "0" * 16
    
    def compute_ssim(self, image1: Image.Image, image2: Image.Image, 
                     target_size: Tuple[int, int] = (256, 256)) -> float:
        """
        Compute SSIM between two images.
        
        Args:
            image1: First PIL Image
            image2: Second PIL Image
            target_size: Size to resize images for comparison
            
        Returns:
            SSIM score between 0 and 1
        """
        try:
            # Resize images to same size
            img1_resized = image1.resize(target_size, Image.Resampling.LANCZOS)
            img2_resized = image2.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert to grayscale arrays
            gray1 = np.array(img1_resized.convert('L'))
            gray2 = np.array(img2_resized.convert('L'))
            
            # Compute SSIM
            ssim_score = ssim(gray1, gray2, data_range=255)
            return float(ssim_score)
        except Exception as e:
            logger.error(f"Failed to compute SSIM: {e}")
            return 0.0
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Compute Hamming distance between two hashes.
        
        Args:
            hash1: First hash string
            hash2: Second hash string
            
        Returns:
            Hamming distance
        """
        try:
            if len(hash1) != len(hash2):
                return len(hash1)  # Maximum distance if different lengths
            
            return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        except Exception:
            return len(hash1) if hash1 else 0
    
    def create_thumbnail(self, image: Image.Image, size: Tuple[int, int] = None) -> Image.Image:
        """
        Create thumbnail of image.
        
        Args:
            image: PIL Image
            size: Thumbnail size (width, height)
            
        Returns:
            Thumbnail PIL Image
        """
        if size is None:
            size = config.THUMBNAIL_SIZE
        
        try:
            thumbnail = image.copy()
            thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
            return thumbnail
        except Exception as e:
            logger.error(f"Failed to create thumbnail: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', size, color='white')
    
    def save_image(self, image: Image.Image, file_path: str, quality: int = 95) -> bool:
        """
        Save PIL image to file.
        
        Args:
            image: PIL Image to save
            file_path: Output file path
            quality: JPEG quality (for JPEG files)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                image.save(file_path, 'JPEG', quality=quality, optimize=True)
            else:
                image.save(file_path, quality=quality, optimize=True)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save image {file_path}: {e}")
            return False
    
    def process_image_metadata(self, image_path: str, image_id: str) -> Optional[ImageMetadata]:
        """
        Process image and extract all metadata.
        
        Args:
            image_path: Path to image file
            image_id: Unique identifier for image
            
        Returns:
            ImageMetadata object or None if failed
        """
        try:
            # Load image
            pil_image = self.load_image_as_pil(image_path)
            if pil_image is None:
                return None
            
            # Get file stats
            file_stat = os.stat(image_path)
            file_name = os.path.basename(image_path)
            
            # Compute quality metrics
            quality_metrics = self.compute_quality_metrics(pil_image)
            
            # Compute fingerprints
            phash = self.compute_phash(pil_image)
            
            metadata = ImageMetadata(
                image_id=image_id,
                file_path=image_path,
                file_name=file_name,
                file_size=file_stat.st_size,
                dimensions=pil_image.size,
                quality_score=quality_metrics['quality_score'],
                sharpness=quality_metrics['sharpness'],
                brightness_mean=quality_metrics['brightness_mean'],
                phash=phash
            )
            
            logger.debug(f"Processed metadata for {file_name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to process image metadata {image_path}: {e}")
            return None

# Global instance
image_processor = ImageProcessor()