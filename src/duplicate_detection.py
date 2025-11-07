"""
Duplicate detection system using pHash, SSIM, OCR Jaccard similarity, 
and Union-Find clustering for grouping near-duplicate images.
"""
import logging
from typing import List, Dict, Set, Tuple, Optional, Any
import hashlib
from collections import defaultdict

from .config import config
from .models import ImageMetadata, ConsolidatedResult
from .image_utils import image_processor
from .text_utils import text_normalizer

logger = logging.getLogger(__name__)

class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for clustering."""
    
    def __init__(self, elements: List[str]):
        """
        Initialize Union-Find structure.
        
        Args:
            elements: List of element identifiers
        """
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}
        self.groups = {elem: {elem} for elem in elements}
    
    def find(self, x: str) -> str:
        """Find root of element with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: str, y: str) -> bool:
        """
        Union two elements by rank.
        
        Returns:
            True if union was performed, False if already in same set
        """
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x
        
        self.parent[root_y] = root_x
        self.groups[root_x].update(self.groups[root_y])
        del self.groups[root_y]
        
        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1
        
        return True
    
    def get_groups(self) -> List[Set[str]]:
        """Get all connected components."""
        return [group for group in self.groups.values() if len(group) > 0]

class DuplicateDetector:
    """Main duplicate detection system."""
    
    def __init__(self):
        self.thresholds = {
            'phash_max_dist': config.PHASH_MAX_DIST,
            'ssim_min': config.SSIM_MIN,
            'jaccard_min': config.JACCARD_MIN,
            'embedding_min': config.EMBEDDING_MIN
        }
    
    def compare_images(self, img1_meta: ImageMetadata, img2_meta: ImageMetadata,
                      img1_results: List[ConsolidatedResult] = None,
                      img2_results: List[ConsolidatedResult] = None) -> Dict[str, float]:
        """
        Compare two images using multiple similarity metrics.
        
        Args:
            img1_meta: First image metadata
            img2_meta: Second image metadata
            img1_results: OCR results for first image
            img2_results: OCR results for second image
            
        Returns:
            Dictionary with similarity scores
        """
        similarities = {
            'phash_similarity': 0.0,
            'ssim_similarity': 0.0,
            'ocr_jaccard': 0.0,
            'embedding_similarity': 0.0  # Placeholder for future implementation
        }
        
        try:
            # 1. pHash comparison
            if img1_meta.phash and img2_meta.phash:
                hamming_dist = image_processor.hamming_distance(img1_meta.phash, img2_meta.phash)
                # Convert to similarity (lower hamming distance = higher similarity)
                max_hamming = len(img1_meta.phash)
                similarities['phash_similarity'] = 1.0 - (hamming_dist / max_hamming)
            
            # 2. SSIM comparison (requires loading images)
            try:
                img1_pil = image_processor.load_image_as_pil(img1_meta.file_path)
                img2_pil = image_processor.load_image_as_pil(img2_meta.file_path)
                
                if img1_pil and img2_pil:
                    ssim_score = image_processor.compute_ssim(img1_pil, img2_pil)
                    similarities['ssim_similarity'] = ssim_score
            except Exception as e:
                logger.debug(f"SSIM comparison failed: {e}")
            
            # 3. OCR Jaccard similarity
            if img1_results and img2_results:
                # Combine all OCR tokens from all crops in each image
                img1_tokens = []
                img2_tokens = []
                
                for result in img1_results:
                    if result and result.product_en:
                        img1_tokens.extend(text_normalizer.tokenize_text(result.product_en))
                    if result and result.brand_en:
                        img1_tokens.extend(text_normalizer.tokenize_text(result.brand_en))
                
                for result in img2_results:
                    if result and result.product_en:
                        img2_tokens.extend(text_normalizer.tokenize_text(result.product_en))
                    if result and result.brand_en:
                        img2_tokens.extend(text_normalizer.tokenize_text(result.brand_en))
                
                if img1_tokens or img2_tokens:
                    jaccard = text_normalizer.calculate_jaccard_similarity(img1_tokens, img2_tokens)
                    similarities['ocr_jaccard'] = jaccard
            
            # 4. Embedding similarity (placeholder - set to 0 as specified)
            similarities['embedding_similarity'] = 0.0
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
        
        return similarities
    
    def is_duplicate_pair(self, similarities: Dict[str, float]) -> Tuple[bool, int]:
        """
        Determine if two images are duplicates based on similarity scores.
        
        Args:
            similarities: Dictionary with similarity metrics
            
        Returns:
            Tuple of (is_duplicate, num_criteria_met)
        """
        criteria_met = 0
        
        # Criterion 1: pHash Hamming distance ≤ 6 (similarity ≥ threshold)
        phash_threshold = 1.0 - (self.thresholds['phash_max_dist'] / 16.0)  # Assuming 16-char hash
        if similarities['phash_similarity'] >= phash_threshold:
            criteria_met += 1
        
        # Criterion 2: SSIM ≥ 0.92
        if similarities['ssim_similarity'] >= self.thresholds['ssim_min']:
            criteria_met += 1
        
        # Criterion 3: OCR Jaccard ≥ 0.80
        if similarities['ocr_jaccard'] >= self.thresholds['jaccard_min']:
            criteria_met += 1
        
        # Criterion 4: Embedding cosine ≥ 0.98 (always 0 for now)
        if similarities['embedding_similarity'] >= self.thresholds['embedding_min']:
            criteria_met += 1
        
        # Mark as duplicate if ≥2 conditions hold
        is_duplicate = criteria_met >= 2
        
        return is_duplicate, criteria_met
    
    def detect_duplicates(self, image_metadata: List[ImageMetadata],
                         consolidated_results: Dict[str, List[ConsolidatedResult]] = None) -> Dict[str, Any]:
        """
        Detect duplicate images and group them using Union-Find.
        
        Args:
            image_metadata: List of image metadata objects
            consolidated_results: Dictionary mapping image_id to list of results
            
        Returns:
            Dictionary with duplicate detection results
        """
        if len(image_metadata) < 2:
            logger.info("Less than 2 images, no duplicates to detect")
            return self._create_empty_duplicate_results(image_metadata)
        
        logger.info(f"Detecting duplicates among {len(image_metadata)} images")
        
        # Initialize Union-Find
        image_ids = [img.image_id for img in image_metadata]
        uf = UnionFind(image_ids)
        
        # Create lookup dictionaries
        metadata_dict = {img.image_id: img for img in image_metadata}
        results_dict = consolidated_results or {}
        
        # Compare all pairs
        duplicate_pairs = []
        total_comparisons = len(image_metadata) * (len(image_metadata) - 1) // 2
        comparison_count = 0
        
        for i in range(len(image_metadata)):
            for j in range(i + 1, len(image_metadata)):
                comparison_count += 1
                if comparison_count % 100 == 0:
                    logger.info(f"Comparison progress: {comparison_count}/{total_comparisons}")
                
                img1 = image_metadata[i]
                img2 = image_metadata[j]
                
                # Get consolidated results for OCR comparison
                img1_results = results_dict.get(img1.image_id, [])
                img2_results = results_dict.get(img2.image_id, [])
                
                # Compare images
                similarities = self.compare_images(img1, img2, img1_results, img2_results)
                is_dup, criteria_met = self.is_duplicate_pair(similarities)
                
                if is_dup:
                    # Union the images
                    uf.union(img1.image_id, img2.image_id)
                    duplicate_pairs.append({
                        'image1_id': img1.image_id,
                        'image2_id': img2.image_id,
                        'similarities': similarities,
                        'criteria_met': criteria_met
                    })
                    
                    logger.debug(f"Duplicate pair found: {img1.image_id} <-> {img2.image_id} "
                                f"(criteria met: {criteria_met})")
        
        # Get duplicate groups
        groups = uf.get_groups()
        duplicate_groups = [group for group in groups if len(group) > 1]
        
        logger.info(f"Found {len(duplicate_groups)} duplicate groups with {len(duplicate_pairs)} pairs")
        
        # Assign group IDs and canonical images
        group_assignments = {}
        canonical_assignments = {}
        
        for group_idx, group in enumerate(duplicate_groups):
            group_id = f"dup_group_{group_idx:03d}"
            
            # Choose canonical image (highest quality score, most crops, highest mean confidence)
            canonical_image_id = self._choose_canonical_image(group, metadata_dict, results_dict)
            
            for image_id in group:
                group_assignments[image_id] = group_id
                canonical_assignments[image_id] = canonical_image_id
        
        # Create results
        duplicate_results = {
            'total_images': len(image_metadata),
            'duplicate_groups': len(duplicate_groups),
            'duplicate_images': sum(len(group) for group in duplicate_groups),
            'canonical_images': len(image_metadata) - sum(len(group) - 1 for group in duplicate_groups),
            'duplicate_pairs': duplicate_pairs,
            'group_assignments': group_assignments,
            'canonical_assignments': canonical_assignments,
            'groups': duplicate_groups
        }
        
        return duplicate_results
    
    def _choose_canonical_image(self, group: Set[str], 
                              metadata_dict: Dict[str, ImageMetadata],
                              results_dict: Dict[str, List[ConsolidatedResult]]) -> str:
        """
        Choose the canonical image from a duplicate group.
        
        Args:
            group: Set of image IDs in the group
            metadata_dict: Dictionary mapping image_id to metadata
            results_dict: Dictionary mapping image_id to results
            
        Returns:
            Image ID of the canonical image
        """
        def score_image(image_id: str) -> Tuple[float, int, float]:
            """Score an image for canonical selection."""
            metadata = metadata_dict[image_id]
            results = results_dict.get(image_id, [])
            
            quality_score = metadata.quality_score
            crop_count = len(results)
            
            # Calculate mean confidence
            confidences = [r.conf_overall for r in results if r is not None]
            mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return (quality_score, crop_count, mean_confidence)
        
        # Sort by scoring criteria (descending order)
        sorted_images = sorted(group, key=score_image, reverse=True)
        canonical = sorted_images[0]
        
        logger.debug(f"Chose canonical image {canonical} from group {group}")
        return canonical
    
    def _create_empty_duplicate_results(self, image_metadata: List[ImageMetadata]) -> Dict[str, Any]:
        """Create empty duplicate results when no duplicates found."""
        return {
            'total_images': len(image_metadata),
            'duplicate_groups': 0,
            'duplicate_images': 0,
            'canonical_images': len(image_metadata),
            'duplicate_pairs': [],
            'group_assignments': {},
            'canonical_assignments': {},
            'groups': []
        }
    
    def update_metadata_with_duplicates(self, image_metadata: List[ImageMetadata],
                                      duplicate_results: Dict[str, Any]) -> List[ImageMetadata]:
        """
        Update image metadata with duplicate information.
        
        Args:
            image_metadata: List of image metadata
            duplicate_results: Results from duplicate detection
            
        Returns:
            Updated list of image metadata
        """
        group_assignments = duplicate_results['group_assignments']
        canonical_assignments = duplicate_results['canonical_assignments']
        
        updated_metadata = []
        
        for metadata in image_metadata:
            # Create a copy to avoid modifying original
            updated = ImageMetadata(**metadata.dict())
            
            # Update duplicate information
            if metadata.image_id in group_assignments:
                updated.duplicate_group_id = group_assignments[metadata.image_id]
                updated.is_duplicate = True
                updated.canonical_image_id = canonical_assignments[metadata.image_id]
            else:
                updated.duplicate_group_id = None
                updated.is_duplicate = False
                updated.canonical_image_id = metadata.image_id  # Self-canonical
            
            updated_metadata.append(updated)
        
        return updated_metadata

# Global instance
duplicate_detector = DuplicateDetector()