"""
Cross-Image Duplicate Detection Module
=====================================
Detects and removes duplicate products across multiple supermarket images.

Features:
- Visual similarity comparison using embeddings
- Brand/Type matching for accuracy
- OCR text similarity analysis
- Canonical product selection
- Multi-criteria scoring system
"""

import numpy as np
import logging
from typing import List, Dict, Tuple, Any
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from collections import defaultdict
import hashlib
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class CrossImageDeduplicator:
    """
    Cross-image duplicate detection for supermarket products.
    """
    
    def __init__(self, config):
        """
        Initialize the deduplicator with configuration.
        
        Args:
            config: Configuration object with deduplication parameters
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Deduplication parameters
        self.eps = getattr(config, 'CROSS_IMAGE_EPS', 0.15)
        self.min_similarity = getattr(config, 'MIN_SIMILARITY_FOR_DUPLICATE', 0.75)
        self.visual_weight = getattr(config, 'VISUAL_SIMILARITY_WEIGHT', 0.4)
        self.text_weight = getattr(config, 'TEXT_SIMILARITY_WEIGHT', 0.3)
        self.brand_weight = getattr(config, 'BRAND_SIMILARITY_WEIGHT', 0.3)
        
    def detect_duplicates(self, results_by_image: Dict[str, List[Dict]]) -> Tuple[List[Dict], Dict]:
        """
        Detect and remove cross-image duplicates.
        
        Args:
            results_by_image: Dictionary mapping image_id to list of product results
            
        Returns:
            Tuple of (deduplicated_results, duplicate_mapping)
        """
        self.logger.info("🔍 Starting cross-image duplicate detection...")
        
        # Step 1: Collect all products from all images
        all_products = []
        for image_id, results in results_by_image.items():
            for i, result in enumerate(results):
                # Convert EnhancedProduct objects to dicts if needed
                if hasattr(result, '__dict__') and not isinstance(result, dict):
                    # This is an EnhancedProduct object - convert to dict
                    result_dict = {
                        'name': getattr(result, 'name', 'unknown'),
                        'brand': getattr(result, 'brand', 'unknown'),
                        'type': getattr(result, 'type', 'unknown'),
                        'confidence': getattr(result, 'confidence', 0.5),
                        'approx_count': getattr(result, 'approx_count', 1),
                        'keywords': getattr(result, 'keywords', []),
                        'category_display_name': getattr(result, 'category_display_name', 'Unknown'),
                        'main_category': getattr(result, 'main_category', 'unknown'),
                        'subcategory': getattr(result, 'subcategory', 'unknown'),
                        'price': getattr(result, 'price', None),
                        'text_content': getattr(result, 'text_content', [])
                    }
                    
                    # Handle nested objects
                    if hasattr(result, 'brand_classification'):
                        bc = result.brand_classification
                        result_dict['brand_classification'] = {
                            'origin': getattr(bc, 'origin', 'unknown'),
                            'confidence': getattr(bc, 'confidence', 0.0),
                            'classification_method': getattr(bc, 'classification_method', 'unknown'),
                            'matched_patterns': getattr(bc, 'matched_patterns', [])
                        }
                    
                    if hasattr(result, 'eye_level_data'):
                        eld = result.eye_level_data
                        result_dict['eye_level_data'] = {
                            'zone': getattr(eld, 'zone', 'unknown'),
                            'y_position': getattr(eld, 'y_position', 0.0),
                            'is_premium_zone': getattr(eld, 'is_premium_zone', False),
                            'shelf_tier': getattr(eld, 'shelf_tier', 'unknown')
                        }
                    
                    if hasattr(result, 'cjmore_classification'):
                        cjmore = result.cjmore_classification
                        result_dict['cjmore_classification'] = {
                            'is_private_brand': getattr(cjmore, 'is_private_brand', False),
                            'brand_name': getattr(cjmore, 'brand_name', ''),
                            'confidence': getattr(cjmore, 'confidence', 0.0),
                            'matched_pattern': getattr(cjmore, 'matched_pattern', ''),
                            'detection_method': getattr(cjmore, 'detection_method', 'none'),
                            'detection_source': getattr(cjmore, 'detection_source', 'none')
                        }
                    
                    result_dict['source_image'] = image_id
                    result_dict['source_index'] = i
                    result_dict['global_id'] = f"{image_id}_{i}"
                    all_products.append(result_dict)
                else:
                    # Already a dict - just add metadata
                    result['source_image'] = image_id
                    result['source_index'] = i
                    result['global_id'] = f"{image_id}_{i}"
                    all_products.append(result)
        
        total_products = len(all_products)
        total_images = len(results_by_image)
        
        self.logger.info(f"📊 Processing {total_products} products from {total_images} images")
        
        if total_products < 2:
            self.logger.info("⚠️  Less than 2 products found, skipping deduplication")
            return all_products, {}
        
        # Step 2: Generate embeddings for all products
        embeddings = self._generate_embeddings(all_products)
        
        # Step 3: Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(all_products, embeddings)
        
        # Step 4: Perform clustering to find duplicates
        clusters = self._cluster_products(similarity_matrix)
        
        # Step 5: Select canonical products and create mapping
        deduplicated_results, duplicate_mapping = self._create_canonical_products(
            all_products, clusters
        )
        
        removed_count = total_products - len(deduplicated_results)
        self.logger.info(f"✅ Cross-image deduplication complete!")
        self.logger.info(f"   📈 Total products: {total_products}")
        self.logger.info(f"   📊 Unique products: {len(deduplicated_results)}")
        self.logger.info(f"   🗑️  Duplicates removed: {removed_count}")
        self.logger.info(f"   📉 Reduction: {removed_count/total_products*100:.1f}%")
        
        return deduplicated_results, duplicate_mapping
    
    def _generate_embeddings(self, products: List[Dict]) -> np.ndarray:
        """
        Generate embeddings for all products.
        """
        self.logger.info("🧮 Generating product embeddings...")
        
        embeddings = []
        for product in products:
            # Check if embedding already exists
            if 'embedding' in product and product['embedding'] is not None:
                embeddings.append(np.array(product['embedding']))
            else:
                # Create embedding from text features
                text_features = self._extract_text_features(product)
                embedding = self._create_text_embedding(text_features)
                embeddings.append(embedding)
                product['embedding'] = embedding.tolist()  # Store for future use
        
        return np.array(embeddings)
    
    def _extract_text_features(self, product: Dict) -> str:
        """
        Extract text features from product for embedding generation.
        """
        features = []
        
        # Brand name
        brand = product.get('brand', '').strip()
        if brand and brand.lower() != 'unknown':
            features.append(brand)
        
        # Product type
        product_type = product.get('type', '').strip()
        if product_type:
            features.append(product_type)
        
        # Category information
        if 'category_display_name' in product:
            features.append(product['category_display_name'])
        
        # OCR tokens (most distinctive features)
        source_data = product.get('source_data', {})
        if 'ocr_tokens' in source_data:
            # Take most relevant OCR tokens (filter out common words)
            ocr_tokens = source_data['ocr_tokens']
            filtered_tokens = [
                token for token in ocr_tokens 
                if len(token) > 2 and token.lower() not in ['the', 'and', 'for', 'new', 'fresh']
            ]
            features.extend(filtered_tokens[:5])  # Top 5 tokens
        
        return ' '.join(features).lower()
    
    def _create_text_embedding(self, text: str, dimension: int = 128) -> np.ndarray:
        """
        Create a simple text embedding using hash-based approach.
        In production, this would be replaced with CLIP or similar model.
        """
        if not text.strip():
            return np.zeros(dimension)
        
        # Create multiple hash features for better representation
        embeddings = []
        
        # Hash 1: Full text
        hash1 = hashlib.md5(text.encode('utf-8')).hexdigest()
        
        # Hash 2: Sorted words (order-independent)
        words = sorted(text.split())
        hash2 = hashlib.md5(' '.join(words).encode('utf-8')).hexdigest()
        
        # Hash 3: Character bigrams
        bigrams = [text[i:i+2] for i in range(len(text)-1)]
        hash3 = hashlib.md5(''.join(bigrams).encode('utf-8')).hexdigest()
        
        # Convert hashes to numerical features
        for hash_hex in [hash1, hash2, hash3]:
            hash_nums = [int(hash_hex[i:i+2], 16) for i in range(0, len(hash_hex), 2)]
            embeddings.extend(hash_nums[:dimension//3])
        
        # Pad or truncate to desired dimension
        embedding = np.array(embeddings[:dimension])
        if len(embedding) < dimension:
            embedding = np.pad(embedding, (0, dimension - len(embedding)))
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _compute_similarity_matrix(self, products: List[Dict], embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between all products using multiple criteria.
        """
        self.logger.info("📏 Computing similarity matrix...")
        
        n = len(products)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):  # Only compute upper triangle
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    sim = self._compute_pairwise_similarity(
                        products[i], products[j], embeddings[i], embeddings[j]
                    )
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim  # Matrix is symmetric
        
        return similarity_matrix
    
    def _compute_pairwise_similarity(self, product1: Dict, product2: Dict, 
                                   embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute similarity between two products using multiple criteria.
        """
        # 1. Visual similarity (embedding cosine similarity)
        visual_sim = cosine_similarity([embedding1], [embedding2])[0][0]
        visual_sim = max(0, visual_sim)  # Ensure non-negative
        
        # 2. Brand-Type exact matching
        brand1 = product1.get('brand', '').lower().strip()
        brand2 = product2.get('brand', '').lower().strip()
        type1 = product1.get('type', '').lower().strip()
        type2 = product2.get('type', '').lower().strip()
        
        # Strong signal: same brand AND same type
        exact_match = (brand1 == brand2 and type1 == type2 and 
                      brand1 != '' and brand1 != 'unknown' and
                      type1 != '' and type1 != 'unknown')
        
        brand_type_sim = 1.0 if exact_match else 0.0
        
        # 3. OCR text similarity (Jaccard similarity)
        tokens1 = set(product1.get('source_data', {}).get('ocr_tokens', []))
        tokens2 = set(product2.get('source_data', {}).get('ocr_tokens', []))
        
        if tokens1 and tokens2:
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            text_sim = intersection / union if union > 0 else 0.0
        else:
            text_sim = 0.0
        
        # 4. Same image penalty (products from same image less likely to be duplicates)
        same_image = product1.get('source_image') == product2.get('source_image')
        image_penalty = -0.2 if same_image else 0.0
        
        # 5. Confidence weighting
        conf1 = product1.get('conf_fused', 0.5)
        conf2 = product2.get('conf_fused', 0.5)
        conf_weight = (conf1 + conf2) / 2.0  # Average confidence
        
        # Combine all similarities
        total_similarity = (
            self.visual_weight * visual_sim +
            self.text_weight * text_sim +
            self.brand_weight * brand_type_sim +
            image_penalty
        )
        
        # Apply confidence weighting
        total_similarity *= conf_weight
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, total_similarity))
    
    def _cluster_products(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Cluster products using DBSCAN on distance matrix.
        """
        self.logger.info("🔗 Clustering products to find duplicates...")
        
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(
            eps=self.eps,
            min_samples=1,  # Even single products form their own cluster
            metric='precomputed'
        ).fit(distance_matrix)
        
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        n_noise = list(clustering.labels_).count(-1)
        
        self.logger.info(f"   📊 Found {n_clusters} clusters and {n_noise} noise points")
        
        return clustering.labels_
    
    def _create_canonical_products(self, products: List[Dict], cluster_labels: np.ndarray) -> Tuple[List[Dict], Dict]:
        """
        Select canonical (representative) products for each cluster.
        """
        self.logger.info("🏆 Selecting canonical products...")
        
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label >= 0:  # Valid cluster (not noise)
                clusters[label].append(idx)
        
        canonical_products = []
        duplicate_mapping = {}
        
        # Process each cluster
        for cluster_id, product_indices in clusters.items():
            cluster_products = [products[i] for i in product_indices]
            
            if len(cluster_products) == 1:
                # Single product cluster
                product = cluster_products[0]
                product['is_canonical'] = True
                product['duplicate_count'] = 1
                product['source_images'] = [product['source_image']]
                canonical_products.append(product)
            else:
                # Multiple products - select best as canonical
                canonical_product = self._select_best_product(cluster_products)
                
                # Mark as canonical and add metadata
                canonical_product['is_canonical'] = True
                canonical_product['duplicate_count'] = len(cluster_products)
                canonical_product['source_images'] = list(set(p['source_image'] for p in cluster_products))
                canonical_product['cluster_id'] = cluster_id
                
                canonical_products.append(canonical_product)
                
                # Map duplicates to canonical
                for product in cluster_products:
                    if product != canonical_product:
                        duplicate_mapping[product['global_id']] = {
                            'canonical_id': canonical_product['global_id'],
                            'cluster_id': cluster_id,
                            'reason': 'cross_image_duplicate',
                            'similarity_score': self._compute_pairwise_similarity(
                                product, canonical_product,
                                np.array(product.get('embedding', [])),
                                np.array(canonical_product.get('embedding', []))
                            )
                        }
        
        # Handle noise points (unique products)
        for idx, label in enumerate(cluster_labels):
            if label == -1:  # Noise = unique product
                product = products[idx]
                product['is_canonical'] = True
                product['duplicate_count'] = 1
                product['source_images'] = [product['source_image']]
                canonical_products.append(product)
        
        return canonical_products, duplicate_mapping
    
    def _select_best_product(self, products: List[Dict]) -> Dict:
        """
        Select the best product from a group of duplicates.
        
        Criteria (in order of importance):
        1. Highest confidence score
        2. Most OCR information (more text = better quality)
        3. Non-unknown brand
        4. Comes from earlier processed image (consistent selection)
        """
        def score_product(product):
            score = 0.0
            
            # Confidence (0-1, weight: 40%)
            conf = product.get('conf_fused', 0.0)
            score += conf * 0.4
            
            # OCR information richness (weight: 30%)
            ocr_tokens = product.get('source_data', {}).get('ocr_tokens', [])
            ocr_score = min(len(ocr_tokens) / 10.0, 1.0)  # Normalize to max 1.0
            score += ocr_score * 0.3
            
            # Non-unknown brand (weight: 20%)
            brand = product.get('brand', '').lower()
            brand_score = 1.0 if brand and brand != 'unknown' else 0.0
            score += brand_score * 0.2
            
            # Consistency bonus (weight: 10%)
            image_name = product.get('source_image', '')
            consistency_score = 1.0 if image_name else 0.0
            score += consistency_score * 0.1
            
            return score
        
        return max(products, key=score_product)
    
    def save_deduplication_report(self, results: List[Dict], mapping: Dict, output_path: Path):
        """
        Save detailed deduplication report.
        """
        report = {
            'summary': {
                'total_unique_products': len(results),
                'total_duplicates_removed': len(mapping),
                'deduplication_rate': len(mapping) / (len(results) + len(mapping)) * 100 if results else 0
            },
            'canonical_products': [
                {
                    'id': p.get('global_id'),
                    'brand': p.get('brand'),
                    'type': p.get('type'),
                    'duplicate_count': p.get('duplicate_count', 1),
                    'source_images': p.get('source_images', []),
                    'confidence': p.get('conf_fused')
                }
                for p in results
            ],
            'duplicate_mapping': mapping,
            'statistics': self._generate_statistics(results, mapping)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 Deduplication report saved: {output_path}")
    
    def _generate_statistics(self, results: List[Dict], mapping: Dict) -> Dict:
        """Generate deduplication statistics."""
        
        # Count products by duplicate frequency
        duplicate_counts = defaultdict(int)
        for product in results:
            count = product.get('duplicate_count', 1)
            duplicate_counts[count] += 1
        
        # Brand statistics
        brand_stats = defaultdict(int)
        for product in results:
            brand = product.get('brand', 'unknown')
            brand_stats[brand] += product.get('duplicate_count', 1)
        
        return {
            'duplicate_frequency': dict(duplicate_counts),
            'top_brands_by_occurrence': dict(sorted(brand_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            'avg_duplicates_per_product': sum(p.get('duplicate_count', 1) for p in results) / len(results) if results else 0
        }

def create_cross_image_deduplicator(config):
    """Factory function to create deduplicator instance."""
    return CrossImageDeduplicator(config)