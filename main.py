"""
Main processing pipeline for the retail audit system.
Orchestrates all components with comprehensive error handling and logging.
"""
import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from tqdm import tqdm

from src.config import config
from src.models import ImageMetadata, ConsolidatedResult, AuditRecord
from src.image_utils import image_processor
from src.vision_api import vision_client, crop_processor
from src.text_utils import text_normalizer
from src.gemini_consolidation import gemini_consolidator
from src.duplicate_detection import duplicate_detector
from src.excel_export import excel_generator

# Configure logging
def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('retail_audit.log', mode='w')
        ]
    )

logger = logging.getLogger(__name__)

class RetailAuditPipeline:
    """Main retail audit processing pipeline."""
    
    def __init__(self, assets_path: str, output_path: str, max_images: int = 0):
        """
        Initialize the pipeline.
        
        Args:
            assets_path: Path to input images
            output_path: Path for outputs
            max_images: Maximum number of images to process (0 = no limit)
        """
        self.assets_path = assets_path
        self.output_path = output_path
        self.max_images = max_images
        
        # Ensure output directories exist
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/json", exist_ok=True)
        os.makedirs(f"{output_path}/crops", exist_ok=True)
        os.makedirs(f"{output_path}/thumbs", exist_ok=True)
        os.makedirs(f"{output_path}/excel", exist_ok=True)
        
        # Pipeline state
        self.image_metadata: List[ImageMetadata] = []
        self.consolidated_results: Dict[str, List[ConsolidatedResult]] = {}
        self.duplicate_results: Dict[str, Any] = {}
    
    def run_complete_pipeline(self, strict: bool = False, no_gemini: bool = False) -> bool:
        """
        Run the complete retail audit pipeline.
        
        Args:
            strict: If True, fail on any JSON parse errors
            no_gemini: If True, skip Gemini consolidation (Vision-only dry run)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting retail audit pipeline")
            start_time = time.time()
            
            # Step 0: Image ingestion and preprocessing
            if not self._step_0_ingestion():
                return False
            
            # Step 1: Object localization and crop creation
            if not self._step_1_object_localization():
                return False
            
            # Step 2: OCR and logo detection per crop
            if not self._step_2_ocr_and_logos():
                return False
            
            # Step 3: Gemini consolidation (optional)
            if not no_gemini:
                if not self._step_3_gemini_consolidation(strict):
                    return False
            else:
                logger.info("Skipping Gemini consolidation (no_gemini=True)")
                # Create dummy results for testing
                self._create_dummy_consolidated_results()
            
            # Step 4: Duplicate detection
            if not self._step_4_duplicate_detection():
                return False
            
            # Step 5: Excel export
            if not self._step_5_excel_export():
                return False
            
            elapsed_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {elapsed_time:.1f} seconds")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}", exc_info=True)
            return False
    
    def _step_0_ingestion(self) -> bool:
        """Step 0: Load and preprocess images."""
        logger.info("Step 0: Image ingestion and preprocessing")
        
        try:
            # Load image paths
            image_paths = image_processor.load_images(self.assets_path)
            
            if not image_paths:
                logger.error("No images found in assets directory")
                return False
            
            # Apply max_images limit if specified
            if self.max_images > 0:
                image_paths = image_paths[:self.max_images]
                logger.info(f"Limited to {self.max_images} images")
            
            # Process each image
            self.image_metadata = []
            
            for i, image_path in enumerate(tqdm(image_paths, desc="Processing images")):
                image_id = f"img{i+1:04d}"
                
                try:
                    metadata = image_processor.process_image_metadata(image_path, image_id)
                    if metadata:
                        self.image_metadata.append(metadata)
                        
                        # Save metadata to JSON
                        metadata_file = f"{self.output_path}/json/{image_id}_metadata.json"
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata.dict(), f, indent=2, default=str)
                    else:
                        logger.warning(f"Failed to process metadata for {image_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing image {image_path}: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(self.image_metadata)} images")
            return len(self.image_metadata) > 0
            
        except Exception as e:
            logger.error(f"Step 0 failed: {e}")
            return False
    
    def _step_1_object_localization(self) -> bool:
        """Step 1: Object localization and crop creation."""
        logger.info("Step 1: Object localization and crop creation")
        
        try:
            total_crops = 0
            
            for metadata in tqdm(self.image_metadata, desc="Object detection"):
                try:
                    # Load image bytes for Vision API
                    image_bytes = image_processor.load_image_as_bytes(metadata.file_path)
                    if not image_bytes:
                        continue
                    
                    # Load PIL image for hybrid cropping
                    pil_image = image_processor.load_image_as_pil(metadata.file_path)
                    if not pil_image:
                        continue
                    
                    # Create hybrid crops (skip object detection step)
                    logger.info("Creating hybrid crops using Vision API labels + text + grid")
                    crops = crop_processor.create_hybrid_crops(pil_image, metadata.image_id)
                    
                    # Save crops and thumbnails
                    crop_count = 0
                    for crop_image, crop_info in crops:
                        crop_id = crop_info['crop_id']
                        
                        # Save crop
                        crop_path = f"{self.output_path}/crops/{crop_id}.jpg"
                        if image_processor.save_image(crop_image, crop_path):
                            crop_count += 1
                        
                        # Save thumbnail
                        thumbnail = image_processor.create_thumbnail(crop_image)
                        thumb_path = f"{self.output_path}/thumbs/{crop_id}_thumb.jpg"
                        image_processor.save_image(thumbnail, thumb_path)
                        
                        # Save crop info
                        crop_info_file = f"{self.output_path}/json/{crop_id}_info.json"
                        with open(crop_info_file, 'w') as f:
                            json.dump(crop_info, f, indent=2)
                    
                    # Update metadata with crop count
                    metadata.crop_count = crop_count
                    total_crops += crop_count
                    
                except Exception as e:
                    logger.error(f"Error in object localization for {metadata.image_id}: {e}")
                    continue
            
            logger.info(f"Created {total_crops} crops from {len(self.image_metadata)} images")
            return total_crops > 0
            
        except Exception as e:
            logger.error(f"Step 1 failed: {e}")
            return False
    
    def _step_2_ocr_and_logos(self) -> bool:
        """Step 2: OCR and logo detection per crop."""
        logger.info("Step 2: OCR and logo detection per crop")
        
        try:
            crop_contexts = []
            
            # Process each image's crops
            for metadata in tqdm(self.image_metadata, desc="OCR and logos"):
                try:
                    # Find crop files for this image
                    crop_pattern = f"{metadata.image_id}_c*"
                    crop_files = list(Path(f"{self.output_path}/crops").glob(f"{crop_pattern}.jpg"))
                    
                    if not crop_files:
                        continue
                    
                    # Process each crop
                    for crop_file in crop_files:
                        crop_id = crop_file.stem
                        
                        try:
                            # Load crop info
                            info_file = f"{self.output_path}/json/{crop_id}_info.json"
                            if not os.path.exists(info_file):
                                continue
                            
                            with open(info_file, 'r') as f:
                                crop_info = json.load(f)
                            
                            # Load crop image
                            crop_image = image_processor.load_image_as_pil(str(crop_file))
                            if not crop_image:
                                continue
                            
                            # Process crop with OCR and logo detection
                            context = crop_processor.process_crop(
                                crop_image, crop_info, metadata.image_id
                            )
                            
                            if context:
                                crop_contexts.append(context)
                                
                                # Save crop context
                                context_file = f"{self.output_path}/json/{crop_id}_context.json"
                                with open(context_file, 'w') as f:
                                    json.dump(context.dict(), f, indent=2, default=str)
                        
                        except Exception as e:
                            logger.error(f"Error processing crop {crop_id}: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error in OCR processing for {metadata.image_id}: {e}")
                    continue
            
            logger.info(f"Processed OCR and logos for {len(crop_contexts)} crops")
            return len(crop_contexts) > 0
            
        except Exception as e:
            logger.error(f"Step 2 failed: {e}")
            return False
    
    def _step_3_gemini_consolidation(self, strict: bool = False) -> bool:
        """Step 3: Gemini consolidation."""
        logger.info("Step 3: Gemini consolidation")
        
        try:
            # Load all crop contexts
            context_files = list(Path(f"{self.output_path}/json").glob("*_context.json"))
            contexts = []
            
            for context_file in context_files:
                try:
                    with open(context_file, 'r') as f:
                        context_data = json.load(f)
                    
                    from src.models import CropContext
                    context = CropContext(**context_data)
                    contexts.append(context)
                    
                except Exception as e:
                    logger.error(f"Error loading context {context_file}: {e}")
                    if strict:
                        return False
                    continue
            
            if not contexts:
                logger.error("No crop contexts found for consolidation")
                return False
            
            # Batch consolidation
            logger.info(f"Consolidating {len(contexts)} crops with Gemini")
            results = gemini_consolidator.batch_consolidate(contexts)
            
            # Group results by image
            self.consolidated_results = {}
            successful_consolidations = 0
            
            for context in contexts:
                result = results.get(context.crop_id)
                
                if result:
                    if context.image_id not in self.consolidated_results:
                        self.consolidated_results[context.image_id] = []
                    self.consolidated_results[context.image_id].append(result)
                    successful_consolidations += 1
                    
                    # Save consolidated result
                    result_file = f"{self.output_path}/json/{context.crop_id}_consolidated.json"
                    with open(result_file, 'w') as f:
                        json.dump(result.dict(), f, indent=2, default=str)
                else:
                    logger.warning(f"Consolidation failed for crop {context.crop_id}")
                    if strict:
                        return False
            
            logger.info(f"Successfully consolidated {successful_consolidations}/{len(contexts)} crops")
            return successful_consolidations > 0
            
        except Exception as e:
            logger.error(f"Step 3 failed: {e}")
            return False
    
    def _step_4_duplicate_detection(self) -> bool:
        """Step 4: Duplicate detection."""
        logger.info("Step 4: Duplicate detection")
        
        try:
            # Detect duplicates
            self.duplicate_results = duplicate_detector.detect_duplicates(
                self.image_metadata, self.consolidated_results
            )
            
            # Update metadata with duplicate information
            self.image_metadata = duplicate_detector.update_metadata_with_duplicates(
                self.image_metadata, self.duplicate_results
            )
            
            # Save duplicate results
            duplicate_file = f"{self.output_path}/json/duplicate_results.json"
            with open(duplicate_file, 'w') as f:
                json.dump(self.duplicate_results, f, indent=2, default=str)
            
            logger.info(f"Detected {self.duplicate_results['duplicate_groups']} duplicate groups")
            return True
            
        except Exception as e:
            logger.error(f"Step 4 failed: {e}")
            return False
    
    def _step_5_excel_export(self) -> bool:
        """Step 5: Excel export."""
        logger.info("Step 5: Excel export")
        
        try:
            # Create audit records
            all_records = excel_generator.create_audit_records(
                self.image_metadata, self.consolidated_results, self.duplicate_results
            )
            
            # Filter canonical records (non-duplicates)
            canonical_records = [r for r in all_records if not r.is_duplicate]
            
            # Create category summaries
            category_summaries = excel_generator.create_category_summaries(canonical_records)
            
            # Create Excel report
            excel_path = f"{self.output_path}/excel/audit.xlsx"
            success = excel_generator.create_excel_report(
                all_records, canonical_records, category_summaries, 
                self.duplicate_results, excel_path
            )
            
            if success:
                logger.info(f"Excel report created: {excel_path}")
                return True
            else:
                logger.error("Failed to create Excel report")
                return False
            
        except Exception as e:
            logger.error(f"Step 5 failed: {e}")
            return False
    
    def _create_dummy_consolidated_results(self):
        """Create dummy consolidated results for testing without Gemini."""
        self.consolidated_results = {}
        
        for metadata in self.image_metadata:
            # Create dummy results based on available crop contexts
            context_files = list(Path(f"{self.output_path}/json").glob(f"{metadata.image_id}_c*_context.json"))
            
            results = []
            for context_file in context_files:
                crop_id = context_file.stem.replace('_context', '')
                
                # Create minimal consolidated result
                result = ConsolidatedResult(
                    image_id=metadata.image_id,
                    crop_id=crop_id,
                    brand_en="Test Brand",
                    product_en="Test Product",
                    category_en="Test Category",
                    conf_overall=0.8,
                    review_needed=False
                )
                results.append(result)
            
            if results:
                self.consolidated_results[metadata.image_id] = results

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Retail Audit Pipeline")
    parser.add_argument("--assets", default="./assets", help="Path to assets directory")
    parser.add_argument("--out", default="./outputs", help="Path to outputs directory") 
    parser.add_argument("--max_images", type=int, default=0, help="Maximum images to process (0=no limit)")
    parser.add_argument("--strict", action="store_true", help="Fail on JSON parse errors")
    parser.add_argument("--no_gemini", action="store_true", help="Skip Gemini consolidation (Vision-only dry run)")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate environment
    if not config.GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set in environment")
        return 1
    
    # Create and run pipeline
    pipeline = RetailAuditPipeline(args.assets, args.out, args.max_images)
    
    success = pipeline.run_complete_pipeline(
        strict=args.strict,
        no_gemini=args.no_gemini
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())