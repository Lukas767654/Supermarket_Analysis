"""
🔄 Checkpoint & Recovery System for Supermarket Analysis
Ermöglicht das Fortsetzen unterbrochener Analysen bei großen Batch-Verarbeitungen.

Author: Enhanced Brand Analysis Pipeline
Created: 2024-10-28
"""

import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

from config_brand_analysis import (
    ENABLE_CHECKPOINTS, CHECKPOINT_FREQUENCY, CHECKPOINT_DIR,
    AUTO_RESUME, ENABLE_BACKUP, BACKUP_FREQUENCY, MAX_BACKUPS,
    ESTIMATE_REMAINING_TIME
)


@dataclass
class CheckpointMetadata:
    """Metadata für Checkpoint-Dateien."""
    timestamp: str
    images_processed: int
    total_images: int
    current_image: str
    processing_time: float
    estimated_remaining: float
    checkpoint_version: str = "1.0"


class CheckpointManager:
    """
    Verwaltet Checkpoints und Recovery für große Batch-Verarbeitungen.
    """
    
    def __init__(self, output_folder: Path):
        self.output_folder = Path(output_folder)
        self.checkpoint_dir = self.output_folder / CHECKPOINT_DIR
        self.logger = logging.getLogger(__name__)
        
        # Runtime statistics
        self.start_time = time.time()
        self.image_times: List[float] = []
        self.processed_images: List[str] = []
        
        # Initialize checkpoint directory
        if ENABLE_CHECKPOINTS:
            self.checkpoint_dir.mkdir(exist_ok=True)
            self.logger.info(f"📄 Checkpoint System aktiviert: {self.checkpoint_dir}")
    
    def get_checkpoint_file(self) -> Path:
        """Erstelle Pfad für Checkpoint-Datei."""
        return self.checkpoint_dir / "analysis_checkpoint.json"
    
    def get_backup_file(self, backup_number: int) -> Path:
        """Erstelle Pfad für Backup-Datei."""
        return self.checkpoint_dir / f"backup_{backup_number:03d}_checkpoint.json"
    
    def should_create_checkpoint(self, image_count: int) -> bool:
        """Prüfe ob ein Checkpoint erstellt werden soll."""
        if not ENABLE_CHECKPOINTS:
            return False
        return image_count % CHECKPOINT_FREQUENCY == 0
    
    def should_create_backup(self, image_count: int) -> bool:
        """Prüfe ob ein Backup erstellt werden soll."""
        if not ENABLE_BACKUP:
            return False
        return image_count % BACKUP_FREQUENCY == 0
    
    def save_checkpoint(self, 
                       processed_images: List[str],
                       results_by_image: Dict[str, Any],
                       current_image: Optional[str] = None) -> bool:
        """
        Speichere aktuellen Analysezustand als Checkpoint.
        
        Args:
            processed_images: Liste bereits verarbeiteter Bilder
            results_by_image: Ergebnisse gruppiert nach Bildern
            current_image: Aktuell verarbeitetes Bild
            
        Returns:
            True wenn erfolgreich gespeichert
        """
        if not ENABLE_CHECKPOINTS:
            return False
        
        try:
            checkpoint_file = self.get_checkpoint_file()
            
            # Calculate processing statistics
            processing_time = time.time() - self.start_time
            images_processed = len(processed_images)
            
            # Estimate remaining time
            estimated_remaining = 0.0
            if ESTIMATE_REMAINING_TIME and self.image_times:
                avg_time_per_image = sum(self.image_times) / len(self.image_times)
                remaining_images = len(self.processed_images) - images_processed
                estimated_remaining = avg_time_per_image * remaining_images
            
            # Create checkpoint metadata
            metadata = CheckpointMetadata(
                timestamp=datetime.now().isoformat(),
                images_processed=images_processed,
                total_images=len(self.processed_images) if self.processed_images else images_processed,
                current_image=current_image or "unknown",
                processing_time=processing_time,
                estimated_remaining=estimated_remaining
            )
            
            # Serialize results for checkpoint (convert objects to dicts if needed)
            serializable_results = {}
            for image_id, results in results_by_image.items():
                serializable_results[image_id] = []
                for result in results:
                    if hasattr(result, '__dict__'):
                        # Convert dataclass/object to dict
                        serializable_results[image_id].append(self._serialize_enhanced_product(result))
                    else:
                        # Already a dict
                        serializable_results[image_id].append(result)
            
            # Create checkpoint data
            checkpoint_data = {
                'metadata': asdict(metadata),
                'processed_images': processed_images,
                'results_by_image': serializable_results,
                'runtime_stats': {
                    'image_processing_times': self.image_times,
                    'average_time_per_image': sum(self.image_times) / len(self.image_times) if self.image_times else 0.0,
                    'total_processing_time': processing_time
                }
            }
            
            # Save checkpoint
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"💾 Checkpoint gespeichert: {images_processed} Bilder verarbeitet")
            
            if ESTIMATE_REMAINING_TIME and estimated_remaining > 0:
                remaining_str = str(timedelta(seconds=int(estimated_remaining)))
                self.logger.info(f"⏱️  Geschätzte verbleibende Zeit: {remaining_str}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint speichern fehlgeschlagen: {e}")
            return False
    
    def load_checkpoint(self) -> Optional[Tuple[List[str], Dict[str, Any]]]:
        """
        Lade letzten Checkpoint wenn verfügbar.
        
        Returns:
            Tuple von (processed_images, results_by_image) oder None
        """
        if not ENABLE_CHECKPOINTS or not AUTO_RESUME:
            return None
        
        checkpoint_file = self.get_checkpoint_file()
        
        if not checkpoint_file.exists():
            self.logger.info("📄 Kein Checkpoint gefunden - starte neue Analyse")
            return None
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            metadata = checkpoint_data['metadata']
            processed_images = checkpoint_data['processed_images']
            results_by_image = checkpoint_data['results_by_image']
            
            # Restore runtime statistics
            if 'runtime_stats' in checkpoint_data:
                stats = checkpoint_data['runtime_stats']
                self.image_times = stats.get('image_processing_times', [])
            
            self.logger.info(f"🔄 Checkpoint geladen: {len(processed_images)} Bilder bereits verarbeitet")
            self.logger.info(f"📅 Checkpoint vom: {metadata['timestamp']}")
            
            if metadata.get('estimated_remaining', 0) > 0:
                remaining_str = str(timedelta(seconds=int(metadata['estimated_remaining'])))
                self.logger.info(f"⏱️  Geschätzte verbleibende Zeit: {remaining_str}")
            
            return processed_images, results_by_image
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint laden fehlgeschlagen: {e}")
            self.logger.info("🆕 Starte neue Analyse")
            return None
    
    def create_backup(self, backup_number: int) -> bool:
        """
        Erstelle Backup der aktuellen Checkpoint-Datei.
        
        Args:
            backup_number: Nummer des Backups
            
        Returns:
            True wenn erfolgreich erstellt
        """
        if not ENABLE_BACKUP:
            return False
        
        try:
            checkpoint_file = self.get_checkpoint_file()
            if not checkpoint_file.exists():
                return False
            
            backup_file = self.get_backup_file(backup_number)
            shutil.copy2(checkpoint_file, backup_file)
            
            self.logger.info(f"💾 Backup erstellt: {backup_file.name}")
            
            # Cleanup old backups
            self._cleanup_old_backups()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backup erstellen fehlgeschlagen: {e}")
            return False
    
    def record_image_processing_time(self, image_name: str, processing_time: float):
        """
        Zeichne Verarbeitungszeit für ein Bild auf.
        
        Args:
            image_name: Name des verarbeiteten Bildes
            processing_time: Verarbeitungszeit in Sekunden
        """
        self.image_times.append(processing_time)
        
        # Log detailed progress
        images_processed = len(self.image_times)
        avg_time = sum(self.image_times) / len(self.image_times)
        
        # Performance statistics
        min_time = min(self.image_times)
        max_time = max(self.image_times)
        
        self.logger.info(f"✅ Bild {images_processed:3d} verarbeitet: {image_name}")
        self.logger.info(f"   ⏱️  Zeit: {processing_time:.1f}s | Ø: {avg_time:.1f}s | Min: {min_time:.1f}s | Max: {max_time:.1f}s")
        
        # Show estimated time remaining and ETA
        if ESTIMATE_REMAINING_TIME and hasattr(self, 'total_images_to_process'):
            remaining_images = self.total_images_to_process - images_processed
            if remaining_images > 0:
                estimated_remaining = avg_time * remaining_images
                remaining_str = str(timedelta(seconds=int(estimated_remaining)))
                progress_pct = (images_processed / self.total_images_to_process) * 100
                
                # Calculate ETA
                eta = datetime.now() + timedelta(seconds=estimated_remaining)
                eta_str = eta.strftime("%H:%M:%S")
                
                self.logger.info(f"   📊 Fortschritt: {progress_pct:.1f}% ({images_processed}/{self.total_images_to_process})")
                self.logger.info(f"   ⏳ Verbleibend: ~{remaining_str} | ETA: {eta_str}")
                
                # Performance trend analysis for large batches
                if images_processed >= 5:
                    recent_times = self.image_times[-5:]  # Last 5 images
                    recent_avg = sum(recent_times) / len(recent_times)
                    trend = "🔺 langsamer" if recent_avg > avg_time * 1.1 else "🔻 schneller" if recent_avg < avg_time * 0.9 else "➡️  stabil"
                    self.logger.info(f"   📈 Trend (letzte 5): {recent_avg:.1f}s {trend}")
                
                # Memory usage warning for very large batches
                if images_processed % 50 == 0 and images_processed > 50:
                    self.logger.info(f"   💾 Checkpoint-Tipp: {images_processed} Bilder verarbeitet - regelmäßiges Speichern aktiv")
                
                # Milestone celebrations for motivation 🎉
                milestones = [10, 25, 50, 100, 150, 200, 250, 500, 1000]
                if images_processed in milestones:
                    total_elapsed = sum(self.image_times)
                    self.logger.info(f"   🎉 MEILENSTEIN: {images_processed} Bilder in {str(timedelta(seconds=int(total_elapsed)))}")
    
    def set_total_images(self, total_count: int):
        """Setze Gesamtanzahl der zu verarbeitenden Bilder."""
        self.total_images_to_process = total_count
        self.logger.info(f"📊 Batch-Verarbeitung: {total_count} Bilder geplant")
    
    def cleanup_checkpoints(self):
        """Lösche alle Checkpoint-Dateien nach erfolgreichem Abschluss."""
        if not ENABLE_CHECKPOINTS:
            return
        
        try:
            if self.checkpoint_dir.exists():
                for file in self.checkpoint_dir.glob("*.json"):
                    file.unlink()
                self.logger.info("🧹 Checkpoint-Dateien aufgeräumt")
        except Exception as e:
            self.logger.warning(f"⚠️  Checkpoint-Cleanup fehlgeschlagen: {e}")
    
    def _cleanup_old_backups(self):
        """Lösche alte Backup-Dateien wenn Limit überschritten."""
        try:
            backup_files = list(self.checkpoint_dir.glob("backup_*_checkpoint.json"))
            backup_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            if len(backup_files) > MAX_BACKUPS:
                for old_backup in backup_files[MAX_BACKUPS:]:
                    old_backup.unlink()
                    self.logger.debug(f"🗑️  Altes Backup gelöscht: {old_backup.name}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️  Backup-Cleanup fehlgeschlagen: {e}")
    
    def _serialize_enhanced_product(self, product) -> Dict:
        """
        Konvertiere EnhancedProduct Objekt zu serialisierbarem Dict.
        
        Args:
            product: EnhancedProduct Objekt oder bereits Dict
            
        Returns:
            Serialisierbares Dictionary
        """
        if isinstance(product, dict):
            return product
        
        try:
            # Standard attributes
            result = {
                'name': getattr(product, 'name', 'unknown'),
                'brand': getattr(product, 'brand', 'unknown'),
                'type': getattr(product, 'type', 'unknown'),
                'confidence': getattr(product, 'confidence', 0.5),
                'approx_count': getattr(product, 'approx_count', 1),
                'keywords': getattr(product, 'keywords', []),
                'category_display_name': getattr(product, 'category_display_name', 'Unknown'),
                'main_category': getattr(product, 'main_category', 'unknown'),
                'subcategory': getattr(product, 'subcategory', 'unknown'),
                'price': getattr(product, 'price', None),
                'text_content': getattr(product, 'text_content', [])
            }
            
            # Handle nested objects
            if hasattr(product, 'brand_classification') and product.brand_classification:
                bc = product.brand_classification
                result['brand_classification'] = {
                    'origin': getattr(bc, 'origin', 'unknown'),
                    'confidence': getattr(bc, 'confidence', 0.0),
                    'classification_method': getattr(bc, 'classification_method', 'unknown'),
                    'matched_patterns': getattr(bc, 'matched_patterns', [])
                }
            
            if hasattr(product, 'eye_level_data') and product.eye_level_data:
                eld = product.eye_level_data
                result['eye_level_data'] = {
                    'zone': getattr(eld, 'zone', 'unknown'),
                    'y_position': getattr(eld, 'y_position', 0.0),
                    'is_premium_zone': getattr(eld, 'is_premium_zone', False),
                    'shelf_tier': getattr(eld, 'shelf_tier', 'unknown')
                }
            
            return result
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"⚠️  Serialization fallback für Produkt: {e}")
            return {
                'name': str(product),
                'brand': 'unknown',
                'type': 'unknown',
                'confidence': 0.5,
                'serialization_error': str(e)
            }


def get_checkpoint_manager(output_folder: Path) -> CheckpointManager:
    """Factory function für CheckpointManager."""
    return CheckpointManager(output_folder)