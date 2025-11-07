"""
Excel export functionality for retail audit results.
Creates comprehensive reports with All Findings and Deduped Summary sheets.
"""
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter
import hashlib

from .config import config
from .models import ImageMetadata, ConsolidatedResult, AuditRecord, CategorySummary

logger = logging.getLogger(__name__)

class ExcelReportGenerator:
    """Generates comprehensive Excel reports from audit results."""
    
    def __init__(self):
        self.brand_families = config.BRAND_FAMILIES
    
    def create_audit_records(self, 
                           image_metadata: List[ImageMetadata],
                           consolidated_results: Dict[str, List[ConsolidatedResult]],
                           duplicate_results: Dict[str, Any]) -> List[AuditRecord]:
        """
        Create comprehensive audit records from all input data.
        
        Args:
            image_metadata: List of image metadata
            consolidated_results: Dictionary mapping image_id to results
            duplicate_results: Duplicate detection results
            
        Returns:
            List of AuditRecord objects
        """
        audit_records = []
        
        # Create metadata lookup
        metadata_dict = {img.image_id: img for img in image_metadata}
        
        # Process each image's results
        for image_id, results in consolidated_results.items():
            metadata = metadata_dict.get(image_id)
            if not metadata:
                continue
            
            for result in results:
                if result is None:
                    continue
                
                # Create audit record
                record = self._create_audit_record(result, metadata, duplicate_results)
                audit_records.append(record)
        
        logger.info(f"Created {len(audit_records)} audit records")
        return audit_records
    
    def _create_audit_record(self, 
                           result: ConsolidatedResult,
                           metadata: ImageMetadata,
                           duplicate_results: Dict[str, Any]) -> AuditRecord:
        """Create individual audit record from consolidated result."""
        
        # Get SKU ID
        sku_id = result.get_sku_id()
        
        # Determine brand family
        brand_family = None
        if result.brand_en:
            brand_family = self.brand_families.get(result.brand_en, None)
        
        # Get duplicate information
        group_assignments = duplicate_results.get('group_assignments', {})
        canonical_assignments = duplicate_results.get('canonical_assignments', {})
        
        duplicate_group_id = group_assignments.get(result.image_id)
        is_duplicate = result.image_id in group_assignments
        canonical_image_id = canonical_assignments.get(result.image_id, result.image_id)
        
        # Determine highlight flag
        highlight_flag = self._calculate_highlight_flag(result)
        
        # Determine store zone
        store_zone = self._determine_store_zone(result)
        
        # Create file paths
        crop_thumbnail_path = f"outputs/thumbs/{result.crop_id}_thumb.jpg"
        source_image_path = metadata.file_path
        
        # Private label display flag
        private_label_display_flag = (
            result.private_label == "yes" and 
            result.signage_type in ["Promo", "Claim"] and
            result.placement in ["EyeLevel", "Endcap"]
        )
        
        record = AuditRecord(
            # Core identifiers
            image_id=result.image_id,
            crop_id=result.crop_id,
            duplicate_group_id=duplicate_group_id,
            is_duplicate=is_duplicate,
            canonical_image_id=canonical_image_id,
            
            # SKU information
            sku_id=sku_id,
            brand_family=brand_family,
            is_private_label=result.private_label,
            
            # Product details (bilingual)
            brand_en=result.brand_en,
            brand_th=result.brand_th,
            product_en=result.product_en,
            product_th=result.product_th,
            variant_en=result.variant_en,
            variant_th=result.variant_th,
            
            # Category information
            category_en=result.category_en,
            category_th=result.category_th,
            subcategory_en=result.subcategory_en,
            subcategory_th=result.subcategory_th,
            
            # Placement and visibility
            facing_count_in_row=result.facings_hint,
            placement=result.placement,
            highlight_flag=highlight_flag,
            
            # Signage and merchandising
            signage_type=result.signage_type,
            signage_text_en=result.product_en if result.signage_type != "None" else None,
            signage_text_th=result.product_th if result.signage_type != "None" else None,
            store_zone=store_zone,
            private_label_display_flag=private_label_display_flag,
            
            # Quality metrics
            conf_overall=result.conf_overall,
            review_needed=result.review_needed,
            
            # File paths
            crop_thumbnail_path=crop_thumbnail_path,
            source_image_path=source_image_path,
            
            # Raw data
            ocr_raw=result.notes,  # Store notes as OCR raw for now
            logo_candidates=None,  # Can be populated if needed
            detector_class=None,  # Can be populated if needed
            notes=result.notes
        )
        
        return record
    
    def _calculate_highlight_flag(self, result: ConsolidatedResult) -> bool:
        """Calculate highlight flag based on placement and signage."""
        return (
            (result.placement in ["Endcap", "EyeLevel"] or 
             result.signage_type in ["Promo", "Claim"] or
             result.conf_logo >= 0.9) and
            result.conf_overall >= config.ACCEPT_CONF
        )
    
    def _determine_store_zone(self, result: ConsolidatedResult) -> str:
        """Determine store zone based on placement and signage."""
        if result.placement == "Endcap":
            return "Endcap"
        elif result.placement == "Checkout" or result.signage_type == "Price":
            return "Checkout"
        elif result.signage_type in ["CategoryHeader", "Promo"]:
            return "Front"
        else:
            return "Aisle"
    
    def create_category_summaries(self, records: List[AuditRecord]) -> List[CategorySummary]:
        """
        Create category-level aggregated summaries.
        
        Args:
            records: List of audit records (canonical only)
            
        Returns:
            List of CategorySummary objects
        """
        # Group records by category
        category_groups = defaultdict(list)
        for record in records:
            if record.category_en:
                category_groups[record.category_en].append(record)
        
        summaries = []
        
        for category_en, category_records in category_groups.items():
            summary = self._create_category_summary(category_en, category_records)
            summaries.append(summary)
        
        logger.info(f"Created {len(summaries)} category summaries")
        return summaries
    
    def _create_category_summary(self, category_en: str, 
                               records: List[AuditRecord]) -> CategorySummary:
        """Create summary for a single category."""
        
        # Basic counts
        sku_count = len(set(r.sku_id for r in records if r.sku_id))
        
        # Variety index (distinct SKUs per subcategory)
        subcategory_skus = defaultdict(set)
        for record in records:
            if record.subcategory_en and record.sku_id:
                subcategory_skus[record.subcategory_en].add(record.sku_id)
        
        variety_index = sum(len(skus) for skus in subcategory_skus.values()) / len(subcategory_skus) if subcategory_skus else 0.0
        
        # Assortment depth (sum of facings)
        assortment_depth = sum(r.facing_count_in_row or 0 for r in records)
        
        # Private label analysis
        pl_skus = [r for r in records if r.is_private_label == "yes"]
        private_label_share_skus = len(pl_skus) / sku_count if sku_count > 0 else 0.0
        
        pl_facings = sum(r.facing_count_in_row or 0 for r in pl_skus)
        private_label_share_facings = pl_facings / assortment_depth if assortment_depth > 0 else 0.0
        
        # Top SKUs and brands by facings
        sku_facings = defaultdict(int)
        brand_facings = defaultdict(int)
        
        for record in records:
            if record.sku_id and record.facing_count_in_row:
                sku_facings[f"{record.brand_en} {record.product_en}"] += record.facing_count_in_row
            if record.brand_en and record.facing_count_in_row:
                brand_facings[record.brand_en] += record.facing_count_in_row
        
        top_skus_by_facings = [sku for sku, _ in sorted(sku_facings.items(), key=lambda x: x[1], reverse=True)[:5]]
        top_brands_by_facings = [brand for brand, _ in sorted(brand_facings.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # Category role determination
        category_role = self._determine_category_role(records)
        
        # Banner and signage counts
        banner_count_by_zone = self._count_banners_by_zone(records)
        on_shelf_signage_count = len([r for r in records if r.signage_type not in ["None", "CategoryHeader"]])
        
        # Private label featured count
        private_label_featured_count = len([r for r in records if r.private_label_display_flag])
        
        # Promo rate
        promo_count = len([r for r in records if r.signage_type in ["Promo", "Claim"]])
        promo_rate = promo_count / len(records) if records else 0.0
        
        # Eye level ratio
        eye_level_count = len([r for r in records if r.placement == "EyeLevel"])
        eye_level_ratio = eye_level_count / len(records) if records else 0.0
        
        return CategorySummary(
            category_en=category_en,
            sku_count=sku_count,
            variety_index=round(variety_index, 2),
            assortment_depth=assortment_depth,
            private_label_share_skus=round(private_label_share_skus, 3),
            private_label_share_facings=round(private_label_share_facings, 3),
            top_skus_by_facings=top_skus_by_facings,
            top_brands_by_facings=top_brands_by_facings,
            category_role=category_role,
            banner_count_by_zone=banner_count_by_zone,
            on_shelf_signage_count=on_shelf_signage_count,
            private_label_featured_count=private_label_featured_count,
            promo_rate=round(promo_rate, 3),
            eye_level_ratio=round(eye_level_ratio, 3)
        )
    
    def _determine_category_role(self, records: List[AuditRecord]) -> str:
        """Determine category role based on heuristics."""
        if not records:
            return "Unknown"
        
        total_facings = sum(r.facing_count_in_row or 0 for r in records)
        category_headers = len([r for r in records if r.signage_type == "CategoryHeader"])
        promo_rate = len([r for r in records if r.signage_type in ["Promo", "Claim"]]) / len(records)
        endcap_rate = len([r for r in records if r.placement == "Endcap"]) / len(records)
        checkout_rate = len([r for r in records if r.placement == "Checkout"]) / len(records)
        
        # Destination: high facings, category headers, low promo rate
        if total_facings > 50 and category_headers > 0 and promo_rate < 0.2:
            return "Destination"
        
        # Impulse: many endcaps/checkout, high promo rate
        elif endcap_rate > 0.2 or checkout_rate > 0.1 or promo_rate > 0.4:
            return "Impulse"
        
        # Routine: moderate facings, regular distribution
        elif total_facings > 10 and promo_rate < 0.4:
            return "Routine"
        
        return "Unknown"
    
    def _count_banners_by_zone(self, records: List[AuditRecord]) -> Dict[str, int]:
        """Count banners by store zone."""
        zone_counts = defaultdict(int)
        for record in records:
            if record.signage_type == "CategoryHeader":
                zone_counts[record.store_zone] += 1
        return dict(zone_counts)
    
    def create_excel_report(self, 
                           all_records: List[AuditRecord],
                           canonical_records: List[AuditRecord],
                           category_summaries: List[CategorySummary],
                           duplicate_results: Dict[str, Any],
                           output_path: str) -> bool:
        """
        Create comprehensive Excel report with multiple sheets.
        
        Args:
            all_records: All audit records including duplicates
            canonical_records: Canonical (deduplicated) records only
            category_summaries: Category-level summaries
            duplicate_results: Duplicate detection results
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                
                # Sheet 1: All Findings
                all_findings_df = self._create_all_findings_dataframe(all_records)
                all_findings_df.to_excel(writer, sheet_name='All Findings', index=False)
                
                # Sheet 2: Deduped Summary
                deduped_df = self._create_deduped_dataframe(canonical_records, category_summaries)
                deduped_df.to_excel(writer, sheet_name='Deduped Summary', index=False)
                
                # Sheet 3: KPIs (optional)
                kpis_df = self._create_kpis_dataframe(canonical_records, duplicate_results)
                kpis_df.to_excel(writer, sheet_name='KPIs', index=False)
                
                # Sheet 4: Issues (optional)
                issues_df = self._create_issues_dataframe(all_records)
                if not issues_df.empty:
                    issues_df.to_excel(writer, sheet_name='Issues', index=False)
            
            logger.info(f"Excel report created successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create Excel report: {e}")
            return False
    
    def _create_all_findings_dataframe(self, records: List[AuditRecord]) -> pd.DataFrame:
        """Create All Findings sheet dataframe."""
        data = []
        
        for record in records:
            data.append({
                'ImageID': record.image_id,
                'CropID': record.crop_id,
                'DuplicateGroupID': record.duplicate_group_id or '',
                'IsDuplicate': record.is_duplicate,
                'CanonicalImageID': record.canonical_image_id,
                'sku_id': record.sku_id,
                'brand_family': record.brand_family or '',
                'is_private_label': record.is_private_label,
                'brand_en': record.brand_en or '',
                'brand_th': record.brand_th or '',
                'product_en': record.product_en or '',
                'product_th': record.product_th or '',
                'variant_en': record.variant_en or '',
                'variant_th': record.variant_th or '',
                'category_en': record.category_en or '',
                'category_th': record.category_th or '',
                'subcategory_en': record.subcategory_en or '',
                'subcategory_th': record.subcategory_th or '',
                'facing_count_in_row': record.facing_count_in_row or 0,
                'placement': record.placement,
                'highlight_flag': record.highlight_flag,
                'signage_type': record.signage_type,
                'signage_text_en': record.signage_text_en or '',
                'signage_text_th': record.signage_text_th or '',
                'store_zone': record.store_zone,
                'private_label_display_flag': record.private_label_display_flag,
                'conf_overall': record.conf_overall,
                'review_needed': record.review_needed,
                'CropThumbnailPath': record.crop_thumbnail_path or '',
                'SourceImagePath': record.source_image_path or '',
                'notes': record.notes or ''
            })
        
        return pd.DataFrame(data)
    
    def _create_deduped_dataframe(self, 
                                canonical_records: List[AuditRecord],
                                category_summaries: List[CategorySummary]) -> pd.DataFrame:
        """Create Deduped Summary sheet dataframe."""
        data = []
        
        for summary in category_summaries:
            data.append({
                'category_en': summary.category_en,
                'sku_count': summary.sku_count,
                'variety_index': summary.variety_index,
                'assortment_depth': summary.assortment_depth,
                'private_label_share_skus': summary.private_label_share_skus,
                'private_label_share_facings': summary.private_label_share_facings,
                'top_skus_by_facings': '; '.join(summary.top_skus_by_facings[:3]),
                'top_brands_by_facings': '; '.join(summary.top_brands_by_facings[:3]),
                'category_role': summary.category_role,
                'banner_count_total': sum(summary.banner_count_by_zone.values()),
                'on_shelf_signage_count': summary.on_shelf_signage_count,
                'private_label_featured_count': summary.private_label_featured_count,
                'promo_rate': summary.promo_rate,
                'eye_level_ratio': summary.eye_level_ratio
            })
        
        return pd.DataFrame(data)
    
    def _create_kpis_dataframe(self, 
                             canonical_records: List[AuditRecord],
                             duplicate_results: Dict[str, Any]) -> pd.DataFrame:
        """Create KPIs sheet dataframe."""
        
        total_skus = len(set(r.sku_id for r in canonical_records if r.sku_id))
        pl_skus = len([r for r in canonical_records if r.is_private_label == "yes"])
        total_facings = sum(r.facing_count_in_row or 0 for r in canonical_records)
        pl_facings = sum(r.facing_count_in_row or 0 for r in canonical_records if r.is_private_label == "yes")
        
        # Category analysis
        category_facings = defaultdict(int)
        for record in canonical_records:
            if record.category_en and record.facing_count_in_row:
                category_facings[record.category_en] += record.facing_count_in_row
        
        top_3_categories = [cat for cat, _ in sorted(category_facings.items(), key=lambda x: x[1], reverse=True)[:3]]
        
        promo_count = len([r for r in canonical_records if r.signage_type in ["Promo", "Claim"]])
        endcap_count = len([r for r in canonical_records if r.placement == "Endcap"])
        eye_level_count = len([r for r in canonical_records if r.placement == "EyeLevel"])
        
        data = [{
            'total_images': duplicate_results.get('total_images', 0),
            'canonical_images': duplicate_results.get('canonical_images', 0),
            'duplicate_groups': duplicate_results.get('duplicate_groups', 0),
            'sku_count_total': total_skus,
            'private_label_share_skus': round(pl_skus / total_skus, 3) if total_skus > 0 else 0,
            'private_label_share_facings': round(pl_facings / total_facings, 3) if total_facings > 0 else 0,
            'top_3_categories_by_facings': '; '.join(top_3_categories),
            'promo_rate_total': round(promo_count / len(canonical_records), 3) if canonical_records else 0,
            'endcap_count_total': endcap_count,
            'eye_level_ratio_total': round(eye_level_count / len(canonical_records), 3) if canonical_records else 0
        }]
        
        return pd.DataFrame(data)
    
    def _create_issues_dataframe(self, records: List[AuditRecord]) -> pd.DataFrame:
        """Create Issues sheet dataframe for items needing review."""
        
        issues = [r for r in records if r.review_needed]
        
        if not issues:
            return pd.DataFrame()
        
        data = []
        for record in issues:
            data.append({
                'image_id': record.image_id,
                'crop_id': record.crop_id,
                'issue_type': 'Low Confidence' if record.conf_overall < config.ACCEPT_CONF else 'Review Required',
                'conf_overall': record.conf_overall,
                'brand_en': record.brand_en or '',
                'product_en': record.product_en or '',
                'category_en': record.category_en or '',
                'notes': record.notes or '',
                'source_image_path': record.source_image_path or ''
            })
        
        return pd.DataFrame(data)

# Global instance
excel_generator = ExcelReportGenerator()