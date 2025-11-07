"""
Data models for the retail audit pipeline using Pydantic.
"""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
import hashlib

class CropContext(BaseModel):
    """Input context for Gemini consolidation."""
    image_id: str
    crop_id: str
    bbox: List[float] = Field(..., min_items=4, max_items=4)
    type: str
    ocr_raw: Optional[str] = None
    ocr_tokens: List[str] = Field(default_factory=list)
    logo_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    scores: Dict[str, float] = Field(default_factory=dict)

class ConsolidatedResult(BaseModel):
    """Structured output from Gemini consolidation."""
    image_id: str
    crop_id: str
    brand_th: Optional[str] = None
    brand_en: Optional[str] = None
    product_th: Optional[str] = None
    product_en: Optional[str] = None
    variant_th: Optional[str] = None
    variant_en: Optional[str] = None
    size_raw: Optional[str] = None
    size_normalized: Optional[str] = None
    category_th: Optional[str] = None
    category_en: Optional[str] = None
    subcategory_th: Optional[str] = None
    subcategory_en: Optional[str] = None
    signage_type: Literal["CategoryHeader", "Promo", "Claim", "Price", "None"] = "None"
    private_label: Literal["yes", "no", "unknown"] = "unknown"
    facings_hint: Optional[int] = Field(None, ge=0)
    placement: Literal["EyeLevel", "Endcap", "Checkout", "None"] = "None"
    conf_logo: float = Field(0.0, ge=0.0, le=1.0)
    conf_ocr: float = Field(0.0, ge=0.0, le=1.0)
    conf_detector: float = Field(0.0, ge=0.0, le=1.0)
    conf_overall: float = Field(0.0, ge=0.0, le=1.0)
    review_needed: bool = False
    notes: Optional[str] = None

    @validator('conf_overall', always=True)
    def compute_overall_confidence(cls, v, values):
        """Compute overall confidence score."""
        if 'conf_logo' in values and 'conf_ocr' in values and 'conf_detector' in values:
            embedding_match = 0  # Set to 0 unless computed
            conf = (
                0.40 * values['conf_logo'] + 
                0.35 * values['conf_ocr'] + 
                0.15 * values['conf_detector'] + 
                0.10 * embedding_match
            )
            return round(conf, 3)
        return v

    @validator('review_needed', always=True)
    def compute_review_needed(cls, v, values):
        """Determine if review is needed."""
        if 'conf_overall' in values:
            return values['conf_overall'] < 0.75
        return v

    def get_sku_id(self) -> str:
        """Generate SKU ID from brand|product|size."""
        components = [
            self.brand_en or "",
            self.product_en or "",
            self.size_normalized or ""
        ]
        sku_string = "|".join(components).lower()
        return hashlib.md5(sku_string.encode()).hexdigest()[:12]

class ImageMetadata(BaseModel):
    """Metadata for processed images."""
    image_id: str
    file_path: str
    file_name: str
    file_size: int
    dimensions: tuple
    quality_score: float
    sharpness: float
    brightness_mean: float
    phash: str
    crop_count: int = 0
    duplicate_group_id: Optional[str] = None
    is_duplicate: bool = False
    canonical_image_id: Optional[str] = None

class AuditRecord(BaseModel):
    """Complete audit record combining all data."""
    # Core identifiers
    image_id: str
    crop_id: str
    duplicate_group_id: Optional[str] = None
    is_duplicate: bool = False
    canonical_image_id: Optional[str] = None
    
    # SKU information
    sku_id: str
    brand_family: Optional[str] = None
    is_private_label: str = "unknown"
    
    # Product details (bilingual)
    brand_en: Optional[str] = None
    brand_th: Optional[str] = None
    product_en: Optional[str] = None
    product_th: Optional[str] = None
    variant_en: Optional[str] = None
    variant_th: Optional[str] = None
    
    # Category information
    category_en: Optional[str] = None
    category_th: Optional[str] = None
    subcategory_en: Optional[str] = None
    subcategory_th: Optional[str] = None
    
    # Placement and visibility
    facing_count_in_row: Optional[int] = None
    placement: str = "None"
    highlight_flag: bool = False
    
    # Signage and merchandising
    signage_type: str = "None"
    signage_text_en: Optional[str] = None
    signage_text_th: Optional[str] = None
    store_zone: str = "Aisle"
    private_label_display_flag: bool = False
    
    # Quality metrics
    conf_overall: float = 0.0
    review_needed: bool = False
    
    # File paths
    crop_thumbnail_path: Optional[str] = None
    source_image_path: Optional[str] = None
    
    # Raw data (optional)
    ocr_raw: Optional[str] = None
    logo_candidates: Optional[str] = None
    detector_class: Optional[str] = None
    notes: Optional[str] = None

class CategorySummary(BaseModel):
    """Category-level aggregated summary."""
    category_en: str
    sku_count: int
    variety_index: float
    assortment_depth: int
    private_label_share_skus: float
    private_label_share_facings: float
    top_skus_by_facings: List[str]
    top_brands_by_facings: List[str]
    category_role: Literal["Destination", "Routine", "Impulse", "Unknown"]
    banner_count_by_zone: Dict[str, int]
    on_shelf_signage_count: int
    private_label_featured_count: int
    promo_rate: float
    eye_level_ratio: float