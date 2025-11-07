"""
Configuration settings for the retail audit pipeline.
"""
import os
from dataclasses import dataclass
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Config:
    """Configuration class for the retail audit pipeline."""
    
    # Paths
    ASSETS_PATH: str = "./assets"
    OUTPUT_PATH: str = "./outputs"
    JSON_PATH: str = "./outputs/json"
    CROPS_PATH: str = "./outputs/crops"
    THUMBS_PATH: str = "./outputs/thumbs"
    EXCEL_PATH: str = "./outputs/excel"
    
    # Google Cloud & Gemini
    GOOGLE_APPLICATION_CREDENTIALS: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    
    # Processing settings
    MAX_IMAGES: int = int(os.getenv("MAX_IMAGES", "0"))  # 0 = no limit
    RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))
    
    # OCR Settings
    OCR_LANGUAGE_HINTS: list = None
    
    # Confidence Thresholds
    LOGO_MIN_CONF: float = float(os.getenv("LOGO_MIN_CONF", "0.75"))
    ACCEPT_CONF: float = float(os.getenv("ACCEPT_CONF", "0.75"))
    REVIEW_CONF: float = float(os.getenv("REVIEW_CONF", "0.50"))
    
    # Duplicate Detection Thresholds
    PHASH_MAX_DIST: int = int(os.getenv("PHASH_MAX_DIST", "6"))
    SSIM_MIN: float = float(os.getenv("SSIM_MIN", "0.92"))
    JACCARD_MIN: float = float(os.getenv("JACCARD_MIN", "0.80"))
    EMBEDDING_MIN: float = float(os.getenv("EMBEDDING_MIN", "0.98"))
    
    # Placement Detection
    EYE_LEVEL_MIN: float = float(os.getenv("EYE_LEVEL_MIN", "0.40"))
    EYE_LEVEL_MAX: float = float(os.getenv("EYE_LEVEL_MAX", "0.60"))
    
    # Thumbnail size
    THUMBNAIL_SIZE: tuple = (200, 200)
    
    # Thai digit mapping
    THAI_DIGITS: Dict[str, str] = None
    
    # Brand family mappings (expandable)
    BRAND_FAMILIES: Dict[str, str] = None
    
    # Size extraction patterns
    SIZE_PATTERNS: list = None
    
    def __post_init__(self):
        """Initialize default values that can't be set directly in dataclass."""
        if self.OCR_LANGUAGE_HINTS is None:
            self.OCR_LANGUAGE_HINTS = ["th", "en"]
        if self.SIZE_PATTERNS is None:
            self.SIZE_PATTERNS = [
                r"(\d+(?:[.,]\d+)?)\s?(?:ml|มล\.?)",
                r"(\d+(?:[.,]\d+)?)\s?(?:l|ลิตร)",
                r"(\d+(?:[.,]\d+)?)\s?(?:g|กรัม|grams?)",
                r"(\d+(?:[.,]\d+)?)\s?(?:kg|กิโลกรัม|kilograms?)",
                r"(\d+(?:[.,]\d+)?)\s?(?:แพ็ค|pack|pcs?|pieces?)",
                r"(\d+(?:[.,]\d+)?)\s?(?:oz|ออนซ์)",
                r"(\d+(?:[.,]\d+)?)\s?(?:lb|ปอนด์|pounds?)"
            ]
        if self.THAI_DIGITS is None:
            self.THAI_DIGITS = {
                "๐": "0", "๑": "1", "๒": "2", "๓": "3", "๔": "4",
                "๕": "5", "๖": "6", "๗": "7", "๘": "8", "๙": "9"
            }
        if self.BRAND_FAMILIES is None:
            self.BRAND_FAMILIES = {
                "Coca-Cola": "The Coca-Cola Company",
                "Pepsi": "PepsiCo", 
                "Sprite": "The Coca-Cola Company",
                "Fanta": "The Coca-Cola Company",
                "7UP": "Keurig Dr Pepper",
                "Nestle": "Nestlé S.A.",
                "Unilever": "Unilever",
                "P&G": "Procter & Gamble"
            }

# Global config instance
config = Config()