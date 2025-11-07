"""
Konfigurationsdatei für die Brand & Product Analysis Pipeline
============================================================
Hier können Sie alle ]



# Marken-Klassifikations-Konfigurationameter zentral anpassen.
"""

import os
from pathlib import Path

# =============================================================================
# 🔧 PFAD-KONFIGURATION
# =============================================================================

# Basis-Verzeichnis (automatisch erkannt)
BASE_DIR = Path(__file__).parent

# Input-Ordner für Bilder
IMAGE_FOLDER = BASE_DIR / "images"

# Output-Ordner für alle Ergebnisse
OUTPUT_FOLDER = BASE_DIR / "brand_analysis_output"

# Beispielbilder-Ordner
EXAMPLES_FOLDER = OUTPUT_FOLDER / "examples"

# Zwischenergebnisse
INTERMEDIATE_FOLDER = OUTPUT_FOLDER / "intermediate"

# =============================================================================
# 🔑 API-KONFIGURATION
# =============================================================================

# Load environment variables for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded in config")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment only")

# Google API Key (Gemini & Vision) - from .env file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', "AIzaSyBxZQmbHOml59U1rxb2_Gd2dRnjnwwzLHY")

# =============================================================================
# 🎯 KONFIGURATIONSVARIABLEN
# =============================================================================

# Unterstützte Bildformate
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']

# Timeouts
VISION_TIMEOUT = 30

# Confidence-Schwellenwerte
CONFIDENCE_THRESHOLD = 0.75        # Mindest-Confidence für Ergebnisse
MIN_CONFIDENCE_FOR_CLUSTERING = 0.5  # Mindest-Confidence für Clustering

# Duplikat-Erkennung
DUPLICATE_SCORE_THRESHOLD = 0.8    # Ähnlichkeitsschwelle für Duplikate
VISUAL_WEIGHT = 0.6               # Gewichtung visueller Ähnlichkeit (0-1)
TEXT_WEIGHT = 0.4                 # Gewichtung Text-Ähnlichkeit (0-1)

# Clustering-Parameter
CLUSTERING_EPS = 0.2              # DBSCAN epsilon (Nachbarschaftsradius)
MIN_SAMPLES = 2                   # DBSCAN min_samples (Min. Punkte pro Cluster)

# Embedding-Fusion
IMAGE_EMBEDDING_WEIGHT = 0.7      # Gewichtung Bild-Embeddings (0-1)
TEXT_EMBEDDING_WEIGHT = 0.3       # Gewichtung Text-Embeddings (0-1)

# =============================================================================
# 🖼️ BILD-VERARBEITUNG
# =============================================================================

# Bild-Größenanpassung
IMAGE_RESIZE_WIDTH = 2048          # Max. Breite für verarbeitete Bilder
JPEG_QUALITY = 95                  # JPEG-Qualität für gespeicherte Bilder

# Segmentierung (DEAKTIVIERT - Vollbild-Analyse)
ENABLE_SEGMENTATION = False        # Deaktiviere Segmentierung - analysiere Vollbilder
SEGMENTS_PER_IMAGE = 1            # Nur ein Segment = Vollbild
MIN_HEIGHT_FOR_SEGMENTATION = 99999 # Nie segmentieren

# Unterstützte Dateiformate
SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.heic']

# =============================================================================
# 👁️ EYE-LEVEL & SHELF-ANALYSE
# =============================================================================

# Eye-Level Detection
ENABLE_EYE_LEVEL_DETECTION = True    # Aktiviere Eye-Level Erkennung
EYE_LEVEL_ZONES = {
    'top_shelf': (0.0, 0.25),       # Oberes 25% des Bildes
    'eye_level': (0.25, 0.65),      # Augenhöhe 25-65%
    'middle': (0.35, 0.75),         # Mittlerer Bereich 35-75%  
    'bottom': (0.65, 1.0)           # Unterer Bereich 65-100%
}

# Shelf-Position Analyse
ANALYZE_SHELF_POSITION = True        # Analysiere Regal-Positionen
PREMIUM_ZONES = ['eye_level', 'middle']  # Premium-Platzierungen

# =============================================================================
# 🇹🇭 MARKEN-KLASSIFIKATION (Thai vs International)
# =============================================================================

# Thailändische vs. Internationale Marken
ENABLE_BRAND_ORIGIN_CLASSIFICATION = True

# Bekannte thailändische Marken
THAI_BRANDS = [
    # Große thailändische Brands
    'mama', 'yum yum', 'wai wai', 'โรบินสัน', 'robinson', 'เซเว่น', '7-eleven',
    'ชัวร์', 'sure', 'ควันน้อย', 'ช้างเวป', 'kayari', 'เคยารี่',
    'โอเลย์', 'เบสท์', 'best', 'เซนต์', 'saint', 'เฟรช', 'fresh',
    
    # Lokale Haushaltsmarken
    'ไทยแลนด์', 'thailand', 'สยาม', 'siam', 'กรุงเทพ', 'bangkok',
    'เมืองไทย', 'muangthai', 'อินโดจีน', 'indochina',
    
    # Lokale Lebensmittelmarken  
    'มาม่า', 'ยำยำ', 'ไวไว', 'อินโดมี่', 'indomie',
    'นกแก้ว', 'นกยูง', 'peacock', 'เสือ', 'tiger',
    
    # Thai Kosmetik/Pflege
    'วาสลีน', 'โออิส', 'oasis', 'เฟรชเจล', 'fresh gel',
    'ฟัมเม่', 'fumme', 'ฟัมมี่', 'fummie',
    
    # Traditionelle Thai Brands
    'หงส์ทอง', 'golden swan', 'เย็นตาโฟ', 'yen ta fo',
    'ต้มยำกุ้ง', 'tom yum', 'น้ำพริก', 'nam prik'
]

# Bekannte internationale Marken  
INTERNATIONAL_BRANDS = [
    # Große Global Brands
    'unilever', 'procter', 'gamble', 'p&g', 'johnson', 'johnson\'s',
    'nivea', 'l\'oreal', 'loreal', 'garnier', 'maybelline', 'dove',
    'olay', 'pantene', 'head', 'shoulders', 'gillette', 'oral-b',
    
    # Haushalts-Globals
    'glade', 'febreze', 'lysol', 'ajax', 'vim', 'comfort',
    'downy', 'tide', 'ariel', 'persil', 'vanish',
    
    # Getränke Global
    'coca-cola', 'coke', 'pepsi', 'sprite', 'fanta', 'schweppes',
    'nestle', 'nescafe', 'milo', 'ovaltine', 'lipton',
    
    # Süßwaren Global
    'mars', 'snickers', 'twix', 'kit-kat', 'oreo', 'ritz',
    'pringles', 'lay\'s', 'doritos', 'cheetos',
    
    # Pflege Global
    'vaseline', 'pond\'s', 'clean', 'clear', 'biore', 'neutrogena',
    'aveeno', 'eucerin', 'cetaphil', 'la roche-posay'
]

# Brand Classification Configuration
BRAND_CLASSIFICATION_CONFIG = {
    'fuzzy_matching': True,
    'similarity_threshold': 0.8,
    'default_classification': 'unknown',
    'manual_classification': True
}

# =============================================================================
# 🏬 CJMORE PRIVATE BRAND CLASSIFICATION
# =============================================================================

# CJMore Private Brand Detection
ENABLE_CJMORE_CLASSIFICATION = True

# CJMore Eigenmarken (Private Brands)
CJMORE_PRIVATE_BRANDS = [
    # Exakte Namen (case-insensitive)
    'uno',
    'nine beauty',
    'bao cafe', 
    'tian tian',
    
    # Variationen und häufige Schreibweisen
    'uno.',
    'uno ',
    'ninebeauty',
    'nine-beauty',
    '9beauty',
    '9 beauty',
    'baocafe',
    'bao-cafe',
    'bao café',
    'tiantian',
    'tian-tian',
    'tiān tiān',  # Mit Akzenten
]

# Fuzzy Matching Threshold für CJMore Brands
CJMORE_FUZZY_THRESHOLD = 0.85

# CJMore Classification Configuration
CJMORE_CLASSIFICATION_CONFIG = {
    'enable_fuzzy_matching': True,
    'case_sensitive': False,
    'min_confidence': 0.7,
    'check_ocr_tokens': True,       # Auch OCR-Text nach Private Brands durchsuchen
    'check_product_name': True      # Produktname auf Private Brands prüfen
}

# =============================================================================
# 📊 BRAND PRODUCT COUNTING SYSTEM
# =============================================================================

# Brand Product Diversity Analysis
ENABLE_BRAND_PRODUCT_COUNTING = True

# Product Counting Configuration
BRAND_COUNTING_CONFIG = {
    'count_method': 'unique_types',     # 'unique_types', 'unique_names', 'all_products'
    'min_confidence': 0.5,              # Mindest-Confidence für Zählung
    'exclude_unknown': True,            # Exkludiere 'unknown' brands
    'deduplicate_similar': True,        # Verwende Cross-Image Deduplication Ergebnisse
    'similarity_threshold': 0.9,        # Ähnlichkeitsgrenze für "gleiche" Produkte
    'group_by_main_category': False     # Gruppiere Zählung nach Hauptkategorie
}

# Minimum Product Count für Reports (filtere Brands mit wenigen Produkten)
MIN_PRODUCTS_FOR_BRAND_REPORT = 1

# =============================================================================
# 🔄 CROSS-IMAGE DUPLICATE DETECTION
# =============================================================================

# Cross-Image Deduplication Configuration
ENABLE_CROSS_IMAGE_DEDUPLICATION = True

# DBSCAN clustering parameters for duplicate detection
CROSS_IMAGE_EPS = 0.15                    # Distance threshold for clustering (lower = stricter)
MIN_SIMILARITY_FOR_DUPLICATE = 0.75      # Minimum similarity to consider as duplicate

# Similarity weights (must sum to ≤ 1.0)
VISUAL_SIMILARITY_WEIGHT = 0.4           # Weight for visual/embedding similarity
TEXT_SIMILARITY_WEIGHT = 0.3             # Weight for OCR text similarity  
BRAND_SIMILARITY_WEIGHT = 0.3            # Weight for exact brand/type matching

# Quality thresholds
MIN_CONFIDENCE_FOR_CANONICAL = 0.3       # Minimum confidence for canonical selection
MIN_OCR_TOKENS_FOR_PREFERENCE = 3        # Prefer products with more OCR tokens

# Performance settings
CROSS_IMAGE_BATCH_SIZE = 1000            # Process products in batches to manage memory
ENABLE_PARALLEL_SIMILARITY = True        # Use parallel processing for similarity computation
MAX_WORKERS_SIMILARITY = 4               # Number of threads for parallel processing

# Reporting
SAVE_DEDUPLICATION_REPORT = True         # Save detailed deduplication report
DEDUPLICATION_REPORT_FILE = "cross_image_deduplication_report.json"

# =============================================================================
# 📄 CHECKPOINT & RECOVERY SYSTEM
# =============================================================================

# Checkpoint Configuration
ENABLE_CHECKPOINTS = True                 # Enable checkpoint system for large batches
CHECKPOINT_FREQUENCY = 5                  # Save checkpoint every N images
CHECKPOINT_DIR = "checkpoints"            # Directory for checkpoint files
AUTO_RESUME = True                        # Automatically resume from last checkpoint

# Progress Logging
ENABLE_DETAILED_LOGGING = True            # Enable detailed progress logs
LOG_EVERY_N_IMAGES = 1                    # Log progress every N images
ESTIMATE_REMAINING_TIME = True            # Show estimated time remaining
SAVE_INTERMEDIATE_RESULTS = True          # Save results after each image

# Backup Settings  
ENABLE_BACKUP = True                      # Create backup of results
BACKUP_FREQUENCY = 10                     # Backup every N images
MAX_BACKUPS = 5                          # Keep max N backup files

# =============================================================================
# 🤖 MODEL-KONFIGURATION
# =============================================================================

# Embedding-Modell
EMBEDDING_MODEL = "clip-ViT-B-32"  # SentenceTransformers Modell

        # Gemini-Modell
GEMINI_MODEL = "gemini-2.5-flash"   # Gemini Vision Model# API-Timeouts (Sekunden)
GEMINI_TIMEOUT = 60               # Timeout für Gemini-Requests
VISION_TIMEOUT = 30               # Timeout für Vision API

# =============================================================================
# 📊 OUTPUT-KONFIGURATION
# =============================================================================

# Excel-Export
EXCEL_FILENAME = "brand_type_summary.xlsx"
INCLUDE_INTERMEDIATE_SHEETS = True  # Zusätzliche Detail-Sheets

# Beispielbilder
COPY_EXAMPLE_IMAGES = True        # Kopiere Beispielbilder
MAX_EXAMPLES_PER_CLUSTER = 1      # Max. Beispiele pro Cluster

# Logging
LOG_LEVEL = "INFO"                # DEBUG, INFO, WARNING, ERROR
LOG_TO_FILE = True                # Schreibe Logs in Datei
LOG_FILENAME = "pipeline.log"

# =============================================================================
# 🔍 FILTER & BEREINIGUNG
# =============================================================================

# OCR-Bereinigung
REMOVE_PRICE_PATTERNS = True      # Entferne Preisangaben
REMOVE_NUMERIC_CODES = True       # Entferne numerische Codes
MIN_TOKEN_LENGTH = 2              # Mindestlänge für Tokens

# Marken-Whitelist (bekannte Marken - wird für bessere Erkennung verwendet)
KNOWN_BRANDS = [
    'nivea', 'loreal', 'olay', 'pond', 'garnier', 'dove', 'vaseline',
    'johnson', 'baby', 'oasis', 'glade', 'kayari', 'mama', 'knorr',
    'colgate', 'oral-b', 'listerine', 'sensodyne', 'pepsodent',
    'pantene', 'head', 'shoulders', 'tresemme', 'clear', 'sunsilk',
    'coke', 'coca-cola', 'pepsi', 'sprite', 'fanta', 'schweppes',
    'nestle', 'maggi', 'nescafe', 'milo', 'kit-kat'
]

# Produkttyp-Kategorien
PRODUCT_CATEGORIES = {
    'beauty': ['lotion', 'cream', 'serum', 'oil', 'soap', 'cleanser'],
    'haircare': ['shampoo', 'conditioner', 'hair-oil', 'styling'],
    'oral-care': ['toothpaste', 'mouthwash', 'toothbrush'],
    'beverages': ['drink', 'juice', 'soda', 'water', 'tea', 'coffee'],
    'food': ['snack', 'powder', 'sauce', 'noodles', 'seasoning'],
    'household': ['detergent', 'cleaner', 'freshener', 'spray']
}

# =============================================================================
# 🚀 PERFORMANCE-OPTIMIERUNG
# =============================================================================

# Parallelisierung
MAX_WORKERS = 4                   # Max. parallele Threads
BATCH_SIZE = 10                   # Batch-Größe für API-Requests

# Cache
ENABLE_CACHING = True            # Cache API-Responses
CACHE_DURATION_HOURS = 24        # Cache-Dauer in Stunden

# Memory-Management
MAX_EMBEDDING_BATCH_SIZE = 100   # Max. Batch für Embedding-Berechnung
CLEAR_CACHE_AFTER_STEP = True    # Leere Cache nach jedem Schritt

def validate_config():
    """
    Validiere die Konfiguration und gib Warnungen aus.
    """
    warnings = []
    
    # Prüfe API Key
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
        warnings.append("⚠️  Google API Key nicht konfiguriert!")
    
    # Prüfe Input-Ordner
    if not IMAGE_FOLDER.exists():
        warnings.append(f"⚠️  Bilder-Ordner nicht gefunden: {IMAGE_FOLDER}")
    
    # Prüfe Parameter-Bereiche
    if not (0 <= VISUAL_WEIGHT <= 1) or not (0 <= TEXT_WEIGHT <= 1):
        warnings.append("⚠️  Gewichtungen müssen zwischen 0 und 1 liegen!")
    
    if abs(VISUAL_WEIGHT + TEXT_WEIGHT - 1.0) > 0.001:
        warnings.append("⚠️  Visual + Text Weights sollten 1.0 ergeben!")
    
    # Ausgabe
    if warnings:
        print("KONFIGURATIONS-WARNUNGEN:")
        for warning in warnings:
            print(f"  {warning}")
        print()
    else:
        print("✅ Konfiguration validiert - alles OK!")
        print()

def print_config_summary():
    """
    Gib eine Zusammenfassung der aktuellen Konfiguration aus.
    """
    print("📋 AKTUELLE KONFIGURATION:")
    print("=" * 40)
    print(f"🖼️  Bilder-Ordner: {IMAGE_FOLDER}")
    print(f"📊 Output-Ordner: {OUTPUT_FOLDER}")
    print(f"🤖 Embedding-Modell: {EMBEDDING_MODEL}")
    print(f"🎯 Confidence-Schwelle: {CONFIDENCE_THRESHOLD}")
    print(f"🔍 Duplikat-Schwelle: {DUPLICATE_SCORE_THRESHOLD}")
    print(f"📏 Clustering EPS: {CLUSTERING_EPS}")
    print(f"🔗 Segmentierung: {'Aktiviert' if ENABLE_SEGMENTATION else 'Deaktiviert'}")
    print("=" * 40)
    print()

if __name__ == "__main__":
    print_config_summary()
    validate_config()