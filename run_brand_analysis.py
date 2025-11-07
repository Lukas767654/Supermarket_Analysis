#!/usr/bin/env python3
"""
🚀 Brand & Product Analysis Pipeline - Launcher
===============================================

Einfacher Launcher für die Supermarket Brand Analysis Pipeline.

Verwendung:
    python run_brand_analysis.py

Optionen:
    --config-check    Nur Konfiguration prüfen
    --dry-run        Testlauf ohne echte API-Calls
    --images PATH    Alternativer Bilder-Ordner
    --output PATH    Alternativer Output-Ordner

Beispiele:
    python run_brand_analysis.py
    python run_brand_analysis.py --config-check
    python run_brand_analysis.py --images ./meine_bilder/
"""

import sys
import argparse
import logging
from pathlib import Path

# Füge aktuelles Verzeichnis zum Path hinzu
sys.path.insert(0, str(Path(__file__).parent))

try:
    import config_brand_analysis as config
    from brand_product_pipeline import main as run_pipeline
    
    # Import aller Konfigurationsvariablen
    from config_brand_analysis import (
        IMAGE_FOLDER, OUTPUT_FOLDER, EXAMPLES_FOLDER, INTERMEDIATE_FOLDER,
        GOOGLE_API_KEY, CONFIDENCE_THRESHOLD, DUPLICATE_SCORE_THRESHOLD,
        CLUSTERING_EPS, MIN_SAMPLES, EMBEDDING_MODEL, IMAGE_RESIZE_WIDTH,
        ENABLE_SEGMENTATION, SEGMENTS_PER_IMAGE, SUPPORTED_EXTENSIONS,
        LOG_LEVEL, LOG_TO_FILE, LOG_FILENAME,
        print_config_summary, validate_config
    )
    
except ImportError as e:
    print(f"❌ Import-Fehler: {e}")
    print("Stellen Sie sicher, dass alle Dateien im gleichen Ordner sind.")
    sys.exit(1)

def setup_logging():
    """Setup Logging basierend auf Konfiguration"""
    log_level = getattr(logging, LOG_LEVEL.upper())
    
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if LOG_TO_FILE:
        log_file = OUTPUT_FOLDER / LOG_FILENAME
        OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def check_requirements():
    """Prüfe ob alle Requirements installiert sind"""
    logger = logging.getLogger(__name__)
    
    required_packages = [
        'numpy', 'pandas', 'PIL', 'cv2', 'torch', 
        'sentence_transformers', 'sklearn', 'faiss',
        'requests', 'openpyxl', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                import PIL
            elif package == 'cv2':
                import cv2
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            logger.debug(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error("Fehlende Pakete gefunden!")
        logger.error("Installieren Sie mit:")
        logger.error(f"pip install -r requirements_brand_analysis.txt")
        logger.error(f"Oder einzeln: pip install {' '.join(missing_packages)}")
        return False
    
    logger.info("✅ Alle Requirements verfügbar!")
    return True

def check_input_folder():
    """Prüfe Input-Ordner"""
    logger = logging.getLogger(__name__)
    
    if not IMAGE_FOLDER.exists():
        logger.error(f"❌ Bilder-Ordner nicht gefunden: {IMAGE_FOLDER}")
        logger.info("Erstelle den Ordner und kopiere Ihre Bilder hinein:")
        logger.info(f"mkdir -p {IMAGE_FOLDER}")
        return False
    
    # Zähle Bilder
    image_count = 0
    for ext in SUPPORTED_EXTENSIONS:
        image_count += len(list(IMAGE_FOLDER.glob(f"*{ext}")))
        image_count += len(list(IMAGE_FOLDER.glob(f"*{ext.upper()}")))
    
    if image_count == 0:
        logger.error(f"❌ Keine Bilder gefunden in: {IMAGE_FOLDER}")
        logger.info(f"Unterstützte Formate: {', '.join(SUPPORTED_EXTENSIONS)}")
        return False
    
    logger.info(f"✅ {image_count} Bilder gefunden in {IMAGE_FOLDER}")
    return True

def dry_run():
    """Führe einen Testlauf durch ohne echte API-Calls"""
    logger = logging.getLogger(__name__)
    
    logger.info("🧪 DRY-RUN MODUS - Keine echten API-Calls")
    logger.info("=" * 50)
    
    # Simuliere Pipeline-Schritte
    steps = [
        "📷 Bilder einlesen und segmentieren",
        "🤖 Gemini Vision API (simuliert)",
        "🔍 OCR & Logo-Erkennung (simuliert)", 
        "🔀 Ergebnis-Fusion",
        "🎯 Embedding-Berechnung",
        "🔍 Clustering & Duplikat-Erkennung",
        "📊 Cluster-Analyse",
        "📈 Excel-Export",
        "🖼️ Beispielbilder kopieren"
    ]
    
    for i, step in enumerate(steps, 1):
        logger.info(f"Schritt {i}/9: {step}")
        
        if "API" in step or "OCR" in step:
            logger.info("  → Simuliert (dry-run)")
        else:
            logger.info("  → Würde ausgeführt werden")
    
    logger.info("✅ Dry-run abgeschlossen")
    logger.info("Starten Sie ohne --dry-run für echte Analyse")

def main():
    """Haupt-Launcher-Funktion"""
    
    parser = argparse.ArgumentParser(
        description="Brand & Product Analysis Pipeline für Supermarktbilder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python run_brand_analysis.py                    # Normale Ausführung
  python run_brand_analysis.py --config-check    # Nur Konfiguration prüfen  
  python run_brand_analysis.py --dry-run         # Testlauf ohne API
  python run_brand_analysis.py --images ./test/  # Anderer Bilder-Ordner
        """
    )
    
    parser.add_argument(
        '--config-check', 
        action='store_true',
        help='Nur Konfiguration und Requirements prüfen'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true', 
        help='Testlauf ohne echte API-Calls'
    )
    
    parser.add_argument(
        '--images',
        type=Path,
        help='Alternativer Pfad zum Bilder-Ordner'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Alternativer Pfad zum Output-Ordner'
    )
    
    args = parser.parse_args()
    
    # Override Pfade wenn angegeben
    if args.images:
        global IMAGE_FOLDER
        IMAGE_FOLDER = args.images
    
    if args.output:
        global OUTPUT_FOLDER, EXAMPLES_FOLDER, INTERMEDIATE_FOLDER
        OUTPUT_FOLDER = args.output
        EXAMPLES_FOLDER = OUTPUT_FOLDER / "examples"
        INTERMEDIATE_FOLDER = OUTPUT_FOLDER / "intermediate"
    
    # Setup Logging
    logger = setup_logging()
    
    # Header
    logger.info("🚀 Brand & Product Analysis Pipeline")
    logger.info("=" * 50)
    
    # Konfiguration anzeigen
    print_config_summary()
    
    # Validierungen
    validate_config()
    
    if not check_requirements():
        sys.exit(1)
    
    if not check_input_folder():
        sys.exit(1)
    
    # Nur Config-Check?
    if args.config_check:
        logger.info("✅ Konfiguration und Requirements OK!")
        logger.info("Pipeline ist bereit für den Start.")
        return
    
    # Dry-Run?
    if args.dry_run:
        dry_run()
        return
    
    # Erstelle Output-Ordner
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    EXAMPLES_FOLDER.mkdir(parents=True, exist_ok=True)
    INTERMEDIATE_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Starte Pipeline
    try:
        logger.info("🚀 Starte Brand Analysis Pipeline...")
        logger.info("=" * 50)
        
        # Update globale Variablen im Pipeline-Modul
        import brand_product_pipeline as pipeline
        pipeline.IMAGE_FOLDER = str(IMAGE_FOLDER)
        pipeline.OUTPUT_FOLDER = str(OUTPUT_FOLDER) 
        pipeline.EXAMPLES_FOLDER = str(EXAMPLES_FOLDER)
        pipeline.GOOGLE_API_KEY = GOOGLE_API_KEY
        pipeline.CONFIDENCE_THRESHOLD = CONFIDENCE_THRESHOLD
        pipeline.DUPLICATE_SCORE_THRESHOLD = DUPLICATE_SCORE_THRESHOLD
        pipeline.CLUSTERING_EPS = CLUSTERING_EPS
        pipeline.MIN_SAMPLES = MIN_SAMPLES
        pipeline.EMBEDDING_MODEL = EMBEDDING_MODEL
        pipeline.IMAGE_RESIZE_WIDTH = IMAGE_RESIZE_WIDTH
        pipeline.ENABLE_SEGMENTATION = ENABLE_SEGMENTATION
        pipeline.SEGMENTS_PER_IMAGE = SEGMENTS_PER_IMAGE
        
        # Führe Pipeline aus
        run_pipeline()
        
        logger.info("🎉 Pipeline erfolgreich abgeschlossen!")
        logger.info(f"📊 Ergebnisse verfügbar in: {OUTPUT_FOLDER}")
        
    except KeyboardInterrupt:
        logger.info("❌ Pipeline durch Benutzer abgebrochen")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Pipeline-Fehler: {e}")
        logger.error("Für Details siehe Log-Datei oder starten Sie mit --debug")
        sys.exit(1)

if __name__ == "__main__":
    main()