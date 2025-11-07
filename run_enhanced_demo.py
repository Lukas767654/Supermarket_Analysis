#!/usr/bin/env python3
"""
Enhanced Brand Analysis - Demo Runner
=====================================
Demonstriert die neue Vollbild-Pipeline mit:
- Eye-Level Detection 
- Thai vs International Brand Classification
- Vollständige Produktanalysierung ohne Segmentierung
"""

import logging
from pathlib import Path
from enhanced_brand_pipeline import main as run_enhanced_pipeline

# Setup Enhanced Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('enhanced_brand_analysis.log')
    ]
)

def demo_main():
    """Führe Enhanced Brand Analysis Demo aus"""
    
    logger = logging.getLogger(__name__)
    
    print("🚀 Enhanced Brand Analysis Pipeline - DEMO")
    print("=" * 60)
    print("📌 Features:")
    print("  • Vollbild-Analyse (KEINE Segmentierung)")
    print("  • Eye-Level Detection für Shelf-Positioning")  
    print("  • Thai vs International Brand Classification")
    print("  • Cloud Vision API Integration")
    print("  • Multi-Level Excel Reports")
    print("=" * 60)
    
    # Prüfe Bilder-Ordner
    from config_brand_analysis import IMAGE_FOLDER
    
    if not IMAGE_FOLDER.exists():
        print(f"⚠️  Bilder-Ordner nicht gefunden: {IMAGE_FOLDER}")
        print("📁 Erstelle Bilder-Ordner...")
        IMAGE_FOLDER.mkdir(parents=True, exist_ok=True)
        
    # Zähle Bilder  
    image_files = []
    from config_brand_analysis import SUPPORTED_EXTENSIONS
    
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(list(IMAGE_FOLDER.glob(f"*{ext}")))
        image_files.extend(list(IMAGE_FOLDER.glob(f"*{ext.upper()}")))
    
    print(f"📷 Gefundene Bilder: {len(image_files)}")
    
    if len(image_files) == 0:
        print("⚠️  Keine Bilder gefunden!")
        print(f"📋 Bitte fügen Sie Bilder in den Ordner ein: {IMAGE_FOLDER}")
        print("   Unterstützte Formate: .jpg, .jpeg, .png, .webp, .bmp")
        return False
    
    for img in image_files:
        print(f"  - {img.name}")
    
    print(f"\n🔄 Starte Enhanced Analysis für {len(image_files)} Bilder...")
    
    try:
        # Führe Enhanced Pipeline aus
        results_path = run_enhanced_pipeline()
        
        print(f"\n✅ Enhanced Analysis abgeschlossen!")
        print(f"📊 Ergebnisse gespeichert: {results_path}")
        
        # Zeige Output-Struktur
        from config_brand_analysis import OUTPUT_FOLDER
        
        print(f"\n📁 Output-Struktur:")
        for output_file in sorted(OUTPUT_FOLDER.rglob("*")):
            if output_file.is_file():
                print(f"  📄 {output_file.relative_to(OUTPUT_FOLDER)}")
        
        print(f"\n🎯 Key Features demonstriert:")
        print(f"  ✅ Vollbild-Analyse ohne Segmentierung")
        print(f"  ✅ Eye-Level Detection implementiert")
        print(f"  ✅ Thai vs International Brand Classification")
        print(f"  ✅ Enhanced Excel Reports erstellt")
        
        return True
        
    except Exception as e:
        logger.error(f"Demo fehlgeschlagen: {e}", exc_info=True)
        print(f"\n❌ Demo fehlgeschlagen: {e}")
        return False

if __name__ == "__main__":
    success = demo_main()
    
    if success:
        print(f"\n🎉 Demo erfolgreich abgeschlossen!")
        print(f"💡 Sie können jetzt weitere Bilder hinzufügen und das System erneut ausführen.")
    else:
        print(f"\n⚠️  Demo nicht erfolgreich. Bitte Logs überprüfen.")