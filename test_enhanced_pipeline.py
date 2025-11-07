#!/usr/bin/env python3
"""
Test-Runner für Enhanced Brand Pipeline
=======================================
Testet die neue Vollbild-Pipeline mit Eye-Level & Brand Classification
"""

import sys
import logging
from pathlib import Path

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_imports():
    """Test alle wichtigen Imports"""
    print("🧪 Teste Imports...")
    
    try:
        from config_brand_analysis import (
            GOOGLE_API_KEY, EYE_LEVEL_ZONES, THAI_BRANDS, 
            INTERNATIONAL_BRANDS, BRAND_CLASSIFICATION_CONFIG,
            OUTPUT_FOLDER, IMAGE_FOLDER
        )
        print("✅ Konfiguration importiert")
        
        from product_categories import get_product_category
        print("✅ Product Categories importiert")
        
        from enhanced_brand_pipeline import (
            EyeLevelDetector, BrandClassifier, EnhancedBrandAnalyzer
        )
        print("✅ Enhanced Pipeline importiert")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import-Fehler: {e}")
        return False

def test_components():
    """Test einzelne Komponenten"""
    print("\n🔧 Teste Komponenten...")
    
    try:
        from enhanced_brand_pipeline import EyeLevelDetector, BrandClassifier
        
        # Test Eye-Level Detector
        detector = EyeLevelDetector()
        print("✅ EyeLevelDetector erstellt")
        
        # Test Brand Classifier  
        classifier = BrandClassifier()
        
        # Test Thai Brand
        thai_result = classifier.classify_brand("mama")
        print(f"✅ Thai Brand Test: mama -> {thai_result.origin} ({thai_result.confidence:.2f})")
        
        # Test International Brand
        intl_result = classifier.classify_brand("glade")  
        print(f"✅ International Brand Test: glade -> {intl_result.origin} ({intl_result.confidence:.2f})")
        
        # Test Unknown Brand
        unknown_result = classifier.classify_brand("UnknownBrand123")
        print(f"✅ Unknown Brand Test: UnknownBrand123 -> {unknown_result.origin} ({unknown_result.confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Komponenten-Test fehlgeschlagen: {e}")
        return False

def test_product_categorization():
    """Test Produktkategorisierung"""
    print("\n📦 Teste Produktkategorisierung...")
    
    try:
        from product_categories import get_product_category
        
        # Test verschiedene Produkttypen
        test_products = [
            ("air freshener", ["air", "scented"]),
            ("cleaning product", ["cleaning", "household"]),
            ("beverage", ["water", "drink"]),
            ("snack", ["chips", "food"])
        ]
        
        for product_type, keywords in test_products:
            main_cat, sub_cat, display_name, confidence = get_product_category(product_type, keywords)
            print(f"✅ {product_type} -> {display_name} ({confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Kategorisierung-Test fehlgeschlagen: {e}")
        return False

def test_configuration():
    """Test Konfiguration"""
    print("\n⚙️ Teste Konfiguration...")
    
    try:
        from config_brand_analysis import (
            EYE_LEVEL_ZONES, PREMIUM_ZONES, THAI_BRANDS, 
            INTERNATIONAL_BRANDS, BRAND_CLASSIFICATION_CONFIG
        )
        
        print(f"✅ Eye-Level Zones: {len(EYE_LEVEL_ZONES)} definiert")
        print(f"✅ Premium Zones: {PREMIUM_ZONES}")
        print(f"✅ Thai Brands: {len(THAI_BRANDS)} definiert")
        print(f"✅ International Brands: {len(INTERNATIONAL_BRANDS)} definiert")
        print(f"✅ Brand Classification Config: {BRAND_CLASSIFICATION_CONFIG}")
        
        return True
        
    except Exception as e:
        print(f"❌ Konfigurations-Test fehlgeschlagen: {e}")
        return False

def main():
    """Führe alle Tests aus"""
    print("🚀 Enhanced Brand Pipeline - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_configuration, 
        test_components,
        test_product_categorization
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} fehlgeschlagen")
        except Exception as e:
            print(f"❌ {test.__name__} Ausnahme: {e}")
    
    print(f"\n📊 Testergebnisse: {passed}/{total} Tests bestanden")
    
    if passed == total:
        print("🎉 Alle Tests erfolgreich! Pipeline bereit für Execution.")
        return True
    else:
        print("⚠️  Einige Tests fehlgeschlagen. Bitte Fehler beheben.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)