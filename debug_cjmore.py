"""
Debug Script für CJMore Classification
Testet warum Brands wie Glade, Fumme als Private Brands klassifiziert werden
"""

import sys
sys.path.append('/Users/lukaswalter/Documents/GitHub/Supermarket_Analysis')

from enhanced_brand_pipeline import CJMoreClassifier
from config_brand_analysis import *

def test_cjmore_classification():
    classifier = CJMoreClassifier()
    
    # Test bekannte Non-Private Brands
    test_brands = [
        'Glade',
        'Fumme', 
        'KOALA THE BEAR',
        'Hygiene',
        'Funme',
        'Chure Chure',
        'Fresh Gel',
        'Room Fresh'
    ]
    
    # Test echte CJMore Private Brands
    real_private_brands = [
        'UNO',
        'NINE BEAUTY', 
        'BAO CAFE',
        'TIAN TIAN'
    ]
    
    print("🔍 DEBUGGING CJMore Classification:")
    print("=" * 60)
    
    print("\n❌ Testing NON-Private Brands (should be FALSE):")
    for brand in test_brands:
        product_data = {
            'brand': brand,
            'name': f'{brand} Product',
            'ocr_tokens': [brand.lower(), 'product', 'test']
        }
        
        result = classifier.classify_product(product_data)
        
        print(f"Brand: {brand:15} -> Private: {result.is_private_brand:5} | "
              f"Confidence: {result.confidence:.3f} | "
              f"Method: {result.detection_method:12} | "
              f"Pattern: '{result.matched_pattern}'")
    
    print("\n✅ Testing REAL Private Brands (should be TRUE):")
    for brand in real_private_brands:
        product_data = {
            'brand': brand,
            'name': f'{brand} Product', 
            'ocr_tokens': [brand.lower(), 'product', 'test']
        }
        
        result = classifier.classify_product(product_data)
        
        print(f"Brand: {brand:15} -> Private: {result.is_private_brand:5} | "
              f"Confidence: {result.confidence:.3f} | "
              f"Method: {result.detection_method:12} | "
              f"Pattern: '{result.matched_pattern}'")

if __name__ == "__main__":
    test_cjmore_classification()