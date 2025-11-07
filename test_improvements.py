#!/usr/bin/env python3
"""
Test-Skript für die Eye-Level und Brand Classification Verbesserungen
"""

import json
import sys
from pathlib import Path
from collections import Counter

def test_brand_classification():
    """Teste die Brand Classification"""
    print("🧪 TESTING BRAND CLASSIFICATION")
    print("=" * 50)
    
    try:
        # Import the classifier
        from enhanced_brand_pipeline import BrandClassifier
        
        classifier = BrandClassifier()
        
        # Test bekannte Marken
        test_cases = [
            ("Fumme", "thai"),
            ("Glade", "international"), 
            ("Downy", "international"),
            ("OASIS", "thai"),
            ("Nivea", "international"),
            ("Sure", "thai"),
            ("UnknownBrand", "unknown")
        ]
        
        results = []
        for brand, expected in test_cases:
            result = classifier.classify_brand(brand)
            success = "✅" if result.origin == expected else "❌"
            results.append((brand, result.origin, expected, success))
            print(f"{success} {brand}: {result.origin} (expected: {expected}) - {result.confidence:.2f}")
        
        # Statistiken
        passed = sum(1 for _, _, _, success in results if success == "✅")
        total = len(results)
        print(f"\n📊 Results: {passed}/{total} passed ({passed/total*100:.1f}%)")
        
        return passed == total
        
    except Exception as e:
        print(f"❌ Error testing brand classification: {e}")
        return False

def analyze_current_results():
    """Analysiere aktuelle Ergebnisse"""
    print("\n🔍 ANALYZING CURRENT RESULTS")
    print("=" * 50)
    
    results_file = Path("brand_analysis_output/enhanced_results.json")
    
    if not results_file.exists():
        print("❌ No results file found yet")
        return
    
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        print(f"📊 Total products: {len(results)}")
        
        # Brand origin analysis
        origins = [p.get('origin', 'unknown') for p in results]
        origin_counts = Counter(origins)
        
        print("\n🏷️ Origin Distribution:")
        for origin, count in origin_counts.items():
            percentage = count / len(results) * 100
            print(f"  {origin}: {count} ({percentage:.1f}%)")
        
        # Eye-level analysis  
        if results and 'eye_level_data' in results[0]:
            zones = [p['eye_level_data'].get('zone', 'unknown') for p in results]
            zone_counts = Counter(zones)
            
            print("\n👁️ Eye-Level Distribution:")
            for zone, count in zone_counts.items():
                percentage = count / len(results) * 100
                print(f"  {zone}: {count} ({percentage:.1f}%)")
            
            # Check if eye-level positions are diverse
            y_positions = [p['eye_level_data'].get('y_position', 0.5) for p in results]
            unique_positions = len(set(y_positions))
            
            if unique_positions == 1:
                print(f"⚠️  All products have same y_position: {y_positions[0]}")
            else:
                print(f"✅ Eye-level positions are diverse: {unique_positions} unique values")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

def main():
    """Hauptfunktion"""
    print("🚀 TESTING ENHANCED PIPELINE IMPROVEMENTS")
    print("=" * 60)
    
    # Test 1: Brand Classification
    classification_ok = test_brand_classification()
    
    # Test 2: Current Results Analysis
    analyze_current_results()
    
    # Summary
    print("\n" + "=" * 60)
    if classification_ok:
        print("✅ Brand classification logic works correctly!")
        print("💡 If results still show 'unknown', the issue is in pipeline integration")
    else:
        print("❌ Brand classification logic has issues")
    
    print("\n📋 NEXT STEPS:")
    print("1. Wait for pipeline to finish running")
    print("2. Check if brand classifications are now working")
    print("3. Verify eye-level positions are diverse")
    print("4. Implement additional improvements if needed")

if __name__ == "__main__":
    main()