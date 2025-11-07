#!/usr/bin/env python3
"""
Test-Skript für DeepSeek Brand Classification
"""

import requests
import json
import sys
from pathlib import Path

def test_ollama_connection():
    """Teste Ollama-Verbindung"""
    print("🔌 TESTING OLLAMA CONNECTION")
    print("=" * 40)
    
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama running, {len(models)} models available:")
            for model in models:
                print(f"  - {model['name']} ({model['size']})")
            return True
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ollama connection failed: {e}")
        return False

def test_deepseek_classification():
    """Teste DeepSeek Brand Classification direkt"""
    print("\n🤖 TESTING DEEPSEEK CLASSIFICATION")
    print("=" * 40)
    
    # Test Cases - unbekannte/schwierige Marken
    test_brands = [
        {
            'brand': 'Fumme',
            'type': 'air freshener',
            'expected': 'thai',
            'ocr': ['air', 'freshener', 'fumme']
        },
        {
            'brand': 'KOALA THE BEAR', 
            'type': 'air freshener',
            'expected': 'unknown',  # Schwieriger Fall
            'ocr': ['koala', 'bear', 'fresh']
        },
        {
            'brand': 'Fresh Gel',
            'type': 'air freshener', 
            'expected': 'thai',  # Könnte thai sein
            'ocr': ['fresh', 'gel']
        },
        {
            'brand': 'ชัวร์ ชัวร์',  # Thai Schrift
            'type': 'air freshener',
            'expected': 'thai',
            'ocr': ['ชัวร์', 'sure']
        }
    ]
    
    results = []
    
    for test_case in test_brands:
        brand = test_case['brand']
        product_type = test_case['type']
        ocr_tokens = test_case['ocr']
        expected = test_case['expected']
        
        print(f"\n🧪 Testing: {brand}")
        
        # DeepSeek Prompt
        context = f"Product type: {product_type} | OCR text: {', '.join(ocr_tokens)}"
        
        prompt = f"""You are a brand classification expert. Use web search to find accurate information about this brand.

TASK: Classify "{brand}" as either "thai" or "international"

BRAND TO RESEARCH: {brand}
PRODUCT CONTEXT: {context}

SEARCH INSTRUCTIONS:
1. Search for "{brand} brand origin country company"
2. Search for "{brand} Thailand manufacturer"
3. Look for company headquarters, founding location, parent company

CLASSIFICATION RULES:
- THAI: Brands from Thailand, local Thai companies, Thai script brands
- INTERNATIONAL: Global brands, multinational companies, foreign brands

REQUIRED OUTPUT FORMAT:
CLASSIFICATION: [thai/international] 
CONFIDENCE: [0.1-1.0]
REASONING: [brief explanation with search findings]

Please search the web first, then classify: {brand}"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gpt-oss:20b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 200,
                        "enable_web_search": True,
                        "web_search": True
                    }
                },
                timeout=90  # Längeres Timeout für Web Search
            )
            
            if response.status_code == 200:
                result = response.json().get('response', '').strip()
                print(f"  DeepSeek: {result}")
                
                # Parse result
                parts = result.split('|')
                if len(parts) >= 2:
                    origin = parts[0].strip().lower()
                    confidence = parts[1].strip()
                    reasoning = parts[2].strip() if len(parts) > 2 else "no_reason"
                    
                    success = "✅" if origin == expected else "❓"
                    print(f"  Result: {origin} (confidence: {confidence}) {success}")
                    print(f"  Expected: {expected}")
                    print(f"  Reasoning: {reasoning}")
                    
                    results.append({
                        'brand': brand,
                        'result': origin,
                        'expected': expected,
                        'confidence': confidence,
                        'correct': origin == expected
                    })
                else:
                    print(f"  ❌ Unparseable response: {result}")
                    results.append({
                        'brand': brand,
                        'result': 'parse_error',
                        'expected': expected,
                        'confidence': 0,
                        'correct': False
                    })
            else:
                print(f"  ❌ API Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Summary
    if results:
        correct = sum(1 for r in results if r['correct'])
        total = len(results)
        print(f"\n📊 DeepSeek Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
        
        return results
    else:
        return []

def test_integrated_pipeline():
    """Teste integrierte Pipeline mit DeepSeek"""
    print("\n🔗 TESTING INTEGRATED PIPELINE")
    print("=" * 40)
    
    try:
        sys.path.append('.')
        from enhanced_brand_pipeline import BrandClassifier
        
        # Teste mit DeepSeek aktiviert
        classifier = BrandClassifier(enable_deepseek=True)
        
        test_brands = [
            'KOALA THE BEAR',
            'Fresh Gel', 
            'ROOM FRESH',
            'Hygiene'
        ]
        
        for brand in test_brands:
            print(f"\n🧪 Testing integrated: {brand}")
            result = classifier.classify_brand(
                brand, 
                keywords=['air', 'freshener'],
                product_type='air freshener',
                ocr_tokens=['air', 'fresh', brand.lower()]
            )
            
            print(f"  Origin: {result.origin}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Method: {result.classification_method}")
            
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def main():
    """Hauptfunktion"""
    print("🚀 DEEPSEEK BRAND CLASSIFICATION TEST")
    print("=" * 60)
    
    # Test 1: Ollama Connection
    if not test_ollama_connection():
        print("❌ Ollama not available, stopping tests")
        return
    
    # Test 2: Direct DeepSeek Classification
    deepseek_results = test_deepseek_classification()
    
    # Test 3: Integrated Pipeline
    pipeline_ok = test_integrated_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    if deepseek_results:
        correct_count = sum(1 for r in deepseek_results if r['correct'])
        print(f"✅ DeepSeek direct: {correct_count}/{len(deepseek_results)} correct")
    
    if pipeline_ok:
        print("✅ Pipeline integration: Working")
    else:
        print("❌ Pipeline integration: Failed")
    
    print("\n💡 NEXT STEPS:")
    print("1. Run enhanced pipeline with DeepSeek fallback")
    print("2. Check reduction in 'unknown' classifications")
    print("3. Monitor DeepSeek accuracy vs cost/time")

if __name__ == "__main__":
    main()