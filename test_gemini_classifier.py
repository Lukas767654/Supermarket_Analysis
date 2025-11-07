#!/usr/bin/env python3
"""
Gemini Flash Lite Fallback Classifier - Test Implementation
"""

import json
import requests
import logging
from dataclasses import dataclass
from typing import List, Optional
import re

@dataclass
class BrandClassification:
    origin: str
    confidence: float
    matched_patterns: List[str]
    classification_method: str

class GeminiFallbackClassifier:
    """Verwendet Gemini 2.5 Flash Lite für schnelle Fallback-Klassifikation unbekannter Marken"""
    
    def __init__(self, api_key="AIzaSyBxZQmbHOml59U1rxb2_Gd2dRnjnwwzLHY"):
        self.api_key = api_key
        self.model_name = "gemini-2.0-flash-exp"  # Verfügbares Modell verwenden
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("Google API Key required for Gemini fallback classifier")
    
    def classify_unknown_brand(self, brand_name: str, product_type: str = "", 
                             ocr_tokens: List[str] = None) -> BrandClassification:
        """Klassifiziere unbekannte Marke mit Gemini"""
        
        # Context erstellen
        context_parts = []
        if product_type:
            context_parts.append(f"Product: {product_type}")
        if ocr_tokens:
            context_parts.append(f"OCR: {', '.join(ocr_tokens[:3])}")
        
        context = " | ".join(context_parts) if context_parts else ""
        
        # Gemini Prompt - kurz und präzise
        prompt = f"""Classify brand "{brand_name}" as Thai or International.

Context: {context}

Rules:
- Thai: Local Thai brands, Thai script (ก-๙), Southeast Asian regional
- International: Global/multinational brands, Western companies

Examples:
- Thai: mama, sure, oasis, fumme, kayari  
- International: nivea, dove, glade, downy, p&g

Answer: CLASSIFICATION: [thai/international] | CONFIDENCE: [0.6-1.0]

Brand: {brand_name}"""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 100,
                    "topP": 0.8
                }
            }
            
            response = requests.post(url, json=payload, timeout=15)
            
            if response.status_code == 200:
                result = response.json()
                if 'candidates' in result and result['candidates']:
                    text = result['candidates'][0]['content']['parts'][0]['text'].strip()
                    return self._parse_gemini_response(text, brand_name)
                else:
                    self.logger.warning(f"No candidates in Gemini response for {brand_name}")
            else:
                self.logger.error(f"Gemini API Error {response.status_code}: {response.text}")
                
        except Exception as e:
            self.logger.error(f"Gemini classification error for '{brand_name}': {e}")
        
        # Fallback - als international (safer default)
        return BrandClassification(
            origin='international',
            confidence=0.4,
            matched_patterns=[brand_name],
            classification_method='gemini_failed'
        )
    
    def _parse_gemini_response(self, response: str, brand_name: str) -> BrandClassification:
        """Parse Gemini Antwort"""
        try:
            response_lower = response.lower().strip()
            
            # Pattern: "CLASSIFICATION: thai/international | CONFIDENCE: 0.X"
            classification_match = re.search(r'classification:\s*(thai|international)', response_lower)
            confidence_match = re.search(r'confidence:\s*([\d.]+)', response_lower)
            
            if classification_match:
                origin = classification_match.group(1)
                confidence = 0.7  # Default
                
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                    except ValueError:
                        confidence = 0.7
                
                return BrandClassification(
                    origin=origin,
                    confidence=min(max(confidence, 0.0), 1.0),
                    matched_patterns=[f"structured: {origin}"],
                    classification_method='gemini_flash_lite'
                )
            
            # Fallback: Keyword-basiert
            if any(word in response_lower for word in ['thai', 'thailand']):
                return BrandClassification(
                    origin='thai',
                    confidence=0.7,
                    matched_patterns=[response[:80]],
                    classification_method='gemini_keyword'
                )
            
            if any(word in response_lower for word in ['international', 'global', 'multinational']):
                return BrandClassification(
                    origin='international',
                    confidence=0.7,
                    matched_patterns=[response[:80]],
                    classification_method='gemini_keyword'
                )
                
        except Exception as e:
            print(f"Parse error: {e}")
        
        # Default fallback
        return BrandClassification(
            origin='international',
            confidence=0.4,
            matched_patterns=[response[:50]],
            classification_method='gemini_fallback'
        )

def test_gemini_classifier():
    """Teste den Gemini Classifier"""
    print("🤖 TESTING GEMINI FALLBACK CLASSIFIER")
    print("=" * 50)
    
    classifier = GeminiFallbackClassifier()
    
    test_brands = [
        {'name': 'KOALA THE BEAR', 'expected': 'unknown'},
        {'name': 'Fresh Gel', 'expected': 'thai'},
        {'name': 'ROOM FRESH', 'expected': 'unknown'},
        {'name': 'Hygiene', 'expected': 'international'}
    ]
    
    results = []
    
    for test_case in test_brands:
        brand = test_case['name']
        expected = test_case['expected']
        
        print(f"\n🧪 Testing: {brand}")
        result = classifier.classify_unknown_brand(
            brand, 
            product_type="air freshener",
            ocr_tokens=['air', 'fresh', brand.lower()]
        )
        
        print(f"  Result: {result.origin}")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Method: {result.classification_method}")
        print(f"  Expected: {expected}")
        
        results.append({
            'brand': brand,
            'result': result.origin,
            'expected': expected,
            'confidence': result.confidence
        })
    
    # Summary
    print(f"\n📊 RESULTS:")
    for r in results:
        status = "✅" if r['result'] != 'unknown' else "❓"
        print(f"  {status} {r['brand']}: {r['result']} (conf: {r['confidence']:.2f})")

if __name__ == "__main__":
    test_gemini_classifier()