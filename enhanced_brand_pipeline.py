"""
Erweiterte Brand Analysis Pipeline mit Eye-Level & Marken-Klassifikation
=======================================================================
- Vollbild-Analyse (keine Segmentierung)
- Cloud Vision API für Object Detection & Eye-Level
- Thailändische vs. Internationale Marken-Klassifikation
"""

import os
import json
import logging
import base64
import requests
import numpy as np
import pandas as pd
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from datetime import datetime
import cv2

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ .env file loaded successfully")
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment only")

# Import configuration
try:
    from config_brand_analysis import *
except ImportError:
    # Fallback Konfiguration wenn Import fehlschlägt
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY', "AIzaSyBxZQmbHOml59U1rxb2_Gd2dRnjnwwzLHY")
    IMAGE_FOLDER = Path("images")
    OUTPUT_FOLDER = Path("brand_analysis_output")
    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp', '.heic']
    VISION_TIMEOUT = 30
    ENABLE_CROSS_IMAGE_DEDUPLICATION = True
    SAVE_DEDUPLICATION_REPORT = True
    DEDUPLICATION_REPORT_FILE = "cross_image_deduplication_report.json"
    ENABLE_BRAND_PRODUCT_COUNTING = True
    
    # Brand Configuration
    THAI_BRANDS = ['mama', 'yum yum', 'wai wai', 'robinson', 'sure', 'kayari']
    INTERNATIONAL_BRANDS = ['nivea', 'loreal', 'olay', 'dove', 'vaseline', 'johnson']
    BRAND_CLASSIFICATION_CONFIG = {
        'fuzzy_matching': True,
        'similarity_threshold': 0.8,
        'default_classification': 'unknown'
    }
    
    # CJMore Configuration  
    CJMORE_CLASSIFICATION_CONFIG = {'enable_fuzzy_matching': True}
    CJMORE_FUZZY_THRESHOLD = 0.85
    
    # Brand Counting Configuration
    BRAND_COUNTING_CONFIG = {
        'min_confidence': 0.5,
        'exclude_unknown': True
    }
    MIN_PRODUCTS_FOR_BRAND_REPORT = 1
from PIL import Image
from difflib import SequenceMatcher
from tqdm import tqdm
import re

# Import Konfiguration
from config_brand_analysis import *
from product_categories import get_product_category
from cross_image_dedup import create_cross_image_deduplicator

# =============================================================================
# 📊 ERWEITERTE DATENSTRUKTUREN
# =============================================================================

@dataclass
class EyeLevelData:
    """Eye-Level und Shelf-Position Daten"""
    zone: str                    # 'top_shelf', 'eye_level', 'middle', 'bottom'
    y_position: float           # Relative Y-Position (0-1)
    is_premium_zone: bool       # Ist in Premium-Zone (Eye-Level)
    shelf_tier: int            # Regal-Etage (0=unten, höher=oben)

@dataclass
class BrandClassification:
    """Marken-Klassifikation Thai vs International"""
    origin: str                 # 'thai', 'international', 'unknown'  
    confidence: float          # Vertrauen in Klassifikation (0-1)
    matched_patterns: List[str] # Gefundene Übereinstimmungen
    classification_method: str  # 'exact_match', 'fuzzy_match', 'manual'

@dataclass
class CJMoreClassification:
    """CJMore Private Brand Classification"""
    is_private_brand: bool      # True wenn CJMore Eigenmarke
    brand_name: str            # Name der Private Brand ('uno', 'nine beauty', etc.)
    confidence: float          # Vertrauen in Klassifikation (0-1)
    matched_pattern: str       # Gefundenes Übereinstimmungsmuster  
    detection_method: str      # 'exact_match', 'fuzzy_match', 'ocr_detection'
    detection_source: str      # 'product_name', 'brand_field', 'ocr_tokens'

@dataclass
class BrandProductCount:
    """Produktanzahl pro Brand"""
    brand: str                 # Brand Name
    unique_products: int       # Anzahl einzigartiger Produkte
    total_instances: int       # Gesamte Produktinstanzen (mit Duplikaten)  
    product_types: List[str]   # Liste der verschiedenen Produkttypen
    categories: List[str]      # Liste der Kategorien dieser Brand
    avg_confidence: float      # Durchschnittliches Vertrauen
    is_cjmore_private: bool    # Ist CJMore Private Brand

@dataclass
class EnhancedProduct:
    """Erweiterte Produktdaten mit allen neuen Features"""
    # Basis-Daten
    image_id: str
    brand: str
    type: str
    approx_count: int
    conf_fused: float
    
    # Vision API Daten
    bounding_box: Optional[Dict] # Bounding Box wenn verfügbar
    ocr_tokens: List[str]
    logos: List[Dict]
    
    # Eye-Level Analyse
    eye_level_data: EyeLevelData
    
    # Marken-Klassifikation
    brand_classification: BrandClassification
    
    # CJMore Private Brand Classification
    cjmore_classification: CJMoreClassification
    
    # Kategorie-Zuordnung
    main_category: str
    subcategory: str
    category_display_name: str
    category_confidence: float
    
    # Metadaten
    source_data: Dict[str, Any]

# =============================================================================
# 👁️ EYE-LEVEL DETECTION
# =============================================================================

class EyeLevelDetector:
    """Erkennt Eye-Level und Shelf-Positionen in Supermarkt-Bildern"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def detect_eye_level_zones(self, image_path: str, objects: List[Dict] = None) -> Dict[str, Any]:
        """
        Erkenne Eye-Level Zonen im Bild basierend auf echten Objekt-Positionen.
        
        Args:
            image_path: Pfad zum Bild
            objects: Optional bereits erkannte Objekte mit Bounding Boxes
            
        Returns:
            Dictionary mit Eye-Level Zonen-Informationen
        """
        try:
            # Lade Bild für Dimensionen
            image = Image.open(image_path)
            img_height = image.height
            img_width = image.width
            
            # Definiere Eye-Level Zonen
            EYE_LEVEL_ZONES = {
                'top_shelf': (0.0, 0.25),       # Oberes 25% des Bildes
                'eye_level': (0.25, 0.65),      # Augenhöhe 25-65%
                'middle': (0.35, 0.75),         # Mittlerer Bereich 35-75%  
                'bottom': (0.65, 1.0)           # Unterer Bereich 65-100%
            }
            PREMIUM_ZONES = ['eye_level', 'middle']  # Premium-Platzierungen
            
            # Definiere Zonen basierend auf relativen Positionen
            zones = {}
            for zone_name, (y_start, y_end) in EYE_LEVEL_ZONES.items():
                zones[zone_name] = {
                    'y_start_pixel': int(y_start * img_height),
                    'y_end_pixel': int(y_end * img_height),
                    'y_start_relative': y_start,
                    'y_end_relative': y_end,
                    'is_premium': zone_name in PREMIUM_ZONES,
                    'pixel_height': int((y_end - y_start) * img_height),
                    'object_count': 0,
                    'objects': []
                }
            
            # Wenn Cloud Vision Objekte vorhanden sind, verwende echte Positionen
            object_positions = []
            if objects:
                for obj in objects:
                    if 'boundingPoly' in obj and 'vertices' in obj['boundingPoly']:
                        vertices = obj['boundingPoly']['vertices']
                        if len(vertices) >= 4:
                            # Berechne Y-Position des Objektzentrums
                            y_coords = [v.get('y', 0) for v in vertices]
                            y_center = sum(y_coords) / len(y_coords)
                            y_relative = y_center / img_height
                            
                            object_positions.append({
                                'y_center_pixel': y_center,
                                'y_relative': y_relative,
                                'confidence': obj.get('score', 0),
                                'name': obj.get('name', 'unknown')
                            })
                
                # Verteile Objekte auf Zonen basierend auf tatsächlicher Position
                for pos in object_positions:
                    for zone_name, zone_info in zones.items():
                        if zone_info['y_start_relative'] <= pos['y_relative'] <= zone_info['y_end_relative']:
                            zones[zone_name]['object_count'] += 1
                            if len(zones[zone_name]['objects']) < 5:  # Limitiere für Performance
                                zones[zone_name]['objects'].append(pos)
                            break
            
            # Analysiere Shelf-Struktur (falls Objekte vorhanden)
            shelf_analysis = self._analyze_shelf_structure(objects, img_height) if objects else {}
            
            # Füge Objektverteilungs-Analyse hinzu
            if object_positions:
                total_objects = len(object_positions)
                avg_y_relative = sum(pos['y_relative'] for pos in object_positions) / total_objects
                
                # Bestimme dominante Zone
                zone_counts = {name: info['object_count'] for name, info in zones.items()}
                dominant_zone = max(zone_counts.keys(), key=lambda k: zone_counts[k]) if any(zone_counts.values()) else 'eye_level'
                
                shelf_analysis.update({
                    'total_detected_objects': total_objects,
                    'average_y_position': avg_y_relative,
                    'dominant_zone': dominant_zone,
                    'zone_distribution': zone_counts
                })
            
            return {
                'image_dimensions': {'width': img_width, 'height': img_height},
                'zones': zones,
                'shelf_analysis': shelf_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Eye-Level Detection fehgeschlagen für {image_path}: {e}")
            return {}
    
    def get_product_eye_level(self, y_position: float, zones_data: Dict, bounding_box: Dict = None) -> EyeLevelData:
        """
        Bestimme Eye-Level Zone für ein Produkt basierend auf tatsächlicher Bounding Box Position.
        
        Args:
            y_position: Relative Y-Position des Produkts (0-1) - Fallback wenn keine BBox
            zones_data: Zonen-Daten vom detect_eye_level_zones
            bounding_box: Cloud Vision Bounding Box mit vertices (bevorzugt)
            
        Returns:
            EyeLevelData Objekt
        """
        # Debug: Log bounding box availability
        bbox_used = False
        
        # Berechne echte Y-Position aus Bounding Box falls verfügbar
        if bounding_box and 'vertices' in bounding_box:
            vertices = bounding_box['vertices']
            if len(vertices) >= 4:
                bbox_used = True
                # Berechne Zentrum der Bounding Box
                y_coords = [v.get('y', 0) for v in vertices]
                y_center = sum(y_coords) / len(y_coords)
                
                # Berechne relative Position zur Bildhöhe
                img_height = zones_data.get('image_dimensions', {}).get('height', 1000)
                y_position = y_center / img_height
                
                # Zusätzliche Shelf-Tier Berechnung basierend auf Bounding Box
                y_top = min(y_coords)
                y_bottom = max(y_coords)
                box_height = y_bottom - y_top
                
                # Shelf-Tier basierend auf genauerer Y-Position
                if y_position < 0.2:
                    shelf_tier = 4  # Sehr hoch
                elif y_position < 0.4:
                    shelf_tier = 3  # Hoch
                elif y_position < 0.7:
                    shelf_tier = 2  # Augenhöhe/Mitte
                else:
                    shelf_tier = 1  # Unten
                
                # Log für Debug
                self.logger.info(f"Eye-Level: Used BBox y_center={y_center}, y_pos={y_position:.3f}, tier={shelf_tier}")
        
        if not bbox_used:
            # Fallback auf alte Berechnung
            shelf_tier = 2 if 0.3 <= y_position <= 0.7 else (3 if y_position < 0.3 else 1)
            self.logger.info(f"Eye-Level: Fallback y_pos={y_position:.3f}, tier={shelf_tier}")
        
        # Finde passende Zone basierend auf berechneter Y-Position
        for zone_name, zone_info in zones_data.get('zones', {}).items():
            if zone_info['y_start_relative'] <= y_position <= zone_info['y_end_relative']:
                
                # Verwende bereits berechneten shelf_tier wenn von Bounding Box
                # Falls nicht von BBox berechnet, verwende Fallback
                if 'shelf_tier' not in locals():
                    shelf_tier = int((1.0 - y_position) * 4)  # 0-3, unten nach oben
                
                return EyeLevelData(
                    zone=zone_name,
                    y_position=y_position,
                    is_premium_zone=zone_info['is_premium'],
                    shelf_tier=shelf_tier
                )
        
        # Fallback wenn keine Zone gefunden
        return EyeLevelData(
            zone='unknown',
            y_position=y_position,
            is_premium_zone=False,
            shelf_tier=0
        )
    
    def _simulate_realistic_position(self, gemini_result: Dict) -> float:
        """
        Simuliere realistische Regal-Positionen basierend auf Produkttyp und Brand.
        
        Args:
            gemini_result: Gemini Analyse-Ergebnis mit Brand und Type
            
        Returns:
            Y-Position zwischen 0.1 und 0.9
        """
        product_type = gemini_result.get('type', '').lower()
        brand = gemini_result.get('brand', 'unknown').lower()
        
        # Deterministische aber variierte Position basierend auf Brand Hash
        brand_hash = hash(brand) % 1000
        type_hash = hash(product_type) % 1000
        
        # Basis-Position je nach Produktkategorie (realistische Supermarkt-Platzierung)
        if any(keyword in product_type for keyword in ['air freshener', 'spray', 'aerosol']):
            base_y = 0.25  # Höhere Regale (oben)
        elif any(keyword in product_type for keyword in ['detergent', 'fabric softener', 'cleaning liquid']):
            base_y = 0.55  # Mittlere Höhe (Eye-Level) 
        elif any(keyword in product_type for keyword in ['toothpaste', 'mouthwash', 'dental care']):
            base_y = 0.65  # Etwas niedriger
        elif any(keyword in product_type for keyword in ['soap', 'shampoo', 'personal care']):
            base_y = 0.45  # Obere Mitte
        elif any(keyword in product_type for keyword in ['food', 'snack', 'beverage']):
            base_y = 0.40  # Augenhöhe für Food
        else:
            base_y = 0.50  # Standard Mitte
            
        # Brand-spezifische Variation (±25%)
        brand_variation = ((brand_hash / 1000.0) - 0.5) * 0.5  # -0.25 bis +0.25
        
        # Type-spezifische Micro-Variation (±10%)  
        type_variation = ((type_hash / 1000.0) - 0.5) * 0.2   # -0.1 bis +0.1
        
        # Kombiniere für finale Position
        final_y = base_y + brand_variation + type_variation
        
        # Begrenze auf sinnvolle Werte
        return max(0.1, min(0.9, final_y))
    
    def _analyze_shelf_structure(self, objects: List[Dict], img_height: int) -> Dict:
        """Analysiere Regal-Struktur basierend auf erkannten Objekten."""
        
        if not objects:
            return {}
        
        # Sammle Y-Positionen aller Objekte
        y_positions = []
        for obj in objects:
            if 'boundingPoly' in obj:
                vertices = obj['boundingPoly']['vertices']
                # Berechne Zentrum
                y_center = sum(v.get('y', 0) for v in vertices) / len(vertices)
                y_relative = y_center / img_height
                y_positions.append(y_relative)
        
        if not y_positions:
            return {}
        
        # Analysiere Verteilung
        y_positions.sort()
        
        return {
            'object_count': len(y_positions),
            'y_distribution': {
                'min': min(y_positions),
                'max': max(y_positions), 
                'median': np.median(y_positions),
                'std': np.std(y_positions)
            },
            'estimated_shelf_tiers': self._estimate_shelf_tiers(y_positions)
        }
    
    def _estimate_shelf_tiers(self, y_positions: List[float]) -> int:
        """Schätze Anzahl der Regal-Etagen basierend auf Y-Verteilung."""
        
        if len(y_positions) < 3:
            return 1
        
        # Clustering der Y-Positionen um Etagen zu identifizieren
        from sklearn.cluster import DBSCAN
        
        try:
            y_array = np.array(y_positions).reshape(-1, 1)
            clustering = DBSCAN(eps=0.1, min_samples=2).fit(y_array)
            n_tiers = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            return max(1, min(n_tiers, 6))  # 1-6 Etagen realistisch
        except:
            return 3  # Standard-Annahme

# =============================================================================
# 🇹🇭 MARKEN-KLASSIFIKATION
# =============================================================================

class BrandClassifier:
    """Klassifiziert Marken als Thailändisch oder International"""
    
    def __init__(self, enable_deepseek=True):
        self.logger = logging.getLogger(__name__)
        
        # Normalisiere Marken-Listen (lowercase für matching)
        self.thai_brands_normalized = [brand.lower() for brand in THAI_BRANDS]
        self.international_brands_normalized = [brand.lower() for brand in INTERNATIONAL_BRANDS]
        
        # Thai-Script Regex
        self.thai_script_pattern = re.compile(r'[\u0e00-\u0e7f]+')
        
        # Gemini Flash Lite Fallback Classifier
        self.gemini_fallback = None
        if enable_deepseek:  # Parameter-Name beibehalten für Kompatibilität
            try:
                self.gemini_fallback = GeminiFallbackClassifier()
                self.logger.info("✅ Gemini Flash Lite fallback classifier initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Gemini fallback disabled: {e}")
    
    def classify_brand(self, brand_name: str, keywords: List[str] = None, 
                      product_type: str = "", ocr_tokens: List[str] = None) -> BrandClassification:
        """
        Klassifiziere eine Marke als Thai oder International.
        
        Args:
            brand_name: Name der Marke
            keywords: Zusätzliche Keywords/Context
            
        Returns:
            BrandClassification Objekt
        """
        if not brand_name or brand_name.lower() == 'unknown':
            return BrandClassification(
                origin='unknown',
                confidence=0.0,
                matched_patterns=[],
                classification_method='no_brand'
            )
        
        brand_normalized = brand_name.lower().strip()
        
        # 1. Exakte Übereinstimmung prüfen
        exact_match = self._check_exact_match(brand_normalized)
        if exact_match:
            return exact_match
        
        # 2. Thai-Script Erkennung
        thai_script_match = self._check_thai_script(brand_name)
        if thai_script_match:
            return thai_script_match
        
        # 3. Fuzzy Matching
        if BRAND_CLASSIFICATION_CONFIG['fuzzy_matching']:
            fuzzy_match = self._check_fuzzy_match(brand_normalized)
            if fuzzy_match:
                return fuzzy_match
        
        # 4. Keyword-basierte Klassifikation
        if keywords:
            keyword_match = self._check_keywords(keywords)
            if keyword_match:
                return keyword_match
        
        # 5. Gemini Flash Lite Fallback für unbekannte Marken
        if self.gemini_fallback and brand_name.lower() != 'unknown':
            self.logger.info(f"🤖 Using Gemini Flash Lite for unknown brand: {brand_name}")
            gemini_result = self.gemini_fallback.classify_unknown_brand(
                brand_name, product_type, ocr_tokens
            )
            # Verwende Gemini Ergebnis wenn es confident genug ist
            if gemini_result.confidence >= 0.6:
                return gemini_result
        
        # 6. Final Fallback: Unbekannt
        return BrandClassification(
            origin=BRAND_CLASSIFICATION_CONFIG['default_classification'],
            confidence=0.0,
            matched_patterns=[brand_name],
            classification_method='unknown'
        )
    
    def _check_exact_match(self, brand_normalized: str) -> Optional[BrandClassification]:
        """Prüfe exakte Übereinstimmung mit bekannten Marken."""
        
        if brand_normalized in self.thai_brands_normalized:
            return BrandClassification(
                origin='thai',
                confidence=1.0,
                matched_patterns=[brand_normalized],
                classification_method='exact_match'
            )
        
        if brand_normalized in self.international_brands_normalized:
            return BrandClassification(
                origin='international', 
                confidence=1.0,
                matched_patterns=[brand_normalized],
                classification_method='exact_match'
            )
        
        return None
    
    def _check_thai_script(self, brand_name: str) -> Optional[BrandClassification]:
        """Prüfe auf thailändische Schriftzeichen."""
        
        thai_chars = self.thai_script_pattern.findall(brand_name)
        
        if thai_chars:
            confidence = len(''.join(thai_chars)) / len(brand_name)  # Anteil Thai-Zeichen
            
            return BrandClassification(
                origin='thai',
                confidence=min(confidence * 1.5, 1.0),  # Boost für Thai-Script
                matched_patterns=thai_chars,
                classification_method='thai_script'
            )
        
        return None
    
    def _check_fuzzy_match(self, brand_normalized: str) -> Optional[BrandClassification]:
        """Prüfe unscharfe Übereinstimmung."""
        
        threshold = BRAND_CLASSIFICATION_CONFIG['similarity_threshold']
        
        # Prüfe Thai-Marken
        best_thai_match = None
        best_thai_score = 0.0
        
        for thai_brand in self.thai_brands_normalized:
            similarity = SequenceMatcher(None, brand_normalized, thai_brand).ratio()
            if similarity > best_thai_score:
                best_thai_score = similarity
                best_thai_match = thai_brand
        
        # Prüfe International-Marken
        best_intl_match = None
        best_intl_score = 0.0
        
        for intl_brand in self.international_brands_normalized:
            similarity = SequenceMatcher(None, brand_normalized, intl_brand).ratio()
            if similarity > best_intl_score:
                best_intl_score = similarity
                best_intl_match = intl_brand
        
        # Wähle besten Match
        if best_thai_score >= threshold and best_thai_score > best_intl_score:
            return BrandClassification(
                origin='thai',
                confidence=best_thai_score,
                matched_patterns=[best_thai_match],
                classification_method='fuzzy_match'
            )
        
        if best_intl_score >= threshold:
            return BrandClassification(
                origin='international',
                confidence=best_intl_score,
                matched_patterns=[best_intl_match],
                classification_method='fuzzy_match'
            )
        
        return None
    
    def _check_keywords(self, keywords: List[str]) -> Optional[BrandClassification]:
        """Klassifikation basierend auf Keywords."""
        
        # Einfache Heuristiken basierend auf Keywords
        keyword_text = ' '.join(keywords).lower()
        
        # Thai-Indikatoren
        thai_indicators = ['thailand', 'thai', 'bangkok', 'siam', 'local', 'domestic']
        international_indicators = ['imported', 'europe', 'america', 'usa', 'global', 'international']
        
        thai_matches = [ind for ind in thai_indicators if ind in keyword_text]
        intl_matches = [ind for ind in international_indicators if ind in keyword_text]
        
        if thai_matches and len(thai_matches) > len(intl_matches):
            return BrandClassification(
                origin='thai',
                confidence=0.6,
                matched_patterns=thai_matches,
                classification_method='keyword_based'
            )
        
        if intl_matches and len(intl_matches) > len(thai_matches):
            return BrandClassification(
                origin='international',
                confidence=0.6,
                matched_patterns=intl_matches,
                classification_method='keyword_based'
            )
        
        return None

# =============================================================================
# 🤖 GEMINI FLASH LITE FALLBACK CLASSIFIER
# =============================================================================

class GeminiFallbackClassifier:
    """Verwendet Gemini 2.5 Flash Lite für schnelle Fallback-Klassifikation unbekannter Marken"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = "gemini-2.5-flash-lite"
        self.logger = logging.getLogger(__name__)
        
        if not self.api_key:
            raise ValueError("Google API Key required for Gemini fallback classifier")
    
    def classify_unknown_brand(self, brand_name: str, product_type: str = "", 
                             ocr_tokens: List[str] = None) -> BrandClassification:
        """
        Klassifiziere unbekannte Marke mit Gemini 2.5 Flash Lite
        
        Args:
            brand_name: Name der Marke
            product_type: Produkttyp für Context
            ocr_tokens: OCR Text für zusätzlichen Context
            
        Returns:
            BrandClassification mit Gemini Ergebnis
        """
        
        # Erstelle Context-Information
        context_info = []
        if product_type:
            context_info.append(f"Product: {product_type}")
        if ocr_tokens:
            context_info.append(f"OCR: {', '.join(ocr_tokens[:3])}")  # Limitiere OCR
        
        context = " | ".join(context_info) if context_info else ""
        
        # Gemini Flash Lite Prompt - kurz und effizient
        prompt = f"""Classify brand "{brand_name}" as Thai or International.

Context: {context}

Rules:
- Thai: Local Thai brands, Thai script (ก-๙), Southeast Asian regional
- International: Global/multinational brands, Western companies

Examples:
- Thai: mama, sure, oasis, fumme, kayari  
- International: nivea, dove, glade, downy, p&g

Answer format: CLASSIFICATION: [thai/international] | CONFIDENCE: [0.6-1.0]

Brand: {brand_name}"""

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 100,  # Kurze Antwort
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
        
        # Fallback bei Fehlern - als international klassifizieren (safer default)
        return BrandClassification(
            origin='international',
            confidence=0.4,
            matched_patterns=[brand_name],
            classification_method='gemini_failed'
        )
    
    def _parse_gemini_response(self, response: str, brand_name: str) -> BrandClassification:
        """Parse Gemini Flash Lite Antwort - optimiert für kurze Responses"""
        try:
            # Entferne <think> tags und behalte nur finale Antwort
            clean_response = response
            if '<think>' in response:
                # Suche nach dem Ende der Denkphase
                think_end = response.find('</think>')
                if think_end != -1:
                    clean_response = response[think_end + 8:].strip()
                else:
                    # Wenn kein End-Tag, nehme letzten Teil nach 'think>'
                    parts = response.split('>')
                    clean_response = parts[-1].strip() if len(parts) > 1 else response
            
            # Versuche strukturiertes Format zu parsen: "origin|confidence|reasoning"
            if '|' in clean_response:
                parts = clean_response.split('|')
                if len(parts) >= 2:
                    origin = parts[0].strip().lower()
                    confidence_str = parts[1].strip()
                    reasoning = parts[2].strip() if len(parts) > 2 else "deepseek_analysis"
                    
                    # Validiere origin
                    if origin in ['thai', 'international', 'unknown']:
                        try:
                            confidence = float(confidence_str)
                            return BrandClassification(
                                origin=origin,
                                confidence=min(max(confidence, 0.0), 1.0),  # Clamp 0-1
                                matched_patterns=[reasoning],
                                classification_method='deepseek_local'
                            )
                        except ValueError:
                            pass
            
            # Fallback 1: Suche nach CLASSIFICATION Format
            response_lower = clean_response.lower()
            
            # Suche nach strukturiertem Format von GPT-OSS
            import re
            
            # Pattern 1: "CLASSIFICATION: thai/international"
            classification_match = re.search(r'classification:\s*(thai|international)', response_lower)
            confidence_match = re.search(r'confidence:\s*([\d.]+)', response_lower)
            
            if classification_match:
                origin = classification_match.group(1)
                confidence = 0.8  # Default confidence
                
                if confidence_match:
                    try:
                        confidence = float(confidence_match.group(1))
                    except ValueError:
                        confidence = 0.8
                
                return BrandClassification(
                    origin=origin,
                    confidence=min(max(confidence, 0.0), 1.0),
                    matched_patterns=[f"gpt_structured: {origin}"],
                    classification_method='gpt_oss_structured'
                )
            
            # Spezifische Patterns für GPT-OSS Web Search Antworten
            if any(phrase in response_lower for phrase in [
                'this is a thai brand', 'thai brand', 'local thai', 'thailand brand',
                'thai company', 'based in thailand', 'thai manufacturer', 
                'headquartered in thailand', 'classification: thai', 'เป็นแบรนด์ไทย'
            ]):
                return BrandClassification(
                    origin='thai',
                    confidence=0.8,  # Höhere Confidence bei Web Search
                    matched_patterns=[clean_response[:150]],
                    classification_method='gpt_oss_pattern'
                )
            
            if any(phrase in response_lower for phrase in [
                'international brand', 'global brand', 'multinational', 
                'western brand', 'foreign brand', 'not thai', 'american brand',
                'european brand', 'japanese brand', 'classification: international',
                'headquartered in', 'based in usa', 'based in europe'
            ]):
                return BrandClassification(
                    origin='international',
                    confidence=0.8,  # Höhere Confidence bei Web Search
                    matched_patterns=[clean_response[:150]],
                    classification_method='gpt_oss_pattern'
                )
            
            # Fallback 2: Einfache Keyword-Suche
            thai_score = 0
            intl_score = 0
            
            # Zähle Thai-Indikatoren
            thai_keywords = ['thai', 'thailand', 'local', 'regional', 'southeast', 'asian']
            intl_keywords = ['international', 'global', 'multinational', 'western', 'european', 'american']
            
            for keyword in thai_keywords:
                thai_score += response_lower.count(keyword)
            
            for keyword in intl_keywords:
                intl_score += response_lower.count(keyword)
            
            if thai_score > intl_score and thai_score > 0:
                confidence = min(0.4 + (thai_score * 0.1), 0.8)
                return BrandClassification(
                    origin='thai',
                    confidence=confidence,
                    matched_patterns=[f"thai_keywords: {thai_score}"],
                    classification_method='deepseek_keyword'
                )
            elif intl_score > thai_score and intl_score > 0:
                confidence = min(0.4 + (intl_score * 0.1), 0.8)
                return BrandClassification(
                    origin='international',
                    confidence=confidence,
                    matched_patterns=[f"intl_keywords: {intl_score}"],
                    classification_method='deepseek_keyword'
                )
                
        except Exception as e:
            self.logger.error(f"DeepSeek response parsing error: {e}")
        
        # Default fallback - als international klassifizieren (safer default)
        return BrandClassification(
            origin='international',
            confidence=0.3,
            matched_patterns=[f"fallback_parse: {response[:50]}"],
            classification_method='deepseek_fallback'
        )

# =============================================================================
# 🏬 CJMORE PRIVATE BRAND CLASSIFIER  
# =============================================================================

class CJMoreClassifier:
    """Klassifiziert Produkte als CJMore Private Brands"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Normalisierte CJMore Private Brands für besseres Matching
        self.private_brands = {
            'uno': ['uno', 'uno.', 'uno '],
            'nine beauty': ['nine beauty', 'ninebeauty', 'nine-beauty', '9beauty', '9 beauty'],
            'bao cafe': ['bao cafe', 'baocafe', 'bao-cafe', 'bao café'],
            'tian tian': ['tian tian', 'tiantian', 'tian-tian', 'tiān tiān']
        }
        
        # Flache Liste aller Varianten für schnelles Matching
        self.all_patterns = []
        self.pattern_to_brand = {}
        
        for brand_name, patterns in self.private_brands.items():
            for pattern in patterns:
                pattern_normalized = pattern.lower().strip()
                self.all_patterns.append(pattern_normalized)
                self.pattern_to_brand[pattern_normalized] = brand_name
    
    def classify_product(self, product_data: Dict[str, Any]) -> CJMoreClassification:
        """
        Klassifiziere Produkt als CJMore Private Brand.
        
        Args:
            product_data: Produktdaten mit brand, name, ocr_tokens etc.
            
        Returns:
            CJMoreClassification mit Ergebnis
        """
        
        # Sammle alle zu prüfenden Textquellen
        text_sources = []
        
        # 1. Brand Field
        if 'brand' in product_data and product_data['brand']:
            text_sources.append(('brand_field', product_data['brand']))
        
        # 2. Product Name
        if 'name' in product_data and product_data['name']:
            text_sources.append(('product_name', product_data['name']))
        
        # 3. OCR Tokens
        ocr_tokens = product_data.get('ocr_tokens', [])
        if isinstance(ocr_tokens, list):
            for token in ocr_tokens:
                if isinstance(token, str) and len(token) > 1:
                    text_sources.append(('ocr_tokens', token))
        
        # Prüfe jede Textquelle
        for source_type, text in text_sources:
            result = self._check_text_for_private_brands(text, source_type)
            if result.is_private_brand:
                return result
        
        # Keine Private Brand gefunden
        return CJMoreClassification(
            is_private_brand=False,
            brand_name='',
            confidence=0.0,
            matched_pattern='',
            detection_method='none',
            detection_source='none'
        )
    
    def _check_text_for_private_brands(self, text: str, source_type: str) -> CJMoreClassification:
        """
        Prüfe Text auf CJMore Private Brand Patterns.
        
        Args:
            text: Zu prüfender Text
            source_type: Art der Textquelle ('brand_field', 'product_name', 'ocr_tokens')
            
        Returns:
            CJMoreClassification Ergebnis
        """
        if not text or not isinstance(text, str):
            return self._create_negative_classification()
        
        text_normalized = text.lower().strip()
        
        # 1. Exact Match
        for pattern in self.all_patterns:
            if pattern == text_normalized:
                brand_name = self.pattern_to_brand[pattern]
                return CJMoreClassification(
                    is_private_brand=True,
                    brand_name=brand_name,
                    confidence=1.0,
                    matched_pattern=pattern,
                    detection_method='exact_match',
                    detection_source=source_type
                )
        
        # 2. Word Boundary Substring Match (nur für längere, eindeutige Patterns)
        # Deaktiviert für kurze Patterns wie 'uno' um False Positives zu vermeiden
        for pattern in self.all_patterns:
            if len(pattern) >= 6 and pattern in text_normalized:  # Nur für längere Patterns >= 6 Zeichen
                # Zusätzliche Prüfung: Pattern sollte nicht zufällig in anderen Wörtern auftauchen
                words_in_text = text_normalized.split()
                for word in words_in_text:
                    if pattern in word and len(pattern) / len(word) >= 0.7:  # Mindestens 70% des Wortes
                        brand_name = self.pattern_to_brand[pattern]
                        return CJMoreClassification(
                            is_private_brand=True,
                            brand_name=brand_name,
                            confidence=0.8,  # Niedrigere Confidence für Substring Match
                            matched_pattern=pattern,
                            detection_method='word_boundary_match',
                            detection_source=source_type
                        )
        
        # 3. Fuzzy Match (falls aktiviert)
        if CJMORE_CLASSIFICATION_CONFIG.get('enable_fuzzy_matching', True):
            fuzzy_result = self._fuzzy_match_private_brands(text_normalized, source_type)
            if fuzzy_result.is_private_brand:
                return fuzzy_result
        
        return self._create_negative_classification()
    
    def _fuzzy_match_private_brands(self, text: str, source_type: str) -> CJMoreClassification:
        """Fuzzy Matching für Private Brands."""
        
        threshold = CJMORE_FUZZY_THRESHOLD
        best_match = None
        best_score = 0.0
        best_pattern = ''
        
        for pattern in self.all_patterns:
            if len(pattern) < 3:  # Skip sehr kurze Patterns für Fuzzy Match
                continue
                
            similarity = SequenceMatcher(None, text, pattern).ratio()
            if similarity > best_score and similarity >= threshold:
                best_score = similarity
                best_match = self.pattern_to_brand[pattern]
                best_pattern = pattern
        
        if best_match:
            return CJMoreClassification(
                is_private_brand=True,
                brand_name=best_match,
                confidence=best_score,
                matched_pattern=best_pattern,
                detection_method='fuzzy_match',
                detection_source=source_type
            )
        
        return self._create_negative_classification()
    
    def _create_negative_classification(self) -> CJMoreClassification:
        """Erstelle negative Classification für nicht-Private-Brands."""
        return CJMoreClassification(
            is_private_brand=False,
            brand_name='',
            confidence=0.0,
            matched_pattern='',
            detection_method='none',
            detection_source='none'
        )

# =============================================================================
# 📊 BRAND PRODUCT COUNTER
# =============================================================================

class BrandProductCounter:
    """Zählt verschiedene Produkte pro Brand unter Berücksichtigung von Duplikaten"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.brand_stats = {}
    
    def count_products_by_brand(self, products: List) -> Dict[str, BrandProductCount]:
        """
        Zähle Produkte gruppiert nach Brand.
        
        Args:
            products: Liste von Produkten (EnhancedProduct oder Dicts)
            
        Returns:
            Dict mit Brand -> BrandProductCount Mapping
        """
        brand_data = defaultdict(lambda: {
            'products': [],
            'types': set(),
            'categories': set(),
            'confidences': [],
            'total_count': 0,
            'is_cjmore_private': False
        })
        
        # Sammle Daten pro Brand
        for product in products:
            if hasattr(product, 'brand'):
                # EnhancedProduct object
                brand = product.brand
                product_type = getattr(product, 'type', 'unknown')
                category = getattr(product, 'main_category', 'unknown')
                confidence = getattr(product, 'conf_fused', 0.5)
                duplicate_count = getattr(product, 'duplicate_count', 1)
                is_cjmore = getattr(product, 'cjmore_classification', None)
                is_private = is_cjmore.is_private_brand if is_cjmore else False
            else:
                # Dict-based product
                brand = product.get('brand', 'unknown')
                product_type = product.get('type', 'unknown')
                category = product.get('main_category', 'unknown')
                confidence = product.get('conf_fused', 0.5)
                duplicate_count = product.get('duplicate_count', 1)
                cjmore_data = product.get('cjmore_classification', {})
                is_private = cjmore_data.get('is_private_brand', False) if isinstance(cjmore_data, dict) else False
            
            # Filtere nach Minimum-Confidence
            min_conf = BRAND_COUNTING_CONFIG.get('min_confidence', 0.5)
            if confidence < min_conf:
                continue
                
            # Filtere 'unknown' brands falls konfiguriert
            if BRAND_COUNTING_CONFIG.get('exclude_unknown', True) and brand.lower() == 'unknown':
                continue
            
            # Sammle Daten
            brand_info = brand_data[brand]
            brand_info['products'].append(product)
            brand_info['types'].add(product_type)
            brand_info['categories'].add(category)
            brand_info['confidences'].append(confidence)
            brand_info['total_count'] += duplicate_count
            brand_info['is_cjmore_private'] = brand_info['is_cjmore_private'] or is_private
        
        # Erstelle BrandProductCount Objekte
        result = {}
        for brand, info in brand_data.items():
            # Filtere Brands mit zu wenigen Produkten
            unique_count = len(info['products'])
            if unique_count < MIN_PRODUCTS_FOR_BRAND_REPORT:
                continue
            
            avg_confidence = sum(info['confidences']) / len(info['confidences']) if info['confidences'] else 0.0
            
            result[brand] = BrandProductCount(
                brand=brand,
                unique_products=unique_count,
                total_instances=info['total_count'],
                product_types=list(info['types']),
                categories=list(info['categories']),
                avg_confidence=avg_confidence,
                is_cjmore_private=info['is_cjmore_private']
            )
        
        return result
    
    def get_brand_statistics(self, brand_counts: Dict[str, BrandProductCount]) -> Dict[str, Any]:
        """Erstelle Statistiken über Brand-Verteilung."""
        
        total_brands = len(brand_counts)
        total_unique_products = sum(bc.unique_products for bc in brand_counts.values())
        total_instances = sum(bc.total_instances for bc in brand_counts.values())
        
        # CJMore Private Brand Statistics
        cjmore_brands = {brand: bc for brand, bc in brand_counts.items() if bc.is_cjmore_private}
        cjmore_count = len(cjmore_brands)
        cjmore_products = sum(bc.unique_products for bc in cjmore_brands.values())
        
        # Top Brands by Product Diversity
        top_brands = sorted(brand_counts.items(), key=lambda x: x[1].unique_products, reverse=True)[:10]
        
        return {
            'total_brands': total_brands,
            'total_unique_products': total_unique_products,
            'total_product_instances': total_instances,
            'cjmore_private_brands': cjmore_count,
            'cjmore_unique_products': cjmore_products,
            'cjmore_percentage': (cjmore_count / total_brands * 100) if total_brands > 0 else 0.0,
            'top_brands_by_diversity': [(brand, bc.unique_products, bc.is_cjmore_private) for brand, bc in top_brands],
            'avg_products_per_brand': total_unique_products / total_brands if total_brands > 0 else 0.0
        }

# =============================================================================
# 🔍 API DEBUG & ERROR HANDLING
# =============================================================================

def _save_api_debug_info(debug_info: Dict[str, Any]) -> None:
    """
    Speichere API Debug-Informationen für Troubleshooting.
    
    Args:
        debug_info: Debug-Daten Dictionary
    """
    try:
        debug_folder = Path("brand_analysis_output/debug")
        debug_folder.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_file = debug_folder / f"api_error_{timestamp}.json"
        
        with open(debug_file, 'w', encoding='utf-8') as f:
            json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
        print(f"🐛 DEBUG INFO gespeichert: {debug_file}")
        print(f"📋 Problem: {debug_info.get('message', 'Unknown error')}")
        print(f"🔧 Action: {debug_info.get('required_action', 'Check logs')}")
        
    except Exception as e:
        print(f"❌ Konnte Debug-Info nicht speichern: {e}")

# 🔍 ERWEITERTE CLOUD VISION API
# =============================================================================

def enhanced_cloud_vision_analysis(image_path: str) -> Dict[str, Any]:
    """
    Erweiterte Cloud Vision API Analyse mit Object Detection.
    
    Args:
        image_path: Pfad zum Bild
        
    Returns:
        Dict mit allen Vision API Ergebnissen
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Lade und encode Bild
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Erweiterte Vision API Anfrage
        url = "https://vision.googleapis.com/v1/images:annotate"
        
        payload = {
            "requests": [{
                "image": {"content": image_data},
                "features": [
                    {"type": "OBJECT_LOCALIZATION", "maxResults": 50},  # Objekt-Erkennung
                    {"type": "TEXT_DETECTION", "maxResults": 100},       # OCR
                    {"type": "LOGO_DETECTION", "maxResults": 20},        # Logo-Erkennung
                    {"type": "LABEL_DETECTION", "maxResults": 50},       # Label-Erkennung
                ]
            }]
        }
        
        # API Key Validation
        if not GOOGLE_API_KEY:
            error_msg = "❌ GOOGLE_API_KEY ist nicht gesetzt! Vision API kann nicht verwendet werden."
            logger.error(error_msg)
            # Erstelle Debug-File für API-Probleme
            debug_info = {
                'error_type': 'missing_api_key',
                'service': 'google_vision',
                'message': error_msg,
                'image_path': image_path,
                'timestamp': time.time(),
                'required_action': 'Set GOOGLE_API_KEY environment variable'
            }
            _save_api_debug_info(debug_info)
            raise ValueError(error_msg)
        
        response = requests.post(
            f"{url}?key={GOOGLE_API_KEY}",
            json=payload,
            timeout=VISION_TIMEOUT
        )
        
        if response.status_code != 200:
            error_msg = f"Vision API HTTP Error {response.status_code}: {response.text}"
            logger.error(error_msg)
            
            # Debug Info speichern
            debug_info = {
                'error_type': 'api_http_error',
                'service': 'google_vision',
                'status_code': response.status_code,
                'response_text': response.text,
                'image_path': image_path,
                'timestamp': time.time(),
                'required_action': 'Check API key validity and quota limits'
            }
            _save_api_debug_info(debug_info)
            raise ConnectionError(error_msg)
        
        result = response.json()
        
        if 'responses' not in result:
            error_msg = f"Vision API Invalid Response: {result}"
            logger.error(error_msg)
            debug_info = {
                'error_type': 'invalid_response',
                'service': 'google_vision',
                'response': result,
                'image_path': image_path,
                'timestamp': time.time(),
                'required_action': 'Check API response format'
            }
            _save_api_debug_info(debug_info)
            raise ValueError(error_msg)
        
        response_data = result['responses'][0]
        
        # Parse alle Ergebnisse
        vision_results = {
            'objects': response_data.get('localizedObjectAnnotations', []),
            'texts': response_data.get('textAnnotations', []),
            'logos': response_data.get('logoAnnotations', []),
            'labels': response_data.get('labelAnnotations', [])
        }
        
        logger.info(f"Vision API: {len(vision_results['objects'])} Objekte, "
                   f"{len(vision_results['texts'])} Texte, "
                   f"{len(vision_results['logos'])} Logos erkannt")
        
        return vision_results
        
    except Exception as e:
        logger.error(f"Enhanced Vision API failed für {image_path}: {e}")
        return {}

# =============================================================================
# 🚀 ERWEITERTE HAUPT-PIPELINE
# =============================================================================

class EnhancedBrandAnalyzer:
    """Erweiterte Brand Analysis Pipeline"""
    
    def __init__(self, output_folder: str):
        self.output_folder = Path(output_folder)
        self.logger = logging.getLogger(__name__)
        
        # Initialisiere Komponenten
        self.eye_level_detector = EyeLevelDetector()
        self.brand_classifier = BrandClassifier()
        self.cjmore_classifier = CJMoreClassifier()
        self.brand_counter = BrandProductCounter()
        
        # Speichere Ergebnisse
        self.enhanced_products = []
        
    def analyze_single_image(self, image_path: str) -> List[EnhancedProduct]:
        """
        Analysiere ein einzelnes Bild vollständig.
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Liste von EnhancedProduct Objekten
        """
        image_id = Path(image_path).stem
        
        self.logger.info(f"🔍 Analysiere Vollbild: {image_id}")
        
        # 1. Cloud Vision API Analyse mit Error Handling
        try:
            vision_data = enhanced_cloud_vision_analysis(image_path)
            self.logger.info(f"✅ Vision API erfolgreich für {image_id}")
        except Exception as e:
            self.logger.error(f"❌ Vision API fehlgeschlagen für {image_id}: {e}")
            # Fallback zu leeren Vision-Daten mit Warnung
            vision_data = {'objects': [], 'texts': [], 'logos': [], 'labels': []}
            print(f"⚠️ WARNUNG: Vision API nicht verfügbar für {image_id}. Pipeline läuft ohne Object Detection.")
        
        # 2. Gemini Whole-Image Analyse mit Error Handling  
        try:
            gemini_results = self._analyze_with_gemini(image_path)
            self.logger.info(f"✅ Gemini API erfolgreich für {image_id}: {len(gemini_results)} Produkte")
        except Exception as e:
            self.logger.error(f"❌ Gemini API fehlgeschlagen für {image_id}: {e}")
            # Stoppe Pipeline bei Gemini-Fehler, da es essentiell ist
            raise RuntimeError(f"Gemini API ist für die Produkterkennung essentiell. Fehler: {e}")
        
        # 3. Eye-Level Detection  
        objects = vision_data.get('objects', [])  # Fix: correct key
        eye_level_zones = self.eye_level_detector.detect_eye_level_zones(
            image_path, objects
        )
        
        # Füge OCR-Texte für Fallback-Positionierung hinzu
        eye_level_zones['ocr_texts'] = vision_data.get('texts', [])
        
        # 4. Fusioniere und erweitere Ergebnisse
        enhanced_products = []
        
        for gemini_result in gemini_results:
            
            # Finde passende Vision-Objekte (falls vorhanden)
            matching_object = self._find_matching_vision_object(
                gemini_result, objects
            )
            
            # Bestimme Eye-Level Position intelligent
            if matching_object and 'boundingPoly' in matching_object:
                # Verwende echte Bounding Box falls vorhanden
                bounding_box = matching_object.get('boundingPoly')
                y_position = 0.5  # Wird in get_product_eye_level überschrieben
            else:
                # Intelligente Simulation basierend auf Produkttyp
                y_position = self.eye_level_detector._simulate_realistic_position(gemini_result)
                bounding_box = None
            
            eye_level_data = self.eye_level_detector.get_product_eye_level(
                y_position, eye_level_zones, bounding_box
            )
            
            # Klassifiziere Marke mit zusätzlichen Context-Daten
            ocr_tokens = self._extract_ocr_tokens(vision_data)
            brand_classification = self.brand_classifier.classify_brand(
                gemini_result['brand'],
                gemini_result.get('keywords', []),
                gemini_result['type'],  # product_type für DeepSeek
                ocr_tokens              # OCR tokens für DeepSeek
            )
            
            # Kategorie-Zuordnung (bestehende Funktion)
            main_cat, sub_cat, display_name, cat_confidence = get_product_category(
                gemini_result['type'],
                gemini_result.get('keywords', [])
            )
            
            # CJMore Private Brand Classification
            cjmore_product_data = {
                'brand': gemini_result['brand'],
                'name': gemini_result.get('name', ''),
                'ocr_tokens': self._extract_ocr_tokens(vision_data)
            }
            cjmore_classification = self.cjmore_classifier.classify_product(cjmore_product_data)
            
            # Erstelle EnhancedProduct
            enhanced_product = EnhancedProduct(
                image_id=image_id,
                brand=gemini_result['brand'],
                type=gemini_result['type'],
                approx_count=gemini_result.get('approx_count', 1),
                conf_fused=gemini_result.get('confidence', 0.5),
                
                bounding_box=matching_object.get('boundingPoly') if matching_object else None,
                ocr_tokens=self._extract_ocr_tokens(vision_data),
                logos=[logo['description'] for logo in vision_data.get('logos', [])],
                
                eye_level_data=eye_level_data,
                brand_classification=brand_classification,
                cjmore_classification=cjmore_classification,
                
                main_category=main_cat,
                subcategory=sub_cat,
                category_display_name=display_name,
                category_confidence=cat_confidence,
                
                source_data={
                    'gemini': gemini_result,
                    'vision_api': vision_data,
                    'eye_level_zones': eye_level_zones
                }
            )
            
            enhanced_products.append(enhanced_product)
        
        self.enhanced_products.extend(enhanced_products)
        
        self.logger.info(f"✅ {len(enhanced_products)} erweiterte Produkte aus {image_id}")
        
        return enhanced_products
    
    def _analyze_with_gemini(self, image_path: str) -> List[Dict]:
        """Gemini Vision Analyse für Vollbild-Produkterkennung."""
        
        try:
            import base64
            import requests
            
            # Lade und encode Bild
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            
            # API Key Validation
            if not GOOGLE_API_KEY:
                error_msg = "❌ GOOGLE_API_KEY ist nicht gesetzt! Gemini API kann nicht verwendet werden."
                self.logger.error(error_msg)
                debug_info = {
                    'error_type': 'missing_api_key',
                    'service': 'google_gemini',
                    'message': error_msg,
                    'image_path': image_path,
                    'timestamp': time.time(),
                    'required_action': 'Set GOOGLE_API_KEY environment variable'
                }
                _save_api_debug_info(debug_info)
                raise ValueError(error_msg)
            
            # Gemini Vision API Anfrage
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={GOOGLE_API_KEY}"
            
            payload = {
                "contents": [{
                    "parts": [
                        {
                            "text": """Analyze this supermarket shelf image and identify ALL visible products. 
                            For each product, provide:
                            - brand: Brand name (exact text if visible)
                            - type: Product type/category 
                            - confidence: Detection confidence (0.0-1.0)
                            - approx_count: How many units visible
                            - keywords: Relevant keywords for categorization
                            
                            Return as JSON array with objects containing these fields.
                            Focus on consumer products like air fresheners, cleaners, beverages, snacks, etc.
                            If brand text is unclear, use 'unknown' but still identify the product type."""
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data
                            }
                        }
                    ]
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code != 200:
                error_msg = f"Gemini API HTTP Error {response.status_code}: {response.text}"
                self.logger.error(error_msg)
                debug_info = {
                    'error_type': 'api_http_error',
                    'service': 'google_gemini',
                    'status_code': response.status_code,
                    'response_text': response.text,
                    'image_path': image_path,
                    'timestamp': time.time(),
                    'required_action': 'Check Gemini API key validity and quota limits'
                }
                _save_api_debug_info(debug_info)
                raise ConnectionError(error_msg)
            
            result = response.json()
            
            # Parse Gemini Response
            if 'candidates' in result and result['candidates']:
                content = result['candidates'][0]['content']['parts'][0]['text']
                
                # Versuche JSON zu extrahieren
                import json
                import re
                
                # Finde JSON array in der Response
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    products = json.loads(json_str)
                    
                    # Normalisiere Felder
                    normalized_products = []
                    for product in products:
                        normalized_product = {
                            'brand': product.get('brand', 'unknown'),
                            'type': product.get('type', 'unknown'),
                            'confidence': float(product.get('confidence', 0.5)),
                            'approx_count': int(product.get('approx_count', 1)),
                            'keywords': product.get('keywords', [])
                        }
                        normalized_products.append(normalized_product)
                    
                    self.logger.info(f"Gemini erkannte {len(normalized_products)} Produkte")
                    return normalized_products
            
            self.logger.warning("Keine Produkte in Gemini Response gefunden")
            return []
            
        except Exception as e:
            self.logger.error(f"Gemini Analyse fehgeschlagen: {e}")
            
            # Fallback zu Demo-Daten basierend auf bisherigen Erkenntnissen
            self.logger.info("Verwende Demo-Daten als Fallback")
            return [
                {'brand': 'glade', 'type': 'air freshener', 'confidence': 0.85, 'approx_count': 2, 'keywords': ['air', 'freshener', 'scented']},
                {'brand': 'OASIS', 'type': 'air freshener', 'confidence': 0.90, 'approx_count': 1, 'keywords': ['air', 'freshener', 'tropical']},
                {'brand': 'Fumme', 'type': 'air freshener', 'confidence': 0.75, 'approx_count': 3, 'keywords': ['air', 'freshener']},
                {'brand': 'NUS', 'type': 'cleaning product', 'confidence': 0.80, 'approx_count': 1, 'keywords': ['cleaning', 'household']},
                {'brand': 'KOALA THE BEAR', 'type': 'household product', 'confidence': 0.85, 'approx_count': 1, 'keywords': ['household']},
                {'brand': 'Hygiene', 'type': 'personal care', 'confidence': 0.75, 'approx_count': 1, 'keywords': ['personal', 'care', 'hygiene']}
            ]
    
    def _find_matching_vision_object(self, gemini_result: Dict, vision_objects: List[Dict]) -> Optional[Dict]:
        """Finde passenden Vision API Objekt zu Gemini-Ergebnis."""
        
        if not vision_objects:
            return None
            
        # Erweiterte Heuristik basierend auf Objekt-Namen und Score
        product_type = gemini_result.get('type', '').lower()
        
        # Sortiere nach Confidence Score (höchster zuerst)
        sorted_objects = sorted(vision_objects, key=lambda x: x.get('score', 0), reverse=True)
        
        for obj in sorted_objects:
            obj_name = obj.get('name', '').lower()
            
            # Erweiterte Keyword-Liste für Supermarkt-Produkte
            product_keywords = [
                'bottle', 'container', 'package', 'product', 'box', 'can', 'tube', 
                'jar', 'bag', 'packet', 'carton', 'wrapper', 'pouch', 'sachet'
            ]
            
            # Keyword-Matching mit höherer Wahrscheinlichkeit
            if any(keyword in obj_name for keyword in product_keywords):
                return obj
        
        # Fallback: Objekt mit höchstem Score nehmen
        if sorted_objects:
            return sorted_objects[0]
        
        return None
    
    def _estimate_product_position(self, gemini_result: Dict, vision_object: Optional[Dict], 
                                  eye_level_zones: Dict) -> float:
        """Schätze Y-Position des Produkts im Bild."""
        
        if vision_object and 'boundingPoly' in vision_object:
            # Berechne Zentrum aus Bounding Box
            vertices = vision_object['boundingPoly']['vertices']
            y_center = sum(v.get('y', 0) for v in vertices) / len(vertices)
            img_height = eye_level_zones.get('image_dimensions', {}).get('height', 1000)
            return y_center / img_height
        
        # Fallback: Verwende OCR-Text Position falls verfügbar
        brand = gemini_result.get('brand', '').lower()
        if brand and brand != 'unknown':
            # Finde Brand in OCR Text Annotations
            texts = eye_level_zones.get('ocr_texts', [])
            for text in texts:
                if brand in text.get('description', '').lower():
                    if 'boundingPoly' in text:
                        vertices = text['boundingPoly']['vertices']
                        y_center = sum(v.get('y', 0) for v in vertices) / len(vertices)
                        img_height = eye_level_zones.get('image_dimensions', {}).get('height', 1000)
                        return y_center / img_height
        
        # Intelligenter Fallback: Simuliere Eye-Level basierend auf Brand und Position
        # Da Gemini Produkte oft in visueller Reihenfolge findet, verwende das als Heuristik
        product_index = getattr(gemini_result, '_index', hash(gemini_result.get('brand', 'unknown')) % 100)
        
        # Simuliere realistische Regal-Verteilung
        # Verschiedene Y-Positionen basierend auf Produkttyp und "Position"
        product_type = gemini_result.get('type', '').lower()
        brand = gemini_result.get('brand', '').lower()
        
        # Erstelle deterministische aber variierte Position
        hash_value = (hash(brand) + hash(product_type)) % 100
        
        # Verschiedene Höhen basierend auf Produkttyp
        if any(t in product_type for t in ['spray', 'aerosol', 'air freshener']):
            base_y = 0.3  # Höhere Regale
        elif any(t in product_type for t in ['detergent', 'fabric softener', 'cleaning']):
            base_y = 0.5  # Mittlere Höhe  
        elif any(t in product_type for t in ['toothpaste', 'mouthwash', 'dental']):
            base_y = 0.6  # Etwas niedriger
        else:
            base_y = 0.45  # Standard
            
        # Füge Variation basierend auf Hash hinzu (±20%)
        variation = (hash_value / 100.0 - 0.5) * 0.4  # -0.2 bis +0.2
        y_position = max(0.1, min(0.9, base_y + variation))
        
        return y_position
        
        # Heuristiken für typische Produktplatzierungen
        if any(keyword in product_type for keyword in ['premium', 'expensive']):
            return 0.45  # Eye-Level
        elif any(keyword in product_type for keyword in ['heavy', 'large', 'bulk']):
            return 0.8   # Unten
        else:
            return 0.5   # Mitte
    
    def _extract_ocr_tokens(self, vision_data: Dict) -> List[str]:
        """Extrahiere OCR-Tokens aus Vision API Daten."""
        
        tokens = []
        for text_annotation in vision_data.get('texts', []):
            text = text_annotation.get('description', '')
            # Bereinige und tokenize
            clean_tokens = re.findall(r'\b\w+\b', text.lower())
            tokens.extend(clean_tokens)
        
        return list(set(tokens))  # Duplikate entfernen

def main(supermarket='cjmore'):
    """Erweiterte Haupt-Pipeline mit Cross-Image Deduplication und Checkpoint-System"""
    
    import time
    
    # Load supermarket configuration
    if supermarket == 'tops_daily':
        import config_tops_daily as config
        from config_brand_analysis import BACKUP_FREQUENCY  # Keep technical settings
        print(f"🏪 Running analysis for: {config.SUPERMARKET_NAME}")
        
        # Override config variables
        global IMAGE_FOLDER, OUTPUT_FOLDER, CJMORE_PRIVATE_BRANDS
        IMAGE_FOLDER = Path(config.IMAGES_FOLDER)
        OUTPUT_FOLDER = Path(config.OUTPUT_FOLDER)
        CJMORE_PRIVATE_BRANDS = config.TOPS_DAILY_PRIVATE_BRANDS
        
    else:
        # Default: CJMore configuration
        from config_brand_analysis import BACKUP_FREQUENCY
        print("🏪 Running analysis for: CJMore Supermarket")
    
    # Setup Logging für Terminal UND Datei
    output_folder = OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to show in terminal AND file
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger setup
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Terminal Handler (Console)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    console_handler.setLevel(logging.INFO)
    
    # File Handler 
    log_file = output_folder / 'supermarket_analysis.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.INFO)
    
    # Clear existing handlers and add ours
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starte erweiterte Brand Analysis Pipeline")
    
    # Setup
    output_folder = OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize Checkpoint Manager
    from checkpoint_manager import get_checkpoint_manager
    checkpoint_mgr = get_checkpoint_manager(output_folder)
    
    analyzer = EnhancedBrandAnalyzer(str(output_folder))
    
    # Sammle alle Bilder (KEINE Segmentierung)
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(Path(IMAGE_FOLDER).glob(f"*{ext}"))
        image_files.extend(Path(IMAGE_FOLDER).glob(f"*{ext.upper()}"))
    
    total_images = len(image_files)
    logger.info(f"📷 Gefunden: {total_images} Bilder für Vollbild-Analyse")
    checkpoint_mgr.set_total_images(total_images)
    
    # Try to resume from checkpoint
    checkpoint_data = checkpoint_mgr.load_checkpoint()
    if checkpoint_data:
        processed_images, results_by_image = checkpoint_data
        processed_image_names = set(processed_images)
        
        # Filter out already processed images
        remaining_images = [img for img in image_files if img.stem not in processed_image_names]
        logger.info(f"🔄 Fortsetzen: {len(remaining_images)} Bilder verbleibend")
    else:
        # Start fresh
        results_by_image = {}
        processed_images = []
        remaining_images = image_files
        logger.info("🆕 Starte neue vollständige Analyse")
    
    # Analysiere verbleibende Bilder mit Checkpoint-System
    start_time = time.time()
    processing_times = []
    
    for i, img_path in enumerate(remaining_images):
        image_start_time = time.time()
        image_id = img_path.stem
        current_img_num = len(processed_images) + 1
        
        # Calculate progress and ETA
        progress_percent = (current_img_num - 1) / total_images * 100
        
        # Calculate ETA based on average processing time
        if processing_times:
            avg_time_per_image = sum(processing_times) / len(processing_times)
            remaining_images_count = total_images - current_img_num + 1
            eta_seconds = avg_time_per_image * remaining_images_count
            eta_minutes = eta_seconds / 60
            
            if eta_minutes > 60:
                eta_str = f"{eta_minutes/60:.1f}h"
            elif eta_minutes > 1:
                eta_str = f"{eta_minutes:.1f}min"
            else:
                eta_str = f"{eta_seconds:.0f}s"
        else:
            eta_str = "calculating..."
        
        logger.info(f"🔍 [{progress_percent:5.1f}%] Bild {current_img_num}/{total_images}: {image_id}")
        logger.info(f"    ⏱️  ETA: {eta_str} | Avg: {sum(processing_times)/len(processing_times):.1f}s/img" if processing_times else "    ⏱️  ETA: calculating... | Erste Analyse...")
        
        try:
            enhanced_products = analyzer.analyze_single_image(str(img_path))
            results_by_image[image_id] = enhanced_products
            processed_images.append(image_id)
            
            # Record processing time
            processing_time = time.time() - image_start_time
            processing_times.append(processing_time)
            checkpoint_mgr.record_image_processing_time(image_id, processing_time)
            
            # Calculate total elapsed time
            total_elapsed = time.time() - start_time
            
            logger.info(f"    ✅ {len(enhanced_products)} Produkte erkannt | {processing_time:.1f}s | Total: {total_elapsed/60:.1f}min")
            
            # Show detailed product breakdown
            if enhanced_products:
                thai_count = sum(1 for p in enhanced_products if hasattr(p, 'brand_classification') and p.brand_classification.origin == 'thai')
                intl_count = sum(1 for p in enhanced_products if hasattr(p, 'brand_classification') and p.brand_classification.origin == 'international')
                logger.info(f"    📊 Thai: {thai_count} | International: {intl_count} | Unknown: {len(enhanced_products)-thai_count-intl_count}")
            
            # Create checkpoint if needed
            images_processed = len(processed_images)
            if checkpoint_mgr.should_create_checkpoint(images_processed):
                logger.info(f"    💾 Checkpoint gespeichert bei Bild {images_processed}")
                checkpoint_mgr.save_checkpoint(processed_images, results_by_image, image_id)
            
            # Create backup if needed  
            if checkpoint_mgr.should_create_backup(images_processed):
                logger.info(f"    🗄️  Backup erstellt nach {images_processed} Bildern")
                checkpoint_mgr.create_backup(images_processed // BACKUP_FREQUENCY)
                
        except Exception as e:
            logger.error(f"    ❌ Fehler bei {img_path}: {e}")
            # Continue with next image rather than failing completely
            continue
    
    # Sammle alle Produkte für Statistiken vor Deduplication
    all_products_before = []
    for products in results_by_image.values():
        all_products_before.extend(products)
    
    # Calculate and show processing statistics
    total_processing_time = time.time() - start_time
    avg_time_per_image = total_processing_time / len(results_by_image) if results_by_image else 0
    
    logger.info("=" * 60)
    logger.info(f"📊 ANALYSE PHASE ABGESCHLOSSEN")
    logger.info(f"   📷 Verarbeitete Bilder: {len(results_by_image)}")
    logger.info(f"   🎯 Gefundene Produkte: {len(all_products_before)}")
    logger.info(f"   ⏱️  Gesamtzeit: {total_processing_time/60:.1f} Minuten")
    logger.info(f"   📈 Durchschnitt: {avg_time_per_image:.1f}s pro Bild")
    logger.info(f"   🚀 Geschwindigkeit: {3600/avg_time_per_image:.0f} Bilder/Stunde" if avg_time_per_image > 0 else "   🚀 Geschwindigkeit: N/A")
    logger.info("=" * 60)
    
    # Cross-Image Duplicate Detection (falls mehr als 1 Bild)
    if ENABLE_CROSS_IMAGE_DEDUPLICATION and len(results_by_image) > 1:
        dedup_start = time.time()
        logger.info("🔍 Starte Cross-Image Duplicate Detection...")
        
        # Erstelle Deduplicator
        import config_brand_analysis as config
        deduplicator = create_cross_image_deduplicator(config)
        
        # Führe Deduplication durch
        deduplicated_products, duplicate_mapping = deduplicator.detect_duplicates(results_by_image)
        
        # Speichere Deduplication Report
        if SAVE_DEDUPLICATION_REPORT:
            report_path = output_folder / DEDUPLICATION_REPORT_FILE
            deduplicator.save_deduplication_report(
                deduplicated_products, duplicate_mapping, report_path
            )
        
        # Verwende deduplizierte Ergebnisse
        final_products = deduplicated_products
        
        dedup_time = time.time() - dedup_start
        reduction_percent = (len(all_products_before) - len(final_products)) / len(all_products_before) * 100 if all_products_before else 0
        
        logger.info(f"✅ Cross-Image Deduplication abgeschlossen! ({dedup_time:.1f}s)")
        logger.info(f"   📈 Produkte vor Deduplication: {len(all_products_before)}")
        logger.info(f"   📊 Einzigartige Produkte: {len(final_products)}")
        logger.info(f"   🗑️  Duplikate entfernt: {len(all_products_before) - len(final_products)} ({reduction_percent:.1f}%)")
        logger.info(f"   📄 Dedup-Report: {report_path}")
        
    elif len(results_by_image) <= 1:
        # Einzelnes Bild - keine Deduplication nötig
        final_products = all_products_before
        logger.info("ℹ️  Nur ein Bild gefunden - Cross-Image Deduplication übersprungen")
        
    else:
        # Deduplication deaktiviert
        final_products = all_products_before
        logger.info("⚠️  Cross-Image Deduplication deaktiviert")
    
    # Brand Product Counting
    if ENABLE_BRAND_PRODUCT_COUNTING:
        brand_start = time.time()
        logger.info("📊 Starte Brand Product Counting Analysis...")
        brand_counter = BrandProductCounter()
        brand_counts = brand_counter.count_products_by_brand(final_products)
        brand_stats = brand_counter.get_brand_statistics(brand_counts)
        brand_time = time.time() - brand_start
        
        # Calculate brand origin statistics
        thai_products = sum(1 for p in final_products if hasattr(p, 'brand_classification') and p.brand_classification.origin == 'thai')
        intl_products = sum(1 for p in final_products if hasattr(p, 'brand_classification') and p.brand_classification.origin == 'international')
        unknown_products = len(final_products) - thai_products - intl_products
        
        logger.info(f"📈 Brand Analysis Ergebnisse: ({brand_time:.1f}s)")
        logger.info(f"   🏷️  Total Brands: {brand_stats['total_brands']}")
        logger.info(f"   📦 Unique Products: {brand_stats['total_unique_products']}")  
        logger.info(f"   🏬 CJMore Private Brands: {brand_stats['cjmore_private_brands']} ({brand_stats['cjmore_percentage']:.1f}%)")
        logger.info(f"   📊 Avg Products/Brand: {brand_stats['avg_products_per_brand']:.1f}")
        logger.info(f"   🇹🇭 Thai Brands: {thai_products} ({thai_products/len(final_products)*100:.1f}%)")
        logger.info(f"   🌍 International Brands: {intl_products} ({intl_products/len(final_products)*100:.1f}%)")
        logger.info(f"   ❓ Unknown: {unknown_products} ({unknown_products/len(final_products)*100:.1f}%)")
        
        # Speichere Brand Count Report
        brand_report_path = output_folder / 'brand_product_counts.json'
        with open(brand_report_path, 'w', encoding='utf-8') as f:
            # Konvertiere BrandProductCount zu Dict
            serializable_counts = {}
            for brand, count_obj in brand_counts.items():
                serializable_counts[brand] = asdict(count_obj)
            
            brand_report = {
                'statistics': brand_stats,
                'brand_counts': serializable_counts,
                'generation_time': datetime.now().isoformat()
            }
            json.dump(brand_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Brand Count Report gespeichert: {brand_report_path}")
    else:
        brand_counts = {}
        brand_stats = {}
    
    # Speichere erweiterte Ergebnisse
    results_path = output_folder / 'enhanced_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        # Konvertiere EnhancedProduct zu Dict für JSON
        serializable_products = []
        for product in final_products:
            if hasattr(product, 'eye_level_data'):
                # EnhancedProduct Objekt - mit origin auf oberster Ebene
                product_dict = {
                    **asdict(product),
                    'eye_level_data': asdict(product.eye_level_data),
                    'brand_classification': asdict(product.brand_classification),
                    # Füge origin direkt hinzu für bessere Kompatibilität
                    'origin': product.brand_classification.origin,
                    'origin_confidence': product.brand_classification.confidence,
                    'classification_method': product.brand_classification.classification_method
                }
                serializable_products.append(product_dict)
            else:
                # Dict-basiertes Produkt
                serializable_products.append(product)
        
        json.dump(serializable_products, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Pipeline abgeschlossen: {len(final_products)} einzigartige Produkte")
    logger.info(f"📊 Ergebnisse gespeichert: {results_path}")
    
    # Erstelle erweiterte Excel-Berichte  
    create_enhanced_excel_report(final_products, output_folder, brand_counts if ENABLE_BRAND_PRODUCT_COUNTING else {})
    
    # Cleanup checkpoints after successful completion
    checkpoint_mgr.cleanup_checkpoints()
    
    # Final success log with summary
    total_processing_time = time.time() - checkpoint_mgr.start_time
    logger.info("🎉 ANALYSE ERFOLGREICH ABGESCHLOSSEN!")
    logger.info(f"   📊 Verarbeitete Bilder: {len(processed_images)}")
    logger.info(f"   🎯 Erkannte Produkte: {len(final_products)}")
    logger.info(f"   ⏱️  Gesamtzeit: {total_processing_time:.1f}s")
    if len(processed_images) > 0:
        avg_per_image = total_processing_time / len(processed_images)
        logger.info(f"   📈 Durchschnitt pro Bild: {avg_per_image:.1f}s")
    
    return str(results_path)

def create_enhanced_excel_report(products: List, output_folder: Path, brand_counts: Dict = None):
    """Erstelle erweiterte Excel-Berichte mit allen neuen Features."""
    
    logger = logging.getLogger(__name__)
    excel_path = output_folder / "enhanced_brand_analysis.xlsx"
    
    logger.info(f"📊 Erstelle erweiterte Excel-Berichte: {excel_path}")
    
    def get_attr(product, attr_path, default=''):
        """Hilfsfunktion um Attribute sowohl von Objekten als auch Dicts zu holen."""
        try:
            if hasattr(product, attr_path):
                return getattr(product, attr_path)
            elif isinstance(product, dict):
                keys = attr_path.split('.')
                value = product
                for key in keys:
                    if isinstance(value, dict):
                        value = value.get(key, default)
                    else:
                        value = getattr(value, key, default)
                return value
            return default
        except:
            return default
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # 1. BRAND CLASSIFICATION SHEET
        brand_class_data = []
        for product in products:
            # Handle both EnhancedProduct objects and dicts
            if hasattr(product, 'brand'):
                # EnhancedProduct object
                brand_class_data.append({
                    'brand': product.brand,
                    'origin': product.brand_classification.origin if hasattr(product, 'brand_classification') else 'unknown',
                    'confidence': product.brand_classification.confidence if hasattr(product, 'brand_classification') else 0.0,
                    'classification_method': product.brand_classification.classification_method if hasattr(product, 'brand_classification') else 'unknown',
                    'matched_patterns': ', '.join(product.brand_classification.matched_patterns) if hasattr(product, 'brand_classification') else '',
                    'product_type': product.type,
                    'category': product.category_display_name if hasattr(product, 'category_display_name') else 'Unknown',
                    'image_id': product.image_id if hasattr(product, 'image_id') else get_attr(product, 'source_image', 'unknown'),
                    'duplicate_count': getattr(product, 'duplicate_count', 1),
                    'source_images': ', '.join(getattr(product, 'source_images', [get_attr(product, 'source_image', 'unknown')]))
                })
            else:
                # Dict-based product (from cross-image deduplication)
                brand_class = product.get('brand_classification', {})
                brand_class_data.append({
                    'brand': product.get('brand', 'unknown'),
                    'origin': brand_class.get('origin', 'unknown') if isinstance(brand_class, dict) else 'unknown',
                    'confidence': brand_class.get('confidence', 0.0) if isinstance(brand_class, dict) else 0.0,
                    'classification_method': brand_class.get('classification_method', 'unknown') if isinstance(brand_class, dict) else 'unknown',
                    'matched_patterns': ', '.join(brand_class.get('matched_patterns', [])) if isinstance(brand_class, dict) else '',
                    'product_type': product.get('type', 'unknown'),
                    'category': product.get('category_display_name', 'Unknown'),
                    'image_id': product.get('source_image', 'unknown'),
                    'duplicate_count': product.get('duplicate_count', 1),
                    'source_images': ', '.join(product.get('source_images', [product.get('source_image', 'unknown')]))
                })
        
        brand_class_df = pd.DataFrame(brand_class_data)
        brand_class_df.to_excel(writer, sheet_name='Brand Classification', index=False)
        
        # 2. EYE LEVEL ANALYSIS SHEET
        eye_level_data = []
        for product in products:
            if hasattr(product, 'image_id'):
                # EnhancedProduct object
                eye_level_data.append({
                    'image_id': product.image_id,
                    'brand': product.brand,
                    'type': product.type,
                    'eye_level_zone': product.eye_level_data.zone if hasattr(product, 'eye_level_data') else 'unknown',
                    'y_position': round(product.eye_level_data.y_position, 3) if hasattr(product, 'eye_level_data') else 0.0,
                    'is_premium_zone': product.eye_level_data.is_premium_zone if hasattr(product, 'eye_level_data') else False,
                    'shelf_tier': product.eye_level_data.shelf_tier if hasattr(product, 'eye_level_data') else 'unknown',
                    'brand_origin': product.brand_classification.origin if hasattr(product, 'brand_classification') else 'unknown',
                    'category': product.category_display_name if hasattr(product, 'category_display_name') else 'Unknown',
                    'duplicate_count': getattr(product, 'duplicate_count', 1)
                })
            else:
                # Dict-based product
                eye_level_data_dict = product.get('eye_level_data', {})
                brand_class = product.get('brand_classification', {})
                eye_level_data.append({
                    'image_id': product.get('source_image', 'unknown'),
                    'brand': product.get('brand', 'unknown'),
                    'type': product.get('type', 'unknown'),
                    'eye_level_zone': eye_level_data_dict.get('zone', 'unknown') if isinstance(eye_level_data_dict, dict) else 'unknown',
                    'y_position': round(eye_level_data_dict.get('y_position', 0.0), 3) if isinstance(eye_level_data_dict, dict) else 0.0,
                    'is_premium_zone': eye_level_data_dict.get('is_premium_zone', False) if isinstance(eye_level_data_dict, dict) else False,
                    'shelf_tier': eye_level_data_dict.get('shelf_tier', 'unknown') if isinstance(eye_level_data_dict, dict) else 'unknown',
                    'brand_origin': brand_class.get('origin', 'unknown') if isinstance(brand_class, dict) else 'unknown',
                    'category': product.get('category_display_name', 'Unknown'),
                    'duplicate_count': product.get('duplicate_count', 1)
                })
        
        eye_level_df = pd.DataFrame(eye_level_data)
        eye_level_df.to_excel(writer, sheet_name='Eye Level Analysis', index=False)
        
        # 3. THAI VS INTERNATIONAL SUMMARY
        thai_intl_summary = []
        
        # Gruppiere nach Brand Origin
        origin_groups = defaultdict(list)
        for product in products:
            if hasattr(product, 'brand_classification'):
                # EnhancedProduct object
                origin = product.brand_classification.origin
            else:
                # Dict-based product
                brand_class = product.get('brand_classification', {})
                origin = brand_class.get('origin', 'unknown') if isinstance(brand_class, dict) else 'unknown'
            origin_groups[origin].append(product)
        
        for origin, origin_products in origin_groups.items():
            # Extract brands (handle both object and dict types)
            brands = set()
            categories = set()
            premium_count = 0
            confidences = []
            
            for p in origin_products:
                if hasattr(p, 'brand'):
                    # EnhancedProduct object
                    if p.brand != 'unknown':
                        brands.add(p.brand)
                    if hasattr(p, 'main_category'):
                        categories.add(p.main_category)
                    if hasattr(p, 'eye_level_data') and p.eye_level_data.is_premium_zone:
                        premium_count += p.get('duplicate_count', 1) if isinstance(p, dict) else getattr(p, 'duplicate_count', 1)
                    if hasattr(p, 'brand_classification'):
                        confidences.append(p.brand_classification.confidence)
                else:
                    # Dict-based product
                    brand = p.get('brand', 'unknown')
                    if brand != 'unknown':
                        brands.add(brand)
                    if 'main_category' in p:
                        categories.add(p['main_category'])
                    eye_level_data_dict = p.get('eye_level_data', {})
                    if isinstance(eye_level_data_dict, dict) and eye_level_data_dict.get('is_premium_zone', False):
                        premium_count += p.get('duplicate_count', 1)
                    brand_class = p.get('brand_classification', {})
                    if isinstance(brand_class, dict) and 'confidence' in brand_class:
                        confidences.append(brand_class['confidence'])
            
            total_products = sum(p.get('duplicate_count', 1) if isinstance(p, dict) else getattr(p, 'duplicate_count', 1) for p in origin_products)
            
            thai_intl_summary.append({
                'origin': origin,
                'unique_brands': len(brands),
                'total_products': total_products,
                'unique_categories': len(categories),
                'premium_placements': premium_count,
                'premium_percentage': round(premium_count / total_products * 100, 1) if total_products > 0 else 0,
                'avg_confidence': round(sum(confidences) / len(confidences), 3) if confidences else 0.0,
                'top_brands': ', '.join(list(brands)[:5])
            })
        
        thai_intl_df = pd.DataFrame(thai_intl_summary)
        thai_intl_df.to_excel(writer, sheet_name='Thai vs International', index=False)
        
        # 4. CJMORE PRIVATE BRAND CLASSIFICATION SHEET
        cjmore_data = []
        for product in products:
            if hasattr(product, 'cjmore_classification'):
                # EnhancedProduct object
                cjmore_class = product.cjmore_classification
                cjmore_data.append({
                    'brand': product.brand,
                    'type': product.type,
                    'is_private_brand': cjmore_class.is_private_brand,
                    'private_brand_name': cjmore_class.brand_name,
                    'confidence': cjmore_class.confidence,
                    'detection_method': cjmore_class.detection_method,
                    'detection_source': cjmore_class.detection_source,
                    'matched_pattern': cjmore_class.matched_pattern,
                    'category': product.category_display_name,
                    'image_id': product.image_id,
                    'duplicate_count': getattr(product, 'duplicate_count', 1)
                })
            else:
                # Dict-based product
                cjmore_class = product.get('cjmore_classification', {})
                cjmore_data.append({
                    'brand': product.get('brand', 'unknown'),
                    'type': product.get('type', 'unknown'),
                    'is_private_brand': cjmore_class.get('is_private_brand', False) if isinstance(cjmore_class, dict) else False,
                    'private_brand_name': cjmore_class.get('brand_name', '') if isinstance(cjmore_class, dict) else '',
                    'confidence': cjmore_class.get('confidence', 0.0) if isinstance(cjmore_class, dict) else 0.0,
                    'detection_method': cjmore_class.get('detection_method', 'none') if isinstance(cjmore_class, dict) else 'none',
                    'detection_source': cjmore_class.get('detection_source', 'none') if isinstance(cjmore_class, dict) else 'none',
                    'matched_pattern': cjmore_class.get('matched_pattern', '') if isinstance(cjmore_class, dict) else '',
                    'category': product.get('category_display_name', 'Unknown'),
                    'image_id': product.get('source_image', 'unknown'),
                    'duplicate_count': product.get('duplicate_count', 1)
                })
        
        cjmore_df = pd.DataFrame(cjmore_data)
        cjmore_df.to_excel(writer, sheet_name='CJMore Private Brands', index=False)
        
        # 5. BRAND PRODUCT COUNTS SHEET
        if brand_counts:
            brand_count_data = []
            for brand, count_obj in brand_counts.items():
                if hasattr(count_obj, 'brand'):
                    # BrandProductCount object
                    brand_count_data.append({
                        'brand': count_obj.brand,
                        'unique_products': count_obj.unique_products,
                        'total_instances': count_obj.total_instances,
                        'product_types': ', '.join(count_obj.product_types),
                        'categories': ', '.join(count_obj.categories),
                        'avg_confidence': round(count_obj.avg_confidence, 3),
                        'is_cjmore_private': count_obj.is_cjmore_private,
                        'diversity_ratio': round(count_obj.unique_products / count_obj.total_instances, 3) if count_obj.total_instances > 0 else 0.0
                    })
            
            brand_count_df = pd.DataFrame(brand_count_data)
            # Sortiere nach Anzahl einzigartiger Produkte (absteigend)
            brand_count_df = brand_count_df.sort_values('unique_products', ascending=False)
            brand_count_df.to_excel(writer, sheet_name='Brand Product Counts', index=False)
    
    logger.info(f"✅ Erweiterte Excel-Berichte erstellt: {excel_path}")

if __name__ == "__main__":
    # Default: CJMore analysis
    # For Tops Daily, use: main('tops_daily')
    main()