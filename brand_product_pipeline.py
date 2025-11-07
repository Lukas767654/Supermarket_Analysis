"""
Supermarket Brand & Product Analysis Pipeline
===========================================
Automatische Erkennung von Marken und Produkttypen aus Supermarktregalen.

Workflow:
1. Bilder einlesen und optional segmentieren
2. Whole-Image-Analyse mit Gemini Vision
3. OCR + Logo-Erkennung
4. Fusion der Ergebnisse
5. Embeddings-Berechnung (CLIP)
6. Dopplungserkennung via Clustering
7. Aggregation und Excel-Export
"""

import os
import json
import logging
import base64
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import cv2
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import faiss
from tqdm import tqdm
import re
import unicodedata

# =============================================================================
# 🔧 KONFIGURATION
# =============================================================================

# Pfade
IMAGE_FOLDER = "./images/"
OUTPUT_FOLDER = "./brand_analysis_output/"
EXAMPLES_FOLDER = "./brand_analysis_output/examples/"

# API Keys
GOOGLE_API_KEY = "AIzaSyBxZQmbHOml59U1rxb2_Gd2dRnjnwwzLHY"

# Schwellenwerte
CONFIDENCE_THRESHOLD = 0.75
DUPLICATE_SCORE_THRESHOLD = 0.8
CLUSTERING_EPS = 0.2
MIN_SAMPLES = 2

# Modelle
EMBEDDING_MODEL = "clip-ViT-B-32"
IMAGE_RESIZE_WIDTH = 2048

# Segmentierung
ENABLE_SEGMENTATION = True
SEGMENTS_PER_IMAGE = 3

# Logging
LOG_LEVEL = logging.INFO

# =============================================================================
# 📊 DATENSTRUKTUREN
# =============================================================================

@dataclass
class VisionResult:
    """Ergebnis der Vision-API-Analyse"""
    brand: str
    type: str
    approx_count: int
    confidence: float
    keywords: List[str]

@dataclass
class OCRResult:
    """Ergebnis der OCR + Logo-Erkennung"""
    ocr_texts: List[Dict[str, Any]]  # [{"text": str, "conf": float}]
    logos: List[Dict[str, Any]]      # [{"brand": str, "conf": float}]
    clean_tokens: List[str]          # bereinigte Tokens

@dataclass
class FusedResult:
    """Fusioniertes Ergebnis pro Bild/Segment"""
    image_id: str
    brand: str
    type: str
    approx_count: int
    conf_fused: float
    source_data: Dict[str, Any]

@dataclass
class ClusterData:
    """Cluster-Metadaten"""
    cluster_id: int
    brand: str
    type: str
    images_in_cluster: int
    conf_mean: float
    example_image: str
    all_images: List[str]

# =============================================================================
# 🔧 SETUP & LOGGING
# =============================================================================

def setup_logging():
    """Konfiguriere Logging"""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{OUTPUT_FOLDER}/pipeline.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def setup_directories():
    """Erstelle Output-Verzeichnisse"""
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(EXAMPLES_FOLDER, exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/intermediate", exist_ok=True)

# =============================================================================
# 🖼️ SCHRITT 1: BILDER EINLESEN UND SEGMENTIEREN
# =============================================================================

def load_and_prepare_images() -> List[Dict[str, Any]]:
    """
    Lade alle Bilder und erstelle optional Segmente.
    
    Returns:
        List von Image-Dictionaries mit Pfaden und Metadaten
    """
    logger = logging.getLogger(__name__)
    logger.info("📷 Schritt 1: Bilder einlesen und vorbereiten")
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []
    
    # Sammle alle Bildateien
    for ext in image_extensions:
        image_files.extend(Path(IMAGE_FOLDER).glob(f"*{ext}"))
        image_files.extend(Path(IMAGE_FOLDER).glob(f"*{ext.upper()}"))
    
    logger.info(f"Gefunden: {len(image_files)} Bilder")
    
    processed_images = []
    
    for img_path in tqdm(image_files, desc="Bilder verarbeiten"):
        try:
            # Lade Bild
            image = Image.open(img_path)
            
            # Resize wenn nötig
            if image.width > IMAGE_RESIZE_WIDTH:
                ratio = IMAGE_RESIZE_WIDTH / image.width
                new_height = int(image.height * ratio)
                image = image.resize((IMAGE_RESIZE_WIDTH, new_height), Image.Resampling.LANCZOS)
            
            image_id = img_path.stem
            
            if ENABLE_SEGMENTATION and image.height > image.width:
                # Erstelle horizontale Segmente für hohe Bilder
                segment_height = image.height // SEGMENTS_PER_IMAGE
                
                for i in range(SEGMENTS_PER_IMAGE):
                    y_start = i * segment_height
                    y_end = min((i + 1) * segment_height, image.height)
                    
                    segment = image.crop((0, y_start, image.width, y_end))
                    segment_id = f"{image_id}_seg{i}"
                    
                    # Speichere Segment
                    segment_path = f"{OUTPUT_FOLDER}/intermediate/{segment_id}.jpg"
                    segment.save(segment_path, quality=95)
                    
                    processed_images.append({
                        'image_id': segment_id,
                        'original_path': str(img_path),
                        'processed_path': segment_path,
                        'is_segment': True,
                        'segment_index': i,
                        'parent_image': image_id,
                        'dimensions': segment.size
                    })
            else:
                # Vollbild verwenden
                processed_path = f"{OUTPUT_FOLDER}/intermediate/{image_id}.jpg"
                image.save(processed_path, quality=95)
                
                processed_images.append({
                    'image_id': image_id,
                    'original_path': str(img_path),
                    'processed_path': processed_path,
                    'is_segment': False,
                    'segment_index': 0,
                    'parent_image': image_id,
                    'dimensions': image.size
                })
                
        except Exception as e:
            logger.error(f"Fehler bei Bild {img_path}: {e}")
            continue
    
    logger.info(f"✅ {len(processed_images)} Bilder/Segmente vorbereitet")
    
    # Speichere Metadaten
    with open(f"{OUTPUT_FOLDER}/intermediate/image_metadata.json", 'w') as f:
        json.dump(processed_images, f, indent=2)
    
    return processed_images

# =============================================================================
# 🤖 SCHRITT 2: WHOLE-IMAGE-ANALYSE MIT GEMINI VISION
# =============================================================================

def analyze_image_with_gemini(image_path: str) -> List[VisionResult]:
    """
    Analysiere Bild mit Gemini Vision API.
    
    Args:
        image_path: Pfad zum Bild
        
    Returns:
        Liste von VisionResult-Objekten
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Lade und encode Bild
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Gemini Vision Prompt
        prompt = """
        Analysiere dieses Supermarktregal-Bild und identifiziere alle sichtbaren Produkte.

        Für jedes erkannte Produkt, gib mir folgende Informationen im JSON-Format:

        {
            "products": [
                {
                    "brand": "Markenname (z.B. Nivea, L'Oreal, Colgate) oder 'unknown'",
                    "type": "Produkttyp (z.B. lotion, shampoo, snack, drink, powder, soap, cream, oil, spray)",
                    "approx_count": "ungefähre Anzahl sichtbarer Packungen dieser Brand+Type-Kombination (1-20)",
                    "confidence": "Vertrauen in die Erkennung (0.0-1.0)",
                    "keywords": ["Liste", "relevanter", "Schlüsselwörter", "aus", "dem", "Bild"]
                }
            ]
        }

        Regeln:
        - Ignoriere Preisschilder, Werbung und Regalstrukturen
        - Konzentriere dich nur auf Produktpackungen
        - Verwende englische Begriffe für 'type' (lotion statt Lotion)
        - Verwende Original-Markennamen (auch Thai-Schrift)
        - Bei unklaren Marken: "unknown"
        - confidence basierend auf Sichtbarkeit der Marke/des Produkts
        """
        
        # API Request
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
        
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }}
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 2048
            }
        }
        
        response = requests.post(
            f"{url}?key={GOOGLE_API_KEY}",
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Gemini API Error {response.status_code}: {response.text}")
            return []
        
        # Parse Response
        result = response.json()
        
        if 'candidates' not in result or not result['candidates']:
            logger.warning(f"No candidates in Gemini response for {image_path}")
            return []
        
        text_response = result['candidates'][0]['content']['parts'][0]['text']
        
        # Extract JSON
        try:
            # Versuche JSON zu extrahieren
            json_match = re.search(r'\{.*\}', text_response, re.DOTALL)
            if json_match:
                json_data = json.loads(json_match.group())
                
                results = []
                for product in json_data.get('products', []):
                    results.append(VisionResult(
                        brand=product.get('brand', 'unknown'),
                        type=product.get('type', 'unknown'),
                        approx_count=int(product.get('approx_count', 1)),
                        confidence=float(product.get('confidence', 0.5)),
                        keywords=product.get('keywords', [])
                    ))
                
                return results
            else:
                logger.warning(f"No JSON found in Gemini response for {image_path}")
                return []
                
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {image_path}: {e}")
            logger.error(f"Response text: {text_response}")
            return []
            
    except Exception as e:
        logger.error(f"Gemini analysis failed for {image_path}: {e}")
        return []

# =============================================================================
# 🔍 SCHRITT 3: OCR + LOGO-ERKENNUNG
# =============================================================================

def clean_ocr_tokens(text: str) -> List[str]:
    """
    Bereinige OCR-Text und extrahiere relevante Tokens.
    
    Args:
        text: OCR-Text
        
    Returns:
        Liste bereinigte Tokens
    """
    # Entferne Preiszeichen und unwichtige Patterns
    price_patterns = [
        r'฿\s*\d+', r'THB\s*\d+', r'\d+\s*฿', r'\d+\s*THB',
        r'ราคา', r'บาท', r'\d+\.\d+', r'\d+,\d+',
        r'\b\d+\s*(ml|g|kg|oz|pack|pcs)\b',
        r'\b(01|02|03)\s+(light|dark|medium)\b',
        r'\b\d{2,}\b'  # Lange Zahlen
    ]
    
    cleaned_text = text
    for pattern in price_patterns:
        cleaned_text = re.sub(pattern, ' ', cleaned_text, flags=re.IGNORECASE)
    
    # Normalisiere Unicode
    cleaned_text = unicodedata.normalize('NFKD', cleaned_text)
    
    # Tokenize
    tokens = re.findall(r'\b\w+\b', cleaned_text)
    
    # Filtere kurze und numerische Tokens
    filtered_tokens = []
    for token in tokens:
        if len(token) >= 2 and not token.isdigit():
            filtered_tokens.append(token.lower())
    
    return list(set(filtered_tokens))  # Duplikate entfernen

def run_ocr_and_logo(image_path: str) -> OCRResult:
    """
    Führe OCR und Logo-Erkennung auf einem Bild aus.
    
    Args:
        image_path: Pfad zum Bild
        
    Returns:
        OCRResult mit OCR-Texten und Logos
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Lade und encode Bild
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Vision API für OCR und Logo Detection
        url = "https://vision.googleapis.com/v1/images:annotate"
        
        payload = {
            "requests": [{
                "image": {"content": image_data},
                "features": [
                    {"type": "TEXT_DETECTION", "maxResults": 100},
                    {"type": "LOGO_DETECTION", "maxResults": 20}
                ]
            }]
        }
        
        response = requests.post(
            f"{url}?key={GOOGLE_API_KEY}",
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            logger.error(f"Vision API Error {response.status_code}: {response.text}")
            return OCRResult([], [], [])
        
        result = response.json()
        
        if 'responses' not in result:
            return OCRResult([], [], [])
        
        response_data = result['responses'][0]
        
        # Parse OCR Results
        ocr_texts = []
        full_text = ""
        
        if 'textAnnotations' in response_data:
            for annotation in response_data['textAnnotations']:
                text = annotation['description']
                conf = 0.8  # Default confidence für Vision API
                
                ocr_texts.append({
                    'text': text,
                    'conf': conf
                })
                
                if len(text) > len(full_text):
                    full_text = text  # Längster Text = vollständiger Text
        
        # Parse Logo Results
        logos = []
        if 'logoAnnotations' in response_data:
            for logo in response_data['logoAnnotations']:
                logos.append({
                    'brand': logo['description'],
                    'conf': logo['score']
                })
        
        # Bereinige Tokens
        clean_tokens = clean_ocr_tokens(full_text)
        
        return OCRResult(
            ocr_texts=ocr_texts,
            logos=logos,
            clean_tokens=clean_tokens
        )
        
    except Exception as e:
        logger.error(f"OCR/Logo detection failed for {image_path}: {e}")
        return OCRResult([], [], [])

# =============================================================================
# 🔀 SCHRITT 4: FUSION DER ERGEBNISSE
# =============================================================================

def fuse_results(image_id: str, vision_results: List[VisionResult], 
                ocr_result: OCRResult) -> List[FusedResult]:
    """
    Fusioniere Vision- und OCR-Ergebnisse pro Bild.
    
    Args:
        image_id: Bild-ID
        vision_results: Gemini Vision Ergebnisse
        ocr_result: OCR/Logo Ergebnisse
        
    Returns:
        Liste fusionierter Ergebnisse
    """
    logger = logging.getLogger(__name__)
    
    fused_results = []
    
    # Extrahiere OCR-Marken und Logos
    detected_brands = set()
    for logo in ocr_result.logos:
        detected_brands.add(logo['brand'].lower())
    
    # Suche nach Marken in OCR-Tokens
    common_brands = [
        'nivea', 'loreal', 'olay', 'pond', 'garnier', 'dove', 'vaseline',
        'johnson', 'baby', 'oasis', 'glade', 'kayari', 'mama', 'knorr'
    ]
    
    for token in ocr_result.clean_tokens:
        for brand in common_brands:
            if brand in token or token in brand:
                detected_brands.add(brand)
    
    # Fusioniere jedes Vision-Ergebnis
    for vision_result in vision_results:
        conf_fused = vision_result.confidence
        
        # Boost wenn OCR/Logo die gleiche Marke bestätigt
        vision_brand = vision_result.brand.lower()
        if any(brand in vision_brand or vision_brand in brand for brand in detected_brands):
            conf_fused += 0.05
            logger.debug(f"Brand match boost: {vision_brand}")
        
        # Boost wenn Keywords mit OCR übereinstimmen
        keyword_matches = 0
        for keyword in vision_result.keywords:
            if keyword.lower() in ocr_result.clean_tokens:
                keyword_matches += 1
        
        if keyword_matches > 0:
            conf_fused += min(0.05 * keyword_matches, 0.15)
            logger.debug(f"Keyword matches: {keyword_matches}")
        
        # Penalty bei widersprüchlichen Typ-Informationen
        type_keywords = vision_result.type.lower()
        conflicting_types = {
            'lotion': ['snack', 'drink', 'food'],
            'snack': ['lotion', 'cream', 'soap'],
            'drink': ['lotion', 'cream', 'soap']
        }
        
        if type_keywords in conflicting_types:
            for token in ocr_result.clean_tokens:
                if any(conflict in token for conflict in conflicting_types[type_keywords]):
                    conf_fused -= 0.15
                    logger.debug(f"Type conflict penalty: {type_keywords} vs {token}")
                    break
        
        # Begrenze Confidence
        conf_fused = max(0.0, min(1.0, conf_fused))
        
        fused_result = FusedResult(
            image_id=image_id,
            brand=vision_result.brand,
            type=vision_result.type,
            approx_count=vision_result.approx_count,
            conf_fused=conf_fused,
            source_data={
                'vision': asdict(vision_result),
                'ocr_tokens': ocr_result.clean_tokens,
                'logos': ocr_result.logos
            }
        )
        
        fused_results.append(fused_result)
    
    return fused_results

# =============================================================================
# 🎯 SCHRITT 5: EMBEDDINGS-BERECHNUNG
# =============================================================================

class EmbeddingCalculator:
    """Berechnet multimodale Embeddings für Bilder und Text"""
    
    def __init__(self):
        """Initialisiere Embedding-Modelle"""
        self.logger = logging.getLogger(__name__)
        
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            self.logger.info(f"✅ Embedding-Modell geladen: {EMBEDDING_MODEL}")
        except Exception as e:
            self.logger.error(f"❌ Fehler beim Laden des Embedding-Modells: {e}")
            raise
    
    def calculate_image_embedding(self, image_path: str) -> np.ndarray:
        """
        Berechne visuelles Embedding für ein Bild.
        
        Args:
            image_path: Pfad zum Bild
            
        Returns:
            Normalisierter Embedding-Vektor
        """
        try:
            image = Image.open(image_path).convert('RGB')
            embedding = self.model.encode([image])[0]
            return normalize([embedding])[0]
        except Exception as e:
            self.logger.error(f"Bild-Embedding-Fehler für {image_path}: {e}")
            return np.zeros(512)  # Fallback
    
    def calculate_text_embedding(self, tokens: List[str]) -> np.ndarray:
        """
        Berechne Text-Embedding für OCR-Tokens.
        
        Args:
            tokens: Liste von OCR-Tokens
            
        Returns:
            Normalisierter Embedding-Vektor
        """
        try:
            if not tokens:
                return np.zeros(512)
            
            text = ' '.join(tokens)
            embedding = self.model.encode([text])[0]
            return normalize([embedding])[0]
        except Exception as e:
            self.logger.error(f"Text-Embedding-Fehler: {e}")
            return np.zeros(512)
    
    def fuse_embeddings(self, image_vec: np.ndarray, 
                       text_vec: np.ndarray,
                       image_weight: float = 0.7) -> np.ndarray:
        """
        Fusioniere Bild- und Text-Embeddings.
        
        Args:
            image_vec: Bild-Embedding
            text_vec: Text-Embedding  
            image_weight: Gewichtung für Bild (0-1)
            
        Returns:
            Fusioniertes, normalisiertes Embedding
        """
        text_weight = 1.0 - image_weight
        fused = image_weight * image_vec + text_weight * text_vec
        return normalize([fused])[0]

def calculate_all_embeddings(processed_images: List[Dict], 
                           all_fused_results: List[FusedResult]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Berechne Embeddings für alle Bilder.
    
    Args:
        processed_images: Liste verarbeiteter Bilder
        all_fused_results: Alle fusionierten Ergebnisse
        
    Returns:
        Tuple aus (embedding_matrix, image_id_to_index_mapping)
    """
    logger = logging.getLogger(__name__)
    logger.info("🎯 Schritt 5: Embeddings berechnen")
    
    calculator = EmbeddingCalculator()
    embeddings = []
    image_id_to_index = {}
    
    # Erstelle Mapping von image_id zu OCR-Tokens
    id_to_tokens = {}
    for result in all_fused_results:
        image_id = result.image_id
        if image_id not in id_to_tokens:
            id_to_tokens[image_id] = []
        id_to_tokens[image_id].extend(result.source_data.get('ocr_tokens', []))
    
    for i, img_data in enumerate(tqdm(processed_images, desc="Embeddings berechnen")):
        image_id = img_data['image_id']
        image_path = img_data['processed_path']
        
        # Berechne Embeddings
        image_vec = calculator.calculate_image_embedding(image_path)
        text_vec = calculator.calculate_text_embedding(id_to_tokens.get(image_id, []))
        
        # Fusioniere
        fused_vec = calculator.fuse_embeddings(image_vec, text_vec)
        
        embeddings.append(fused_vec)
        image_id_to_index[image_id] = i
    
    embedding_matrix = np.array(embeddings)
    
    # Speichere Embeddings
    np.save(f"{OUTPUT_FOLDER}/intermediate/embeddings.npy", embedding_matrix)
    with open(f"{OUTPUT_FOLDER}/intermediate/embedding_index.json", 'w') as f:
        json.dump(image_id_to_index, f, indent=2)
    
    logger.info(f"✅ {len(embeddings)} Embeddings berechnet")
    
    return embedding_matrix, image_id_to_index

# =============================================================================
# 🔍 SCHRITT 6: DOPPLUNGSERKENNUNG & CLUSTERING
# =============================================================================

def jaccard_similarity(set1: set, set2: set) -> float:
    """Berechne Jaccard-Ähnlichkeit zwischen zwei Token-Sets"""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def detect_duplicates_and_cluster(embedding_matrix: np.ndarray,
                                processed_images: List[Dict],
                                all_fused_results: List[FusedResult],
                                image_id_to_index: Dict[str, int]) -> List[List[int]]:
    """
    Erkennung von Duplikaten und Clustering ähnlicher Produkte.
    
    Args:
        embedding_matrix: Matrix aller Embeddings
        processed_images: Verarbeitete Bilder
        all_fused_results: Fusionierte Ergebnisse
        image_id_to_index: Mapping von image_id zu Index
        
    Returns:
        Liste von Clustern (jeder Cluster ist eine Liste von Indizes)
    """
    logger = logging.getLogger(__name__)
    logger.info("🔍 Schritt 6: Dopplungserkennung und Clustering")
    
    n_images = len(processed_images)
    
    # Erstelle Mapping für OCR-Tokens
    index_to_tokens = {}
    for result in all_fused_results:
        if result.image_id in image_id_to_index:
            idx = image_id_to_index[result.image_id]
            if idx not in index_to_tokens:
                index_to_tokens[idx] = set()
            index_to_tokens[idx].update(result.source_data.get('ocr_tokens', []))
    
    # Berechne Ähnlichkeitsmatrix
    logger.info("Berechne visuelle Ähnlichkeiten...")
    visual_similarities = cosine_similarity(embedding_matrix)
    
    logger.info("Berechne Text-Ähnlichkeiten...")
    text_similarities = np.zeros((n_images, n_images))
    
    for i in tqdm(range(n_images), desc="Text-Ähnlichkeiten"):
        tokens_i = index_to_tokens.get(i, set())
        for j in range(i, n_images):
            tokens_j = index_to_tokens.get(j, set())
            
            sim = jaccard_similarity(tokens_i, tokens_j)
            text_similarities[i, j] = sim
            text_similarities[j, i] = sim  # Symmetrisch
    
    # Fusioniere Ähnlichkeiten
    logger.info("Fusioniere Ähnlichkeiten...")
    combined_similarities = 0.6 * visual_similarities + 0.4 * text_similarities
    
    # Konvertiere zu Distanzen für DBSCAN
    distances = 1.0 - combined_similarities
    
    # Stelle sicher, dass Distanzen nicht negativ sind
    distances = np.clip(distances, 0.0, 1.0)
    
    # Setze Diagonale auf 0 (Distanz zu sich selbst)
    np.fill_diagonal(distances, 0.0)
    
    # Führe DBSCAN-Clustering durch
    logger.info(f"Führe DBSCAN-Clustering durch (eps={CLUSTERING_EPS}, min_samples={MIN_SAMPLES})...")
    
    # Verwende precomputed distance matrix
    clustering = DBSCAN(
        eps=CLUSTERING_EPS, 
        min_samples=MIN_SAMPLES,
        metric='precomputed'
    ).fit(distances)
    
    labels = clustering.labels_
    
    # Gruppiere in Cluster
    clusters = defaultdict(list)
    noise_points = []
    
    for i, label in enumerate(labels):
        if label == -1:  # Noise
            noise_points.append(i)
        else:
            clusters[label].append(i)
    
    cluster_list = list(clusters.values())
    
    logger.info(f"✅ {len(cluster_list)} Cluster gefunden, {len(noise_points)} Noise-Punkte")
    
    # Speichere Clustering-Ergebnisse
    clustering_results = {
        'clusters': {i: cluster for i, cluster in enumerate(cluster_list)},
        'noise_points': noise_points,
        'similarities': {
            'visual': visual_similarities.tolist(),
            'text': text_similarities.tolist(),
            'combined': combined_similarities.tolist()
        }
    }
    
    with open(f"{OUTPUT_FOLDER}/intermediate/clustering_results.json", 'w') as f:
        json.dump(clustering_results, f, indent=2)
    
    return cluster_list

# =============================================================================
# 📊 SCHRITT 7: CLUSTER-METADATEN
# =============================================================================

def analyze_clusters(cluster_list: List[List[int]],
                    processed_images: List[Dict],
                    all_fused_results: List[FusedResult],
                    image_id_to_index: Dict[str, int]) -> List[ClusterData]:
    """
    Analysiere Cluster und erstelle Metadaten.
    
    Args:
        cluster_list: Liste von Clustern
        processed_images: Verarbeitete Bilder
        all_fused_results: Fusionierte Ergebnisse
        image_id_to_index: Mapping
        
    Returns:
        Liste von ClusterData-Objekten
    """
    logger = logging.getLogger(__name__)
    logger.info("📊 Schritt 7: Cluster-Metadaten erstellen")
    
    # Erstelle Mapping von image_id zu fused_results
    id_to_results = defaultdict(list)
    for result in all_fused_results:
        id_to_results[result.image_id].append(result)
    
    cluster_data_list = []
    
    for cluster_id, cluster_indices in enumerate(tqdm(cluster_list, desc="Cluster analysieren")):
        # Sammle alle Ergebnisse in diesem Cluster
        cluster_results = []
        cluster_images = []
        
        for idx in cluster_indices:
            image_id = processed_images[idx]['image_id']
            cluster_images.append(image_id)
            cluster_results.extend(id_to_results[image_id])
        
        if not cluster_results:
            continue
        
        # Bestimme Mehrheits-Brand und -Type
        brands = [r.brand for r in cluster_results if r.conf_fused >= CONFIDENCE_THRESHOLD]
        types = [r.type for r in cluster_results if r.conf_fused >= CONFIDENCE_THRESHOLD]
        
        if not brands:  # Fallback zu allen Ergebnissen
            brands = [r.brand for r in cluster_results]
            types = [r.type for r in cluster_results]
        
        # Mehrheitsentscheidung
        brand_counter = Counter(brands)
        type_counter = Counter(types)
        
        cluster_brand = brand_counter.most_common(1)[0][0] if brand_counter else "unknown"
        cluster_type = type_counter.most_common(1)[0][0] if type_counter else "unknown"
        
        # Durchschnittliche Confidence
        conf_values = [r.conf_fused for r in cluster_results]
        conf_mean = sum(conf_values) / len(conf_values) if conf_values else 0.0
        
        # Beispiel-Bild (höchste Confidence)
        best_result = max(cluster_results, key=lambda x: x.conf_fused)
        example_image = best_result.image_id
        
        cluster_data = ClusterData(
            cluster_id=cluster_id,
            brand=cluster_brand,
            type=cluster_type,
            images_in_cluster=len(cluster_images),
            conf_mean=conf_mean,
            example_image=example_image,
            all_images=cluster_images
        )
        
        cluster_data_list.append(cluster_data)
    
    logger.info(f"✅ {len(cluster_data_list)} Cluster analysiert")
    
    # Speichere Cluster-Metadaten
    with open(f"{OUTPUT_FOLDER}/intermediate/cluster_metadata.json", 'w') as f:
        json.dump([asdict(cd) for cd in cluster_data_list], f, indent=2)
    
    return cluster_data_list

# =============================================================================
# 📈 SCHRITT 8: AGGREGATION & EXCEL-EXPORT
# =============================================================================

def create_brand_type_summary(cluster_data_list: List[ClusterData]) -> pd.DataFrame:
    """
    Erstelle Zusammenfassung nach Brand und Type.
    
    Args:
        cluster_data_list: Liste von Cluster-Daten
        
    Returns:
        DataFrame mit Brand-Type-Zusammenfassung
    """
    logger = logging.getLogger(__name__)
    logger.info("📈 Schritt 8: Brand-Type-Aggregation")
    
    # Gruppiere nach (brand, type)
    brand_type_groups = defaultdict(list)
    
    for cluster in cluster_data_list:
        key = (cluster.brand, cluster.type)
        brand_type_groups[key].append(cluster)
    
    # Erstelle Summary-Daten
    summary_data = []
    
    for (brand, product_type), clusters in brand_type_groups.items():
        product_kinds = len(clusters)  # Anzahl verschiedene Cluster = Produktarten
        avg_confidence = sum(c.conf_mean for c in clusters) / len(clusters)
        total_images = sum(c.images_in_cluster for c in clusters)
        
        # Bestes Beispielbild (höchste Confidence)
        best_cluster = max(clusters, key=lambda x: x.conf_mean)
        example_image = best_cluster.example_image
        
        summary_data.append({
            'brand': brand,
            'type': product_type,
            'product_kinds': product_kinds,
            'avg_confidence': round(avg_confidence, 3),
            'total_images': total_images,
            'example_image': example_image,
            'cluster_ids': [c.cluster_id for c in clusters]
        })
    
    # Sortiere nach Brand und Produktanzahl
    summary_data.sort(key=lambda x: (x['brand'], -x['product_kinds']))
    
    df = pd.DataFrame(summary_data)
    
    logger.info(f"✅ Zusammenfassung: {len(df)} Brand-Type-Kombinationen")
    
    return df

def export_to_excel(summary_df: pd.DataFrame, cluster_data_list: List[ClusterData]):
    """
    Exportiere Ergebnisse nach Excel.
    
    Args:
        summary_df: Brand-Type-Zusammenfassung
        cluster_data_list: Detaillierte Cluster-Daten
    """
    logger = logging.getLogger(__name__)
    logger.info("📊 Excel-Export wird erstellt...")
    
    excel_path = f"{OUTPUT_FOLDER}/brand_type_summary.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Haupt-Summary
        summary_df.to_excel(writer, sheet_name='Brand_Type_Summary', index=False)
        
        # Detaillierte Cluster-Daten
        cluster_df = pd.DataFrame([asdict(cd) for cd in cluster_data_list])
        cluster_df.to_excel(writer, sheet_name='Cluster_Details', index=False)
        
        # Top-Brands
        brand_stats = summary_df.groupby('brand').agg({
            'product_kinds': 'sum',
            'total_images': 'sum',
            'avg_confidence': 'mean'
        }).sort_values('product_kinds', ascending=False)
        brand_stats.to_excel(writer, sheet_name='Top_Brands')
        
        # Top-Produkttypen
        type_stats = summary_df.groupby('type').agg({
            'product_kinds': 'sum', 
            'total_images': 'sum',
            'avg_confidence': 'mean'
        }).sort_values('product_kinds', ascending=False)
        type_stats.to_excel(writer, sheet_name='Top_Product_Types')
    
    logger.info(f"✅ Excel-Report erstellt: {excel_path}")
    
    return excel_path

# =============================================================================
# 🖼️ SCHRITT 9: BEISPIEL-BILDER KOPIEREN
# =============================================================================

def copy_example_images(cluster_data_list: List[ClusterData], processed_images: List[Dict]):
    """
    Kopiere Beispielbilder für jeden Cluster.
    
    Args:
        cluster_data_list: Cluster-Daten
        processed_images: Verarbeitete Bilder
    """
    logger = logging.getLogger(__name__)
    logger.info("🖼️ Schritt 9: Beispielbilder kopieren")
    
    # Erstelle Mapping von image_id zu Pfad
    id_to_path = {img['image_id']: img['processed_path'] for img in processed_images}
    
    copied_count = 0
    
    for cluster in tqdm(cluster_data_list, desc="Beispiele kopieren"):
        example_id = cluster.example_image
        
        if example_id in id_to_path:
            source_path = id_to_path[example_id]
            
            # Ziel-Dateiname
            safe_brand = re.sub(r'[^\w\-_\.]', '_', cluster.brand)
            safe_type = re.sub(r'[^\w\-_\.]', '_', cluster.type)
            
            dest_filename = f"cluster_{cluster.cluster_id:03d}_{safe_brand}_{safe_type}_{example_id}.jpg"
            dest_path = os.path.join(EXAMPLES_FOLDER, dest_filename)
            
            try:
                # Kopiere Bild
                import shutil
                shutil.copy2(source_path, dest_path)
                copied_count += 1
                
            except Exception as e:
                logger.error(f"Fehler beim Kopieren von {source_path}: {e}")
    
    logger.info(f"✅ {copied_count} Beispielbilder kopiert nach {EXAMPLES_FOLDER}")

# =============================================================================
# 🎛️ HAUPTPIPELINE
# =============================================================================

def main():
    """Hauptfunktion der Brand & Product Analysis Pipeline"""
    
    # Setup
    setup_directories()
    logger = setup_logging()
    
    logger.info("🚀 Brand & Product Analysis Pipeline gestartet")
    logger.info("="*60)
    
    try:
        # Schritt 1: Bilder vorbereiten
        processed_images = load_and_prepare_images()
        
        if not processed_images:
            logger.error("❌ Keine Bilder gefunden!")
            return
        
        # Schritt 2-4: Vision + OCR + Fusion
        logger.info("🤖 Schritt 2-4: Vision-Analyse, OCR und Fusion")
        
        all_fused_results = []
        
        for img_data in tqdm(processed_images, desc="Bilder analysieren"):
            image_id = img_data['image_id']
            image_path = img_data['processed_path']
            
            # Vision-Analyse
            vision_results = analyze_image_with_gemini(image_path)
            
            # OCR + Logo
            ocr_result = run_ocr_and_logo(image_path)
            
            # Fusion
            fused_results = fuse_results(image_id, vision_results, ocr_result)
            all_fused_results.extend(fused_results)
        
        # Speichere fusionierte Ergebnisse
        with open(f"{OUTPUT_FOLDER}/intermediate/fused_results.json", 'w') as f:
            json.dump([asdict(fr) for fr in all_fused_results], f, indent=2)
        
        logger.info(f"✅ {len(all_fused_results)} fusionierte Ergebnisse erstellt")
        
        # Schritt 5: Embeddings
        embedding_matrix, image_id_to_index = calculate_all_embeddings(
            processed_images, all_fused_results
        )
        
        # Schritt 6: Clustering
        cluster_list = detect_duplicates_and_cluster(
            embedding_matrix, processed_images, all_fused_results, image_id_to_index
        )
        
        # Schritt 7: Cluster-Analyse
        cluster_data_list = analyze_clusters(
            cluster_list, processed_images, all_fused_results, image_id_to_index
        )
        
        # Schritt 8: Aggregation & Excel
        summary_df = create_brand_type_summary(cluster_data_list)
        excel_path = export_to_excel(summary_df, cluster_data_list)
        
        # Schritt 9: Beispielbilder
        copy_example_images(cluster_data_list, processed_images)
        
        # Finale Statistiken
        logger.info("="*60)
        logger.info("🎉 PIPELINE ERFOLGREICH ABGESCHLOSSEN!")
        logger.info("="*60)
        logger.info(f"📊 Analysierte Bilder: {len(processed_images)}")
        logger.info(f"🔍 Gefundene Cluster: {len(cluster_data_list)}")
        logger.info(f"🏷️ Brand-Type-Kombinationen: {len(summary_df)}")
        logger.info(f"📈 Excel-Report: {excel_path}")
        logger.info(f"🖼️ Beispielbilder: {EXAMPLES_FOLDER}")
        
        # Top-Ergebnisse anzeigen
        logger.info("\n🏆 TOP BRANDS:")
        brand_counts = summary_df.groupby('brand')['product_kinds'].sum().sort_values(ascending=False)
        for brand, count in brand_counts.head(10).items():
            logger.info(f"   {brand}: {count} Produktarten")
        
        logger.info("\n🏆 TOP PRODUCT TYPES:")
        type_counts = summary_df.groupby('type')['product_kinds'].sum().sort_values(ascending=False)
        for ptype, count in type_counts.head(10).items():
            logger.info(f"   {ptype}: {count} Produktarten")
        
    except Exception as e:
        logger.error(f"❌ Pipeline-Fehler: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()