#!/usr/bin/env python3
"""
Tops Daily Supermarket Configuration
===================================

Configuration file for Tops Daily supermarket brand analysis.
Contains private brand definitions and analysis settings.
"""

import os
from pathlib import Path

# Load environment variables for API keys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed, using system environment only")

# Tops Daily Private Brands
TOPS_DAILY_PRIVATE_BRANDS = [
    'My Choice',
    'My Choice Thai', 
    'Tops',
    'Smart-r',
    'Love The Value'
]

# Ordner-Konfiguration  
SUPERMARKET_NAME = "Tops Daily"
IMAGES_FOLDER = "images/images_tops_daily"
OUTPUT_FOLDER = "tops_daily_analysis_output"
ANALYSIS_FOLDER = "Tops_Daily_Analysis"

# Excel Report Namen
EXCEL_FILENAME = "tops_daily_brand_analysis.xlsx"

# CSV Export Namen  
CSV_FILES = {
    'eye_level': 'tops_daily_eye_level_analysis.csv',
    'brand_classification': 'tops_daily_brand_classification.csv', 
    'private_brands': 'tops_daily_private_brands.csv',
    'thai_vs_international': 'tops_daily_thai_vs_international.csv'
}

# Logging
LOG_FILENAME = "tops_daily_analysis.log"

# Display Namen für Reports
PRIVATE_BRAND_SHEET_NAME = "Tops Daily Private Brands"
SUPERMARKET_DISPLAY_NAME = "Tops Daily Supermarket"

print(f"✅ Loaded {SUPERMARKET_NAME} configuration")
print(f"   📁 Images: {IMAGES_FOLDER}")
print(f"   📁 Output: {OUTPUT_FOLDER}")
print(f"   🏷️  Private Brands: {len(TOPS_DAILY_PRIVATE_BRANDS)} brands")