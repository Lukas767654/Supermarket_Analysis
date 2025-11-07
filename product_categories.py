"""
Product Category Utilities
==========================
Hilfsfunktionen für Produktkategorisierung
"""

def get_product_category(product_type: str, keywords=None):
    """
    Ordnet Produkttyp einer Hauptkategorie zu.
    
    Args:
        product_type: Produkttyp string
        keywords: Optional keywords für bessere Zuordnung
        
    Returns:
        tuple: (main_category, subcategory, display_name, confidence)
    """
    if keywords is None:
        keywords = []
    
    # Normalisiere Eingabe
    product_lower = product_type.lower().strip()
    keywords_lower = [k.lower() for k in keywords]
    
    # Kombination für Matching
    search_text = f"{product_lower} {' '.join(keywords_lower)}"
    
    # Kategorien-Mappings
    categories = {
        'household_care': {
            'keywords': ['air freshener', 'freshener', 'cleaner', 'detergent', 'fabric softener', 
                        'toilet paper', 'tissue', 'cleaning', 'household', 'laundry'],
            'display_name': 'Household & Cleaning'
        },
        'beverages': {
            'keywords': ['water', 'juice', 'drink', 'beverage', 'soda', 'tea', 'coffee', 'milk'],
            'display_name': 'Beverages'
        },
        'food_snacks': {
            'keywords': ['snack', 'chips', 'crackers', 'cookies', 'candy', 'chocolate', 'nuts'],
            'display_name': 'Food & Snacks'
        },
        'personal_care': {
            'keywords': ['shampoo', 'soap', 'toothpaste', 'deodorant', 'cosmetics', 'skincare'],
            'display_name': 'Personal Care'
        },
        'health_pharmacy': {
            'keywords': ['medicine', 'vitamin', 'supplement', 'pharmacy', 'health', 'medical'],
            'display_name': 'Health & Pharmacy'
        },
        'baby_kids': {
            'keywords': ['baby', 'diaper', 'formula', 'kids', 'children', 'toy'],
            'display_name': 'Baby & Kids'
        },
        'home_garden': {
            'keywords': ['garden', 'plant', 'tools', 'hardware', 'home improvement'],
            'display_name': 'Home & Garden'
        },
        'other': {
            'keywords': [],
            'display_name': 'Other Products'
        }
    }
    
    # Finde beste Kategorie
    best_match = 'other'
    best_confidence = 0.0
    
    for category, data in categories.items():
        if category == 'other':
            continue
        
        # Zähle Keyword-Treffer
        matches = sum(1 for keyword in data['keywords'] if keyword in search_text)
        
        if matches > 0:
            # Berechne Konfidenz basierend auf Anzahl Matches
            confidence = min(matches / len(data['keywords']) * 2, 1.0)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = category
    
    # Fallback für 'other'
    if best_match == 'other':
        best_confidence = 0.3
    
    category_info = categories[best_match]
    
    return (
        best_match,
        best_match,  # subcategory gleich main_category  
        category_info['display_name'],
        best_confidence
    )