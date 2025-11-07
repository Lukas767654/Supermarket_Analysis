"""
Supermarket Product Catalog and Multi-Level Analysis
====================================================
Enhanced Configuration for Brand and Category Analysis
"""

import re
from difflib import SequenceMatcher

# =============================================================================
# 🏪 SUPERMARKET PRODUCT CATALOG
# =============================================================================

# Comprehensive supermarket product catalog with hierarchical structure
SUPERMARKET_PRODUCT_CATALOG = {
    
    # 🧴 PERSONAL CARE & COSMETICS
    'personal_care': {
        'display_name': 'Personal Care & Cosmetics',
        'subcategories': {
            'skincare': {
                'display_name': 'Skin Care',
                'keywords': ['lotion', 'cream', 'moisturizer', 'serum', 'gel', 'balm', 'oil', 'butter', 'cleansing', 'toner', 'mask', 'scrub', 'sunscreen', 'anti-aging', 'whitening', 'face wash', 'cleanser', 'exfoliant', 'โลชั่น', 'ครีม', 'เซรั่ม', 'ทำความสะอาดผิว', 'กันแดด', 'ผิวขาว']
            },
            'haircare': {
                'display_name': 'Hair Care', 
                'keywords': ['shampoo', 'conditioner', 'hair mask', 'hair oil', 'styling gel', 'wax', 'mousse', 'spray', 'treatment', 'serum', 'hair dye', 'hair color', 'hair cream', 'pomade']
            },
            'oral_care': {
                'display_name': 'Oral Care',
                'keywords': ['toothpaste', 'mouthwash', 'toothbrush', 'dental floss', 'whitening', 'gum care', 'breath freshener', 'dental rinse']
            },
            'bath_shower': {
                'display_name': 'Bath & Shower',
                'keywords': ['soap', 'shower gel', 'body wash', 'bubble bath', 'bath salt', 'sponge', 'body scrub', 'exfoliating soap', 'bar soap']
            },
            'deodorant_fragrance': {
                'display_name': 'Deodorant & Fragrance',
                'keywords': ['deodorant', 'antiperspirant', 'perfume', 'cologne', 'body spray', 'fragrance', 'eau de toilette', 'eau de parfum']
            },
            'makeup_cosmetics': {
                'display_name': 'Makeup & Cosmetics',
                'keywords': ['lipstick', 'foundation', 'mascara', 'eyeliner', 'eyeshadow', 'blush', 'powder', 'concealer', 'nail polish', 'makeup remover']
            }
        }
    },
    
    # 🏠 HOUSEHOLD & CLEANING  
    'household': {
        'display_name': 'Household & Cleaning',
        'subcategories': {
            'air_fresheners': {
                'display_name': 'Air Fresheners & Room Fragrances',
                'keywords': ['air freshener', 'room spray', 'scented', 'fragrance', 'deodorizer', 'fabric freshener', 'car freshener', 'gel freshener', 'automatic freshener', 'plug-in freshener', 'aerosol freshener', 'ฟรีชเชนเนอร์', 'น้ำหอม', 'ปรับอากาศ', 'กลิ่นหอม']
            },
            'cleaning_supplies': {
                'display_name': 'Cleaning Supplies',
                'keywords': ['detergent', 'cleaner', 'disinfectant', 'bleach', 'soap', 'washing powder', 'fabric softener', 'stain remover', 'all-purpose cleaner', 'multi-surface cleaner', 'ผงซักฟอก', 'น้ำยาซักผ้า', 'สบู่', 'น้ำยาทำความสะอาด', 'ปรับผ้านุ่ม']
            },
            'pest_control': {
                'display_name': 'Pest Control',
                'keywords': ['mosquito coil', 'insect spray', 'repellent', 'ant killer', 'cockroach', 'fly', 'pest control', 'insecticide', 'bug spray', 'mosquito repellent', 'rat poison', 'ยากันยุง', 'ไล่ยุง', 'กำจัดแมลง', 'ควันไล่ยุง', 'สเปรย์ฆ่าแมลง']
            },
            'kitchen_cleaning': {
                'display_name': 'Kitchen Cleaning',
                'keywords': ['dish soap', 'dishwasher', 'degreaser', 'oven cleaner', 'sponge', 'scrubber', 'kitchen cleaner', 'grease remover']
            },
            'floor_care': {
                'display_name': 'Floor Care',
                'keywords': ['floor cleaner', 'mop', 'wax', 'polish', 'vacuum', 'broom', 'floor polish', 'tile cleaner']
            },
            'laundry': {
                'display_name': 'Laundry Care',
                'keywords': ['laundry detergent', 'fabric softener', 'bleach', 'stain remover', 'washing powder', 'liquid detergent', 'fabric conditioner', 'color protector']
            },
            'paper_products': {
                'display_name': 'Paper Products & Disposables',
                'keywords': ['toilet paper', 'tissue', 'paper towel', 'napkin', 'kitchen roll', 'facial tissue', 'wet wipes', 'paper plates', 'disposable cups']
            }
        }
    },
    
    # 🍽️ FOOD & BEVERAGES
    'food_beverages': {
        'display_name': 'Food & Beverages',
        'subcategories': {
            'beverages': {
                'display_name': 'Beverages',
                'keywords': ['drink', 'juice', 'soda', 'water', 'tea', 'coffee', 'milk', 'energy drink', 'sports drink', 'soft drink', 'carbonated', 'non-carbonated', 'fruit juice', 'vegetable juice']
            },
            'snacks_confectionery': {
                'display_name': 'Snacks & Confectionery',
                'keywords': ['snack', 'chips', 'crackers', 'nuts', 'candy', 'chocolate', 'cookies', 'biscuits', 'wafer', 'gum', 'mints', 'popcorn', 'pretzels']
            },
            'instant_convenience': {
                'display_name': 'Instant & Convenience Foods',
                'keywords': ['instant noodles', 'cup noodles', 'ready meal', 'frozen food', 'microwave meal', 'instant soup', 'instant rice', 'quick meal']
            },
            'seasonings_condiments': {
                'display_name': 'Seasonings & Condiments',
                'keywords': ['sauce', 'seasoning', 'spice', 'salt', 'pepper', 'herbs', 'marinade', 'dressing', 'powder', 'paste', 'ketchup', 'mustard', 'soy sauce', 'fish sauce']
            },
            'dairy_alternatives': {
                'display_name': 'Dairy & Alternatives',
                'keywords': ['milk', 'yogurt', 'cheese', 'butter', 'cream', 'soy milk', 'almond milk', 'coconut milk', 'lactose-free']
            },
            'canned_preserved': {
                'display_name': 'Canned & Preserved Foods',
                'keywords': ['canned', 'preserved', 'pickled', 'jarred', 'vacuum packed', 'tinned', 'bottled']
            }
        }
    },
    
    # 🍎 FRESH PRODUCE
    'fresh_produce': {
        'display_name': 'Fresh Produce',
        'subcategories': {
            'fruits': {
                'display_name': 'Fresh Fruits',
                'keywords': ['apple', 'banana', 'orange', 'grape', 'strawberry', 'mango', 'pineapple', 'watermelon', 'papaya', 'kiwi', 'dragon fruit', 'lychee', 'rambutan', 'durian', 'coconut', 'lime', 'lemon', 'pomelo']
            },
            'vegetables': {
                'display_name': 'Fresh Vegetables',
                'keywords': ['tomato', 'cucumber', 'lettuce', 'cabbage', 'carrot', 'onion', 'potato', 'garlic', 'ginger', 'chili', 'bell pepper', 'broccoli', 'cauliflower', 'spinach', 'kale', 'eggplant', 'zucchini']
            },
            'herbs_aromatics': {
                'display_name': 'Fresh Herbs & Aromatics',
                'keywords': ['basil', 'cilantro', 'parsley', 'mint', 'lemongrass', 'galangal', 'thai basil', 'holy basil', 'kaffir lime', 'fresh herbs', 'aromatic leaves']
            }
        }
    },
    
    # 🥖 BAKERY & BREAD
    'bakery_bread': {
        'display_name': 'Bakery & Bread Products',
        'subcategories': {
            'bread_rolls': {
                'display_name': 'Bread & Rolls',
                'keywords': ['bread', 'roll', 'baguette', 'sandwich bread', 'whole wheat', 'white bread', 'multigrain', 'sourdough']
            },
            'pastries_desserts': {
                'display_name': 'Pastries & Desserts',
                'keywords': ['cake', 'pastry', 'croissant', 'muffin', 'donut', 'danish', 'tart', 'pie', 'eclair', 'cream puff']
            }
        }
    },
    
    # 🏥 HEALTH & WELLNESS
    'health_wellness': {
        'display_name': 'Health & Wellness',
        'subcategories': {
            'medicines': {
                'display_name': 'Medicines & Pharmaceuticals',
                'keywords': ['medicine', 'pill', 'tablet', 'syrup', 'ointment', 'cream', 'drops', 'supplement', 'painkiller', 'antibiotic', 'cough syrup', 'cold medicine']
            },
            'vitamins_supplements': {
                'display_name': 'Vitamins & Supplements',
                'keywords': ['vitamin', 'supplement', 'calcium', 'protein powder', 'health drink', 'energy powder', 'multivitamin', 'omega-3', 'probiotics', 'mineral supplement']
            },
            'first_aid': {
                'display_name': 'First Aid & Medical Supplies',
                'keywords': ['bandage', 'plaster', 'antiseptic', 'thermometer', 'cotton', 'gauze', 'adhesive tape', 'medical mask', 'hand sanitizer', 'rubbing alcohol']
            },
            'feminine_hygiene': {
                'display_name': 'Feminine Hygiene',
                'keywords': ['sanitary pad', 'tampon', 'panty liner', 'feminine wash', 'intimate hygiene', 'menstrual cup']
            }
        }
    },
    
    # 🍼 BABY & KIDS
    'baby_kids': {
        'display_name': 'Baby & Kids',
        'subcategories': {
            'baby_care': {
                'display_name': 'Baby Care Products',
                'keywords': ['baby cream', 'baby lotion', 'baby shampoo', 'baby soap', 'diaper cream', 'baby oil', 'powder', 'baby wash', 'baby moisturizer', 'rash cream']
            },
            'diapers_hygiene': {
                'display_name': 'Diapers & Baby Hygiene',
                'keywords': ['diaper', 'nappy', 'wipes', 'baby wipes', 'training pants', 'diaper rash cream', 'baby powder']
            },
            'baby_food': {
                'display_name': 'Baby Food & Formula',
                'keywords': ['baby food', 'formula', 'baby milk', 'baby cereal', 'baby puree', 'baby snacks', 'infant formula', 'baby formula powder']
            },
            'kids_products': {
                'display_name': 'Kids Products',
                'keywords': ['kids toothbrush', 'kids shampoo', 'kids soap', 'kids sunscreen', 'children bath', 'kids oral care']
            }
        }
    },
    
    # 🐕 PET CARE
    'pet_care': {
        'display_name': 'Pet Care & Supplies',
        'subcategories': {
            'pet_food': {
                'display_name': 'Pet Food & Treats',
                'keywords': ['dog food', 'cat food', 'pet food', 'treats', 'bird food', 'fish food', 'pet snacks', 'wet food', 'dry food', 'pet biscuits']
            },
            'pet_hygiene': {
                'display_name': 'Pet Hygiene & Care',
                'keywords': ['pet shampoo', 'flea spray', 'pet deodorizer', 'litter', 'pet wipes', 'flea collar', 'pet toothpaste', 'ear cleaner']
            }
        }
    },
    
    # 🔋 ELECTRONICS & HARDWARE
    'electronics_hardware': {
        'display_name': 'Electronics & Hardware',
        'subcategories': {
            'batteries_power': {
                'display_name': 'Batteries & Power',
                'keywords': ['battery', 'rechargeable', 'alkaline', 'lithium', 'power bank', 'charger', 'adapter', 'power supply']
            },
            'storage_organization': {
                'display_name': 'Storage & Organization',
                'keywords': ['container', 'storage box', 'plastic bag', 'ziplock', 'foil', 'wrap', 'food storage', 'kitchen storage', 'organizer']
            },
            'small_appliances': {
                'display_name': 'Small Appliances & Gadgets',
                'keywords': ['kitchen appliance', 'blender', 'mixer', 'coffee maker', 'toaster', 'electric kettle', 'rice cooker', 'gadget']
            }
        }
    },
    
    # 🏃‍♂️ SPORTS & OUTDOOR
    'sports_outdoor': {
        'display_name': 'Sports & Outdoor',
        'subcategories': {
            'sports_nutrition': {
                'display_name': 'Sports Nutrition',
                'keywords': ['protein powder', 'energy drink', 'sports drink', 'energy bar', 'electrolyte', 'pre-workout', 'post-workout', 'creatine']
            },
            'outdoor_gear': {
                'display_name': 'Outdoor & Recreation',
                'keywords': ['sunscreen', 'insect repellent', 'camping gear', 'hiking supplies', 'outdoor equipment', 'beach supplies']
            }
        }
    },
    
    # 🏠 HOME & GARDEN
    'home_garden': {
        'display_name': 'Home & Garden',
        'subcategories': {
            'gardening': {
                'display_name': 'Gardening Supplies',
                'keywords': ['plant food', 'fertilizer', 'soil', 'seeds', 'plant pot', 'gardening tools', 'watering can', 'garden hose']
            },
            'home_improvement': {
                'display_name': 'Home Improvement',
                'keywords': ['paint', 'brush', 'tools', 'hardware', 'screws', 'nails', 'glue', 'tape', 'repair kit']
            },
            'seasonal_decor': {
                'display_name': 'Seasonal & Decoration',
                'keywords': ['decoration', 'holiday decor', 'seasonal items', 'candles', 'artificial flowers', 'home decor']
            }
        }
    },
    
    # 🚗 AUTOMOTIVE
    'automotive': {
        'display_name': 'Automotive & Car Care',
        'subcategories': {
            'car_care': {
                'display_name': 'Car Care Products',
                'keywords': ['car wash', 'car wax', 'tire shine', 'car polish', 'car cleaner', 'windshield washer', 'car shampoo']
            },
            'car_accessories': {
                'display_name': 'Car Accessories',
                'keywords': ['car freshener', 'car mat', 'car cover', 'phone holder', 'car charger', 'dashboard cleaner']
            }
        }
    },
    
    # 💼 OFFICE & STATIONERY
    'office_stationery': {
        'display_name': 'Office & Stationery',
        'subcategories': {
            'writing_supplies': {
                'display_name': 'Writing Supplies',
                'keywords': ['pen', 'pencil', 'marker', 'highlighter', 'eraser', 'ruler', 'notebook', 'paper']
            },
            'office_supplies': {
                'display_name': 'Office Supplies',
                'keywords': ['stapler', 'tape', 'scissors', 'glue', 'file folder', 'envelope', 'calculator', 'sticky notes']
            }
        }
    }
}

# =============================================================================
# 🔄 MULTI-LEVEL ANALYSIS CONFIGURATION
# =============================================================================

# Brand Analysis Configuration
BRAND_ANALYSIS_CONFIG = {
    'min_products_per_brand': 1,      # Minimum products per brand
    'merge_similar_brands': True,      # Merge similar brand names
    'brand_similarity_threshold': 0.85, # Threshold for similar brands
    'ignore_unknown_brands': False,    # Ignore "unknown" brands
    'consolidate_variants': True       # Consolidate variants (Fumme/Fummie)
}

# Category Mapping Configuration  
CATEGORY_MAPPING_CONFIG = {
    'fuzzy_matching': True,           # Fuzzy category matching
    'similarity_threshold': 0.1,      # Threshold for category match (lowered for better matching)
    'allow_multiple_categories': True, # Multiple categories per product
    'prioritize_specific': True,      # Specific over general categories
    'include_subcategories': True     # Include subcategories
}

# Clustering Strategies
CLUSTERING_STRATEGY = {
    'method': 'hierarchical',         # 'dbscan', 'hierarchical', 'manual'
    'brand_separation': True,         # Separate different brands
    'category_separation': True,      # Separate different categories
    'confidence_weighting': True,     # Include confidence in clustering
    'visual_similarity_weight': 0.3,  # Reduced for brand focus
    'text_similarity_weight': 0.4,    # Increased for brand names
    'brand_similarity_weight': 0.3    # New dimension
}

# Excel Export Configuration
EXCEL_EXPORT_CONFIG = {
    'include_brand_summary': True,     # Brand overview sheet
    'include_category_summary': True,  # Category overview sheet  
    'include_detailed_products': True, # Detailed product list
    'include_hierarchy_view': True,    # Hierarchical view
    'include_statistics': True,        # Statistics sheet
    'brand_consolidation': True,       # Brand consolidation
    'category_hierarchy': True         # Category hierarchy
}

def get_product_category(product_type, keywords=None):
    """
    Determine product category based on type and keywords with Thai support.
    
    Args:
        product_type: Detected product type (Thai or English)
        keywords: Additional keywords (Thai or English)
        
    Returns:
        Tuple (main_category, subcategory, display_name, confidence)
    """
    if keywords is None:
        keywords = []
    
    # Normalize and extract Thai/English terms
    all_terms = [product_type] + keywords
    
    # Extract both Thai and English terms from all inputs
    combined_thai_terms = []
    combined_english_terms = []
    
    for term in all_terms:
        if term:
            term_analysis = extract_thai_and_english_terms(str(term))
            combined_thai_terms.extend(term_analysis['thai'])
            combined_english_terms.extend(term_analysis['english'])
    
    # Create search text including both languages
    search_terms = combined_english_terms + combined_thai_terms
    search_text = ' '.join([t.lower() for t in search_terms if t])
    
    best_match = None
    best_confidence = 0.0
    
    # Search all categories
    for main_cat, main_data in SUPERMARKET_PRODUCT_CATALOG.items():
        for sub_cat, sub_data in main_data['subcategories'].items():
            
            # Check subcategory keywords
            matches = 0
            total_keywords = len(sub_data['keywords'])
            
            for keyword in sub_data['keywords']:
                if keyword.lower() in search_text:
                    matches += 1
                    
                    # Bonus for exact match
                    if keyword.lower() == product_type.lower():
                        matches += 2
            
            # Calculate confidence
            if total_keywords > 0:
                confidence = matches / total_keywords
                
                # Bonus for multiple matches
                if matches > 1:
                    confidence *= 1.2
                    
                # Update best match
                if confidence > best_confidence:
                    best_match = (main_cat, sub_cat, sub_data['display_name'])
                    best_confidence = confidence
    
    if best_match and best_confidence >= CATEGORY_MAPPING_CONFIG['similarity_threshold']:
        return best_match[0], best_match[1], best_match[2], best_confidence
    else:
        return 'uncategorized', 'unknown', 'Unknown', 0.0

def consolidate_brands(brand_list):
    """
    Consolidate similar brands (e.g. Fumme/Fummie).
    
    Args:
        brand_list: List of brand names
        
    Returns:
        Dict with consolidated brands
    """
    from difflib import SequenceMatcher
    
    consolidated = {}
    processed = set()
    
    for brand in brand_list:
        if brand in processed:
            continue
            
        # Find similar brands
        similar = [brand]
        processed.add(brand)
        
        for other_brand in brand_list:
            if other_brand != brand and other_brand not in processed:
                # Calculate similarity
                similarity = SequenceMatcher(None, brand.lower(), other_brand.lower()).ratio()
                
                if similarity >= BRAND_ANALYSIS_CONFIG['brand_similarity_threshold']:
                    similar.append(other_brand)
                    processed.add(other_brand)
        
        # Choose representative name (longest/most frequent)
        if len(similar) > 1:
            representative = max(similar, key=len)  # Longest name
            consolidated[representative] = similar
        else:
            consolidated[brand] = [brand]
    
    return consolidated

def normalize_thai_text(text):
    """
    Normalize Thai text for better matching.
    
    Args:
        text: Input text (Thai or English)
        
    Returns:
        Normalized text string
    """
    if not text:
        return ""
    
    # Convert to lowercase
    normalized = text.lower().strip()
    
    # Remove common Thai diacritics that might interfere
    # (keeping main characters but normalizing variants)
    thai_normalizations = {
        'ั': '',  # Remove mai hanakat if it causes issues
        'ิ': 'ิ',  # Keep sara i
        'ี': 'ี',  # Keep sara ii
        'ึ': 'ึ',  # Keep sara ue
        'ื': 'ื',  # Keep sara uee
    }
    
    for old, new in thai_normalizations.items():
        normalized = normalized.replace(old, new)
    
    # Remove extra whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized

def extract_thai_and_english_terms(text):
    """
    Extract both Thai and English terms from mixed text.
    
    Args:
        text: Mixed Thai/English text
        
    Returns:
        Dict with 'thai' and 'english' term lists
    """
    if not text:
        return {'thai': [], 'english': []}
    
    # Thai unicode range: U+0E00-U+0E7F
    thai_pattern = r'[\u0E00-\u0E7F]+'
    english_pattern = r'[A-Za-z]+'
    
    thai_terms = re.findall(thai_pattern, text)
    english_terms = re.findall(english_pattern, text)
    
    # Clean and normalize
    thai_terms = [normalize_thai_text(term) for term in thai_terms if len(term) > 1]
    english_terms = [term.lower().strip() for term in english_terms if len(term) > 1]
    
    return {
        'thai': thai_terms,
        'english': english_terms
    }

# Test category assignment
if __name__ == "__main__":
    # Test some examples (English and Thai)
    test_products = [
        ("air freshener", ["scented", "room"]),
        ("ฟรีชเชนเนอร์", ["กลิ่นหอม"]),  # Thai air freshener
        ("mosquito coil", ["insect", "repellent"]),
        ("ยากันยุง", ["ไล่ยุง"]),  # Thai mosquito repellent
        ("lotion", ["skin", "moisturizer"]),
        ("โลชั่น", ["ครีม"]),  # Thai lotion
        ("shampoo", ["hair", "cleansing"]),
        ("powder", ["baby", "talc"]),
        ("drink", ["juice", "beverage"]),
        ("apple", ["fresh", "fruit"]),
        ("bread", ["bakery", "wheat"]),
        ("toothpaste", ["oral", "dental"]),
        ("vitamin", ["health", "supplement"])
    ]
    
    print("🧪 CATEGORY ASSIGNMENT TESTS:")
    print("=" * 50)
    
    for product_type, keywords in test_products:
        main_cat, sub_cat, display_name, confidence = get_product_category(product_type, keywords)
        print(f"📦 {product_type} → {display_name} ({confidence:.2f})")
    
    print(f"\n📊 CATALOG STATISTICS:")
    print("=" * 50)
    
    total_categories = len(SUPERMARKET_PRODUCT_CATALOG)
    total_subcategories = sum(len(cat['subcategories']) for cat in SUPERMARKET_PRODUCT_CATALOG.values())
    total_keywords = sum(
        len(subcat['keywords']) 
        for cat in SUPERMARKET_PRODUCT_CATALOG.values()
        for subcat in cat['subcategories'].values()
    )
    
    print(f"📋 Main Categories: {total_categories}")
    print(f"📋 Subcategories: {total_subcategories}")
    print(f"🏷️  Total Keywords: {total_keywords}")
    
    print(f"\n🏷️ BRAND CONSOLIDATION TEST:")
    print("=" * 50)
    
    test_brands = ["Fumme", "Fummie", "Glade", "glade", "OASIS", "Oasis"]
    consolidated = consolidate_brands(test_brands)
    
    for main_brand, variants in consolidated.items():
        if len(variants) > 1:
            print(f"🔗 {main_brand} ← {variants}")
        else:
            print(f"📋 {main_brand}")