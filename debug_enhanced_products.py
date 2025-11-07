"""
Debug einzelnes Produkt aus enhanced_results.json
"""

import json

# Lade enhanced results
with open('/Users/lukaswalter/Documents/GitHub/Supermarket_Analysis/brand_analysis_output/enhanced_results.json', 'r') as f:
    products = json.load(f)

print("🔍 DEBUGGING Enhanced Products CJMore Classification:")
print("=" * 70)

# Prüfe erste 5 Produkte
for i, product in enumerate(products[:10]):
    brand = product.get('brand', 'unknown')
    cjmore_data = product.get('cjmore_classification', {})
    
    is_private = cjmore_data.get('is_private_brand', 'MISSING')
    brand_name = cjmore_data.get('brand_name', '')
    confidence = cjmore_data.get('confidence', 0.0)
    method = cjmore_data.get('detection_method', 'none')
    
    print(f"Produkt {i+1:2}: Brand='{brand:15}' | Private={is_private:6} | "
          f"Conf={confidence:.3f} | Method={method:12} | PrivateName='{brand_name}'")

# Schaue nach echten Private Brands
print("\n🔍 Suche nach echten Private Brands (UNO, NINE BEAUTY, BAO CAFE, TIAN TIAN):")
private_brands_found = []

for i, product in enumerate(products):
    cjmore_data = product.get('cjmore_classification', {})
    if cjmore_data.get('is_private_brand', False):
        brand = product.get('brand', 'unknown')
        brand_name = cjmore_data.get('brand_name', '')
        confidence = cjmore_data.get('confidence', 0.0)
        method = cjmore_data.get('detection_method', 'none')
        
        private_brands_found.append((brand, brand_name, confidence, method))

if private_brands_found:
    print(f"✅ Gefunden: {len(private_brands_found)} Private Brand Produkte:")
    for brand, priv_name, conf, method in private_brands_found:
        print(f"  Brand='{brand}' -> PrivateBrand='{priv_name}' (Conf={conf:.3f}, {method})")
else:
    print("❌ Keine Private Brands gefunden!")