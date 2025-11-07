"""
Mehrstufige Brand & Kategorie Analyse
=====================================
Erweiterte Pipeline für hierarchische Produktanalyse
"""

import json
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple
import logging
from pathlib import Path

# Import der Katalog-Konfiguration
try:
    from supermarket_catalog import (
        SUPERMARKET_PRODUCT_CATALOG, 
        get_product_category,
        consolidate_brands,
        BRAND_ANALYSIS_CONFIG,
        CATEGORY_MAPPING_CONFIG,
        EXCEL_EXPORT_CONFIG
    )
except ImportError:
    print("⚠️ Supermarket-Katalog nicht gefunden - erstelle Dummy-Konfiguration")
    def get_product_category(product_type, keywords=None):
        return 'uncategorized', 'unknown', 'Unbekannt', 0.5

class MultilevelAnalyzer:
    """
    Mehrstufige Analyse für Supermarkt-Produktdaten.
    
    Analysiert auf verschiedenen Ebenen:
    1. Einzelprodukt-Level (jede Erkennung)
    2. Brand-Level (pro Marke) 
    3. Kategorie-Level (nach Produkttypen)
    4. Hierarchie-Level (Haupt- und Unterkategorien)
    """
    
    def __init__(self, fused_results: List[Dict], output_folder: str):
        """
        Initialisiere Analyzer mit fusionierten Ergebnissen.
        
        Args:
            fused_results: Liste der fusionierten Produkterkennungen
            output_folder: Output-Ordner für Ergebnisse
        """
        self.fused_results = fused_results
        self.output_folder = Path(output_folder)
        self.logger = logging.getLogger(__name__)
        
        # Analyseergebnisse
        self.product_analysis = None
        self.brand_analysis = None  
        self.category_analysis = None
        self.hierarchy_analysis = None
        
        self.logger.info(f"🔍 Mehrstufiger Analyzer initialisiert mit {len(fused_results)} Produkten")
    
    def analyze_all_levels(self):
        """Führe Analyse auf allen Ebenen durch."""
        
        self.logger.info("🎯 Starte mehrstufige Analyse...")
        
        # Stufe 1: Einzelprodukt-Analyse mit Kategorisierung
        self.product_analysis = self._analyze_products()
        
        # Stufe 2: Brand-Level Analyse mit Konsolidierung  
        self.brand_analysis = self._analyze_brands()
        
        # Stufe 3: Kategorie-Level Analyse
        self.category_analysis = self._analyze_categories()
        
        # Stufe 4: Hierarchische Analyse
        self.hierarchy_analysis = self._analyze_hierarchy()
        
        self.logger.info("✅ Mehrstufige Analyse abgeschlossen")
        
        return {
            'products': self.product_analysis,
            'brands': self.brand_analysis, 
            'categories': self.category_analysis,
            'hierarchy': self.hierarchy_analysis
        }
    
    def _analyze_products(self) -> List[Dict]:
        """
        Analysiere jedes einzelne Produkt und ordne Kategorien zu.
        
        Returns:
            Liste erweitreter Produktdaten mit Kategorien
        """
        self.logger.info("📦 Analysiere Einzelprodukte...")
        
        enhanced_products = []
        
        for product in self.fused_results:
            # Kategorisierung
            main_cat, sub_cat, display_name, cat_confidence = get_product_category(
                product['type'], 
                product['source_data']['vision'].get('keywords', [])
            )
            
            # Erweiterte Produktdaten
            enhanced = {
                **product,  # Alle ursprünglichen Daten
                'main_category': main_cat,
                'subcategory': sub_cat,
                'category_display_name': display_name,
                'category_confidence': cat_confidence,
                'analysis_metadata': {
                    'has_ocr_data': len(product['source_data'].get('ocr_tokens', [])) > 0,
                    'keyword_count': len(product['source_data']['vision'].get('keywords', [])),
                    'brand_normalized': product['brand'].strip().lower(),
                    'type_normalized': product['type'].strip().lower()
                }
            }
            
            enhanced_products.append(enhanced)
        
        # Speichere erweiterte Produktdaten
        self._save_json(enhanced_products, 'enhanced_products.json')
        
        self.logger.info(f"✅ {len(enhanced_products)} Produkte analysiert und kategorisiert")
        
        return enhanced_products
    
    def _analyze_brands(self) -> Dict[str, Any]:
        """
        Analysiere auf Brand-Level mit Marken-Konsolidierung.
        
        Returns:
            Brand-Analyse Ergebnisse
        """
        self.logger.info("🏷️ Analysiere Marken-Level...")
        
        # Sammle alle Marken
        all_brands = [p['brand'] for p in self.product_analysis if p['brand'] != 'unknown']
        
        # Konsolidiere ähnliche Marken
        consolidated_brands = consolidate_brands(all_brands)
        
        # Analysiere jede konsolidierte Marke
        brand_stats = {}
        
        for main_brand, brand_variants in consolidated_brands.items():
            
            # Sammle alle Produkte dieser Marke (alle Varianten)
            brand_products = [
                p for p in self.product_analysis 
                if p['brand'] in brand_variants
            ]
            
            if not brand_products:
                continue
            
            # Produkttypen dieser Marke
            product_types = Counter(p['type'] for p in brand_products)
            categories = Counter(p['main_category'] for p in brand_products)
            subcategories = Counter(p['subcategory'] for p in brand_products)
            
            # Berechne Statistiken
            total_products = len(brand_products)
            unique_types = len(product_types)
            avg_confidence = sum(p['conf_fused'] for p in brand_products) / total_products
            total_items = sum(p['approx_count'] for p in brand_products)
            
            # Finde repräsentative Produkte
            best_product = max(brand_products, key=lambda x: x['conf_fused'])
            
            brand_stats[main_brand] = {
                'brand_name': main_brand,
                'brand_variants': brand_variants,
                'total_product_detections': total_products,
                'unique_product_types': unique_types,
                'total_estimated_items': total_items,
                'avg_confidence': round(avg_confidence, 3),
                'product_types': dict(product_types),
                'main_categories': dict(categories),
                'subcategories': dict(subcategories),
                'best_example': {
                    'image_id': best_product['image_id'],
                    'type': best_product['type'],
                    'confidence': best_product['conf_fused'],
                    'category': best_product['category_display_name']
                },
                'all_products': brand_products
            }
        
        # Sortiere nach Produktvielfalt
        sorted_brands = dict(sorted(
            brand_stats.items(), 
            key=lambda x: x[1]['unique_product_types'], 
            reverse=True
        ))
        
        brand_summary = {
            'total_brands': len(sorted_brands),
            'brands_with_multiple_types': len([b for b in brand_stats.values() if b['unique_product_types'] > 1]),
            'most_diverse_brand': max(brand_stats.values(), key=lambda x: x['unique_product_types']) if brand_stats else None,
            'brand_details': sorted_brands
        }
        
        # Speichere Brand-Analyse
        self._save_json(brand_summary, 'brand_analysis.json')
        
        self.logger.info(f"✅ {len(sorted_brands)} Marken analysiert")
        
        return brand_summary
    
    def _analyze_categories(self) -> Dict[str, Any]:
        """
        Analysiere auf Kategorie-Level.
        
        Returns:
            Kategorie-Analyse Ergebnisse  
        """
        self.logger.info("📂 Analysiere Kategorie-Level...")
        
        # Gruppiere nach Kategorien
        category_groups = defaultdict(list)
        subcategory_groups = defaultdict(list)
        
        for product in self.product_analysis:
            category_groups[product['main_category']].append(product)
            subcategory_groups[product['subcategory']].append(product)
        
        # Analysiere Hauptkategorien
        main_category_stats = {}
        for category, products in category_groups.items():
            
            brands = Counter(p['brand'] for p in products if p['brand'] != 'unknown')
            types = Counter(p['type'] for p in products)
            
            main_category_stats[category] = {
                'category_name': category,
                'display_name': SUPERMARKET_PRODUCT_CATALOG.get(category, {}).get('display_name', category),
                'total_products': len(products),
                'unique_brands': len(brands),
                'unique_types': len(types),
                'avg_confidence': sum(p['conf_fused'] for p in products) / len(products),
                'total_items': sum(p['approx_count'] for p in products),
                'top_brands': dict(brands.most_common(5)),
                'top_types': dict(types.most_common(5)),
                'products': products
            }
        
        # Analysiere Unterkategorien  
        subcategory_stats = {}
        for subcategory, products in subcategory_groups.items():
            
            brands = Counter(p['brand'] for p in products if p['brand'] != 'unknown')
            
            subcategory_stats[subcategory] = {
                'subcategory_name': subcategory,
                'display_name': products[0]['category_display_name'] if products else subcategory,
                'main_category': products[0]['main_category'] if products else 'unknown',
                'total_products': len(products),
                'unique_brands': len(brands),
                'avg_confidence': sum(p['conf_fused'] for p in products) / len(products),
                'brands': dict(brands),
                'products': products
            }
        
        category_summary = {
            'main_categories': main_category_stats,
            'subcategories': subcategory_stats,
            'category_distribution': {cat: len(prods) for cat, prods in category_groups.items()},
            'subcategory_distribution': {sub: len(prods) for sub, prods in subcategory_groups.items()}
        }
        
        # Speichere Kategorie-Analyse
        self._save_json(category_summary, 'category_analysis.json')
        
        self.logger.info(f"✅ {len(main_category_stats)} Hauptkategorien analysiert")
        
        return category_summary
    
    def _analyze_hierarchy(self) -> Dict[str, Any]:
        """
        Analysiere hierarchische Struktur: Kategorie → Unterkategorie → Marke → Typ.
        
        Returns:
            Hierarchische Analyse
        """
        self.logger.info("🌳 Analysiere hierarchische Struktur...")
        
        # Baue Hierarchie auf
        hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        
        for product in self.product_analysis:
            main_cat = product['main_category']
            sub_cat = product['subcategory'] 
            brand = product['brand']
            prod_type = product['type']
            
            hierarchy[main_cat][sub_cat][brand][prod_type].append(product)
        
        # Konvertiere zu normaler dict-Struktur mit Statistiken
        hierarchy_stats = {}
        
        for main_cat, subcats in hierarchy.items():
            hierarchy_stats[main_cat] = {
                'display_name': SUPERMARKET_PRODUCT_CATALOG.get(main_cat, {}).get('display_name', main_cat),
                'total_products': sum(len(products) for subcats in subcats.values() 
                                    for brands in subcats.values() 
                                    for types in brands.values() 
                                    for products in types.values()),
                'subcategories': {}
            }
            
            for sub_cat, brands in subcats.items():
                hierarchy_stats[main_cat]['subcategories'][sub_cat] = {
                    'display_name': None,  # Wird aus erstem Produkt geholt
                    'total_products': sum(len(products) for brands in brands.values() 
                                        for types in brands.values() 
                                        for products in types.values()),
                    'brands': {}
                }
                
                # Display-Name aus erstem Produkt holen
                first_product = None
                for brands in brands.values():
                    for types in brands.values():
                        for products in types.values():
                            if products:
                                first_product = products[0]
                                break
                        if first_product:
                            break
                    if first_product:
                        break
                
                if first_product:
                    hierarchy_stats[main_cat]['subcategories'][sub_cat]['display_name'] = first_product['category_display_name']
                
                for brand, types in brands.items():
                    hierarchy_stats[main_cat]['subcategories'][sub_cat]['brands'][brand] = {
                        'total_products': sum(len(products) for products in types.values()),
                        'unique_types': len(types),
                        'product_types': {}
                    }
                    
                    for prod_type, products in types.items():
                        hierarchy_stats[main_cat]['subcategories'][sub_cat]['brands'][brand]['product_types'][prod_type] = {
                            'count': len(products),
                            'total_items': sum(p['approx_count'] for p in products),
                            'avg_confidence': sum(p['conf_fused'] for p in products) / len(products),
                            'products': products
                        }
        
        # Speichere Hierarchie-Analyse
        self._save_json(hierarchy_stats, 'hierarchy_analysis.json')
        
        self.logger.info(f"✅ Hierarchische Struktur mit {len(hierarchy_stats)} Hauptkategorien erstellt")
        
        return hierarchy_stats
    
    def export_to_excel(self, filename: str = None) -> str:
        """
        Exportiere alle Analyseergebnisse in mehrseitige Excel-Datei.
        
        Args:
            filename: Optional custom filename
            
        Returns:
            Pfad zur Excel-Datei
        """
        if filename is None:
            filename = "multilevel_brand_category_analysis.xlsx"
        
        excel_path = self.output_folder / filename
        
        self.logger.info(f"📊 Exportiere mehrstufige Analyse nach Excel: {excel_path}")
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            
            # 1. BRAND OVERVIEW - Hauptübersicht nach Marken
            if self.brand_analysis:
                brand_rows = []
                for brand_name, data in self.brand_analysis['brand_details'].items():
                    brand_rows.append({
                        'brand': brand_name,
                        'brand_variants': ', '.join(data['brand_variants']),
                        'unique_product_types': data['unique_product_types'],
                        'total_detections': data['total_product_detections'], 
                        'estimated_total_items': data['total_estimated_items'],
                        'avg_confidence': data['avg_confidence'],
                        'main_categories': ', '.join(data['main_categories'].keys()),
                        'example_product': data['best_example']['type'],
                        'example_confidence': data['best_example']['confidence'],
                        'example_image': data['best_example']['image_id']
                    })
                
                brand_df = pd.DataFrame(brand_rows)
                brand_df = brand_df.sort_values('unique_product_types', ascending=False)
                brand_df.to_excel(writer, sheet_name='Brand Overview', index=False)
            
            # 2. CATEGORY OVERVIEW - Übersicht nach Kategorien
            if self.category_analysis:
                category_rows = []
                for cat_name, data in self.category_analysis['main_categories'].items():
                    category_rows.append({
                        'category': data['display_name'],
                        'category_code': cat_name,
                        'total_products': data['total_products'],
                        'unique_brands': data['unique_brands'],
                        'unique_types': data['unique_types'],
                        'estimated_items': data['total_items'],
                        'avg_confidence': round(data['avg_confidence'], 3),
                        'top_brands': ', '.join(list(data['top_brands'].keys())[:3]),
                        'top_types': ', '.join(list(data['top_types'].keys())[:3])
                    })
                
                category_df = pd.DataFrame(category_rows)
                category_df = category_df.sort_values('total_products', ascending=False)
                category_df.to_excel(writer, sheet_name='Category Overview', index=False)
            
            # 3. DETAILED HIERARCHY - Vollständige hierarchische Ansicht  
            if self.hierarchy_analysis:
                hierarchy_rows = []
                for main_cat, main_data in self.hierarchy_analysis.items():
                    for sub_cat, sub_data in main_data['subcategories'].items():
                        for brand, brand_data in sub_data['brands'].items():
                            for prod_type, type_data in brand_data['product_types'].items():
                                hierarchy_rows.append({
                                    'main_category': main_data['display_name'],
                                    'subcategory': sub_data['display_name'],
                                    'brand': brand,
                                    'product_type': prod_type,
                                    'detections': type_data['count'],
                                    'estimated_items': type_data['total_items'],
                                    'avg_confidence': round(type_data['avg_confidence'], 3),
                                    'example_images': ', '.join([p['image_id'] for p in type_data['products'][:2]])
                                })
                
                hierarchy_df = pd.DataFrame(hierarchy_rows)
                hierarchy_df = hierarchy_df.sort_values(['main_category', 'subcategory', 'brand', 'product_type'])
                hierarchy_df.to_excel(writer, sheet_name='Hierarchy View', index=False)
            
            # 4. ALL PRODUCTS - Alle erkannten Produkte
            if self.product_analysis:
                products_df = pd.DataFrame([{
                    'image_id': p['image_id'],
                    'brand': p['brand'],
                    'type': p['type'],
                    'category': p['category_display_name'],
                    'subcategory': p['subcategory'],
                    'approx_count': p['approx_count'],
                    'confidence': round(p['conf_fused'], 3),
                    'category_confidence': round(p['category_confidence'], 3),
                    'keywords': ', '.join(p['source_data']['vision'].get('keywords', [])),
                    'has_ocr': p['analysis_metadata']['has_ocr_data']
                } for p in self.product_analysis])
                
                products_df = products_df.sort_values(['brand', 'type'])
                products_df.to_excel(writer, sheet_name='All Products', index=False)
            
            # 5. STATISTICS - Zusammenfassung und Statistiken
            stats_data = []
            
            # Allgemeine Statistiken
            stats_data.append(['Total Products Detected', len(self.product_analysis)])
            stats_data.append(['Total Unique Brands', len(self.brand_analysis['brand_details']) if self.brand_analysis else 0])
            stats_data.append(['Total Main Categories', len(self.category_analysis['main_categories']) if self.category_analysis else 0])
            stats_data.append(['', ''])
            
            # Top-Statistiken
            if self.brand_analysis and self.brand_analysis['most_diverse_brand']:
                diverse_brand = self.brand_analysis['most_diverse_brand']
                stats_data.append(['Most Diverse Brand', diverse_brand['brand_name']])
                stats_data.append(['- Unique Product Types', diverse_brand['unique_product_types']])
                stats_data.append(['', ''])
            
            # Kategorie-Verteilung
            if self.category_analysis:
                stats_data.append(['Category Distribution:', ''])
                for cat, count in self.category_analysis['category_distribution'].items():
                    display_name = SUPERMARKET_PRODUCT_CATALOG.get(cat, {}).get('display_name', cat)
                    stats_data.append([f'- {display_name}', count])
            
            stats_df = pd.DataFrame(stats_data, columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        self.logger.info(f"✅ Excel-Export abgeschlossen: {excel_path}")
        
        return str(excel_path)
    
    def _save_json(self, data: Any, filename: str):
        """Speichere Daten als JSON."""
        json_path = self.output_folder / 'intermediate' / filename
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

def run_multilevel_analysis(fused_results_path: str, output_folder: str) -> str:
    """
    Hauptfunktion für mehrstufige Analyse.
    
    Args:
        fused_results_path: Pfad zur fused_results.json
        output_folder: Output-Ordner
        
    Returns:
        Pfad zur Excel-Datei
    """
    logger = logging.getLogger(__name__)
    
    # Lade fusionierte Ergebnisse
    with open(fused_results_path, 'r', encoding='utf-8') as f:
        fused_results = json.load(f)
    
    logger.info(f"📊 Starte mehrstufige Analyse mit {len(fused_results)} Produkten")
    
    # Erstelle Analyzer
    analyzer = MultilevelAnalyzer(fused_results, output_folder)
    
    # Führe alle Analysen durch
    all_results = analyzer.analyze_all_levels()
    
    # Exportiere nach Excel
    excel_path = analyzer.export_to_excel()
    
    # Übersicht ausgeben
    logger.info("🎉 MEHRSTUFIGE ANALYSE ABGESCHLOSSEN!")
    logger.info("=" * 60)
    
    if analyzer.brand_analysis:
        logger.info(f"🏷️  Analysierte Marken: {analyzer.brand_analysis['total_brands']}")
        if analyzer.brand_analysis['most_diverse_brand']:
            diverse = analyzer.brand_analysis['most_diverse_brand']
            logger.info(f"🏆 Vielfältigste Marke: {diverse['brand_name']} ({diverse['unique_product_types']} Produkttypen)")
    
    if analyzer.category_analysis:
        logger.info(f"📂 Hauptkategorien: {len(analyzer.category_analysis['main_categories'])}")
        logger.info(f"📋 Unterkategorien: {len(analyzer.category_analysis['subcategories'])}")
    
    logger.info(f"📊 Excel-Report: {excel_path}")
    
    return excel_path

if __name__ == "__main__":
    # Test mit aktuellen Daten
    import sys
    
    if len(sys.argv) > 1:
        fused_results_path = sys.argv[1]
        output_folder = sys.argv[2] if len(sys.argv) > 2 else "./output"
        
        excel_path = run_multilevel_analysis(fused_results_path, output_folder)
        print(f"✅ Analyse abgeschlossen: {excel_path}")
    else:
        print("Verwendung: python multilevel_analysis.py <fused_results.json> [output_folder]")