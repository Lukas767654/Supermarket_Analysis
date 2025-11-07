#!/usr/bin/env python3
"""
CSV Category Fixer - Korrigiert 'unknown' und 'Other Products' Kategorien in CSV-Dateien
"""

import pandas as pd
import os
from pathlib import Path
import re

def get_category_mapping():
    """Definiere Mapping von Produkttypen zu Kategorien"""
    return {
        # Personal Care & Beauty
        'Body Wash': 'Personal Care',
        'Body Lotion': 'Personal Care', 
        'Face Cream': 'Personal Care',
        'Mouthwash': 'Personal Care',
        'Shampoo': 'Personal Care',
        'Conditioner': 'Personal Care',
        'Soap': 'Personal Care',
        'Toothpaste': 'Personal Care',
        'Toothbrush': 'Personal Care',
        'Deodorant': 'Personal Care',
        'Perfume': 'Personal Care',
        'Moisturizer': 'Personal Care',
        'Lotion': 'Personal Care',
        'Cream': 'Personal Care',
        'Lipstick': 'Personal Care',
        'Mascara': 'Personal Care',
        'Makeup': 'Personal Care',
        'Cosmetics': 'Personal Care',
        'Skincare': 'Personal Care',
        'Facial': 'Personal Care',
        'Shower Gel': 'Personal Care',
        'Hair Care': 'Personal Care',
        'Nail Polish': 'Personal Care',
        'Sunscreen': 'Personal Care',
        'Face Wash': 'Personal Care',
        'Serum': 'Personal Care',
        'Cleanser': 'Personal Care',
        'Toner': 'Personal Care',
        'Moisturizing Cream': 'Personal Care',
        'Moisturizing Gel': 'Personal Care',
        'Face Serum': 'Personal Care',
        'Acne Gel': 'Personal Care',
        'Acne treatment': 'Personal Care',
        'BB Cream': 'Personal Care',
        'CC Cream': 'Personal Care',
        'DD Watermelon Cream': 'Personal Care',
        'Foundation': 'Personal Care',
        'Primer': 'Personal Care',
        'Corrector': 'Personal Care',
        'Blusher': 'Personal Care',
        'Lip Gloss': 'Personal Care',
        'Lip Tint': 'Personal Care',
        'Lip Balm': 'Personal Care',
        'Lip Treatment': 'Personal Care',
        'Lip Oil': 'Personal Care',
        'Lip Serum': 'Personal Care',
        'Lip Mask': 'Personal Care',
        'Lip Gel': 'Personal Care',
        'Lip & Cheek Tint': 'Personal Care',
        'Lip Duo': 'Personal Care',
        'Lip Glass': 'Personal Care',
        'Day Cream': 'Personal Care',
        'Face Mask': 'Personal Care',
        'Cleansing Water': 'Personal Care',
        
        # Food & Beverages
        'Beer': 'Beverages',
        'Instant Noodle': 'Food & Snacks',
        'Instant Noodles': 'Food & Snacks', 
        'Rice': 'Food & Snacks',
        'Cereal': 'Food & Snacks',
        'Flavor Cubes': 'Food & Snacks',
        'Snacks': 'Food & Snacks',
        'Chips': 'Food & Snacks',
        'Cookies': 'Food & Snacks',
        'Chocolate': 'Food & Snacks',
        'Candy': 'Food & Snacks',
        'Juice': 'Beverages',
        'Soda': 'Beverages',
        'Water': 'Beverages',
        'Coffee': 'Beverages',
        'Tea': 'Beverages',
        'Milk': 'Beverages',
        'Yogurt': 'Food & Snacks',
        'Cheese': 'Food & Snacks',
        'Bread': 'Food & Snacks',
        'Noodles': 'Food & Snacks',
        'Pasta': 'Food & Snacks',
        'Sauce': 'Food & Snacks',
        'Oil': 'Food & Snacks',
        'Vinegar': 'Food & Snacks',
        'Spices': 'Food & Snacks',
        'Seasoning': 'Food & Snacks',
        'Instant': 'Food & Snacks',
        'Canned': 'Food & Snacks',
        'Frozen': 'Food & Snacks',
        'Fresh': 'Food & Snacks',
        'Instant Coffee': 'Beverages',
        'Instant Beverage': 'Beverages',
        'Tea Mix': 'Beverages',
        'Jasmine Tea': 'Beverages',
        'Tea Powder': 'Beverages',
        'Herbal Drink': 'Beverages',
        'Instant Tea Drink': 'Beverages',
        'Instant Coffee Drink': 'Beverages',
        'Coffee Creamer': 'Beverages',
        'Sweetened Condensed Milk': 'Beverages',
        'Evaporated Milk': 'Beverages',
        'Flavored Milk': 'Beverages',
        'Soy Milk': 'Beverages',
        'Corn Milk': 'Beverages',
        'Malt Beverage': 'Beverages',
        'Juice Boxes': 'Beverages',
        'Ice Cream': 'Food & Snacks',
        'Sweetener': 'Food & Snacks',
        'Honey': 'Food & Snacks',
        'Sesame Powder': 'Food & Snacks',
        
        # Household & Cleaning
        'Detergent': 'Household & Cleaning',
        'Cleaner': 'Household & Cleaning',
        'Disinfectant': 'Household & Cleaning',
        'Bleach': 'Household & Cleaning',
        'Fabric Softener': 'Household & Cleaning',
        'Air Freshener': 'Household & Cleaning',
        'Toilet Paper': 'Household & Cleaning',
        'Tissue': 'Household & Cleaning',
        'Paper Towel': 'Household & Cleaning',
        'Sponge': 'Household & Cleaning',
        'Brush': 'Household & Cleaning',
        'Mop': 'Household & Cleaning',
        'Vacuum': 'Household & Cleaning',
        'Laundry': 'Household & Cleaning',
        'Dish Soap': 'Household & Cleaning',
        'Bathroom Cleaner': 'Household & Cleaning',
        'Kitchen Cleaner': 'Household & Cleaning',
        'Floor Cleaner': 'Household & Cleaning',
        'Glass Cleaner': 'Household & Cleaning',
        'Laundry Detergent': 'Household & Cleaning',
        'Air conditioner cleaner spray': 'Household & Cleaning',
        'Toilet bowl cleaner': 'Household & Cleaning',
        'Multi surface cleaner spray': 'Household & Cleaning',
        'Multi surface cleaner': 'Household & Cleaning',
        'Original Powder cleaner': 'Household & Cleaning',
        'Organic baby detergent': 'Household & Cleaning',
        'Shoe polish': 'Household & Cleaning',
        
        # Health & Medicine
        'Medicine': 'Health & Pharmacy',
        'Medication': 'Health & Pharmacy',
        'Pill': 'Health & Pharmacy',
        'Tablet': 'Health & Pharmacy',
        'Capsule': 'Health & Pharmacy',
        'Syrup': 'Health & Pharmacy',
        'Bandage': 'Health & Pharmacy',
        'First Aid': 'Health & Pharmacy',
        'Thermometer': 'Health & Pharmacy',
        'Vitamins': 'Health & Pharmacy',
        'Supplements': 'Health & Pharmacy',
        'Pain Relief': 'Health & Pharmacy',
        'Cough': 'Health & Pharmacy',
        'Cold': 'Health & Pharmacy',
        'Fever': 'Health & Pharmacy',
        'Antiseptic': 'Health & Pharmacy',
        'Ointment': 'Health & Pharmacy',
        
        # Baby & Child Care
        'Baby': 'Baby & Kids',
        'Infant': 'Baby & Kids',
        'Diaper': 'Baby & Kids',
        'Formula': 'Baby & Kids',
        'Baby Food': 'Baby & Kids',
        'Baby Bottle': 'Baby & Kids',
        'Pacifier': 'Baby & Kids',
        'Baby Shampoo': 'Baby & Kids',
        'Baby Lotion': 'Baby & Kids',
        'Baby Powder': 'Baby & Kids',
        'Wipes': 'Baby & Kids',
        'Toy': 'Baby & Kids',
        'Children': 'Baby & Kids',
        'Kids': 'Baby & Kids',
        'Baby toothbrush': 'Baby & Kids',
        'Sponge Brush': 'Baby & Kids',
        'Baby comb': 'Baby & Kids',
        'Baby toy': 'Baby & Kids',
        'Baby feeding spoon': 'Baby & Kids',
        'Baby nail care kit': 'Baby & Kids',
        'Baby bottle': 'Baby & Kids',
        'Baby care product': 'Baby & Kids',
        'Baby Bath Set': 'Baby & Kids',
        'Manual Breast Pump': 'Baby & Kids',
        'Breast Pads': 'Baby & Kids',
        'Baby Blanket': 'Baby & Kids',
        'Baby Wipes/Washcloths': 'Baby & Kids',
        'Baby Oil': 'Baby & Kids',
        'Baby Bath': 'Baby & Kids',
        'toys': 'Baby & Kids',
        
        # Electronics & Accessories
        'Battery': 'Electronics & Accessories',
        'Charger': 'Electronics & Accessories',
        'Cable': 'Electronics & Accessories',
        'Headphones': 'Electronics & Accessories',
        'Speaker': 'Electronics & Accessories',
        'Phone': 'Electronics & Accessories',
        'Electronic': 'Electronics & Accessories',
        'Device': 'Electronics & Accessories',
        'Gadget': 'Electronics & Accessories',
        'Adapter': 'Electronics & Accessories',
        'Memory Card': 'Electronics & Accessories',
        
        # Stationery & Office
        'Pen': 'Stationery & Office',
        'Pencil': 'Stationery & Office',
        'Paper': 'Stationery & Office',
        'Notebook': 'Stationery & Office',
        'Stapler': 'Stationery & Office',
        'Tape': 'Stationery & Office',
        'Glue': 'Stationery & Office',
        'Marker': 'Stationery & Office',
        'Highlighter': 'Stationery & Office',
        'Envelope': 'Stationery & Office',
        'Folder': 'Stationery & Office',
        'Binder': 'Stationery & Office',
        'Calculator': 'Stationery & Office',
        
        # Home & Garden
        'Plant': 'Home & Garden',
        'Flower': 'Home & Garden',
        'Seed': 'Home & Garden',
        'Fertilizer': 'Home & Garden',
        'Garden': 'Home & Garden',
        'Pot': 'Home & Garden',
        'Vase': 'Home & Garden',
        'Candle': 'Home & Garden',
        'Decoration': 'Home & Garden',
        'Home Decor': 'Home & Garden',
        'Furniture': 'Home & Garden',
        'Storage': 'Home & Garden',
        'Organizer': 'Home & Garden',
        'Container': 'Home & Garden'
    }

def categorize_product_type(product_type):
    """Kategorisiere einen Produkttyp basierend auf dem Mapping"""
    if not product_type or pd.isna(product_type):
        return 'Other Products'
    
    # Direct mapping
    mapping = get_category_mapping()
    if product_type in mapping:
        return mapping[product_type]
    
    # Partial matching for compound product types
    product_type_lower = product_type.lower()
    
    # Check each mapping for partial matches
    for key, category in mapping.items():
        key_lower = key.lower()
        if (key_lower in product_type_lower or 
            product_type_lower in key_lower or
            any(word in product_type_lower for word in key_lower.split() if len(word) > 3)):
            return category
    
    # Special cases based on keywords
    if any(keyword in product_type_lower for keyword in ['cream', 'lotion', 'serum', 'gel', 'oil', 'wash', 'soap']):
        return 'Personal Care'
    elif any(keyword in product_type_lower for keyword in ['food', 'snack', 'cookie', 'cake', 'bread', 'rice', 'noodle']):
        return 'Food & Snacks'
    elif any(keyword in product_type_lower for keyword in ['drink', 'juice', 'coffee', 'tea', 'milk', 'soda', 'water']):
        return 'Beverages'
    elif any(keyword in product_type_lower for keyword in ['clean', 'detergent', 'wash']):
        return 'Household & Cleaning'
    elif any(keyword in product_type_lower for keyword in ['baby', 'child', 'kid', 'infant']):
        return 'Baby & Kids'
    elif any(keyword in product_type_lower for keyword in ['medicine', 'health', 'vitamin', 'supplement']):
        return 'Health & Pharmacy'
    
    # If no match found, keep as Other Products
    return 'Other Products'

def analyze_csv_files():
    """Analysiere alle CSV-Dateien um zu sehen was korrigiert werden muss"""
    
    csv_files = [
        'Analysis/csv_exports/eye_level_analysis.csv',
        'Analysis/csv_exports/brand_classification.csv', 
        'Analysis/csv_exports/cjmore_private_brands.csv'
    ]
    
    print("🔍 ANALYZING ALL CSV FILES")
    print("=" * 50)
    
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            print(f"\n📄 {csv_file}")
            print("-" * 30)
            
            df = pd.read_csv(csv_file)
            print(f"Total rows: {len(df)}")
            
            # Check if category column exists
            if 'category' in df.columns:
                category_counts = df['category'].value_counts()
                print("Categories:")
                for cat, count in category_counts.items():
                    print(f"  {cat}: {count}")
                
                # Check for unknown/other products
                unknown_categories = ['unknown', 'Unknown', 'Other Products']
                needs_fixing = df[df['category'].isin(unknown_categories)]
                
                if len(needs_fixing) > 0:
                    print(f"\n❌ Needs fixing: {len(needs_fixing)} products")
                    if 'type' in df.columns:
                        type_counts = needs_fixing['type'].value_counts()
                        print("Product types that need categorization:")
                        for ptype, count in type_counts.head(10).items():
                            print(f"  {ptype}: {count}")
                else:
                    print("✅ No products need fixing")
            else:
                print("⚠️ No 'category' column found")
                print("Columns available:", list(df.columns))
        else:
            print(f"❌ File not found: {csv_file}")

def fix_csv_categories():
    """Aktualisiere alle CSV-Dateien mit korrigierten Kategorien"""
    
    csv_files = [
        'Analysis/csv_exports/eye_level_analysis.csv',
        'Analysis/csv_exports/brand_classification.csv', 
        'Analysis/csv_exports/cjmore_private_brands.csv'
    ]
    
    print("🔧 FIXING CATEGORIES IN CSV FILES")
    print("=" * 50)
    
    total_fixed = 0
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"❌ File not found: {csv_file}")
            continue
            
        print(f"\n📄 Processing {csv_file}")
        print("-" * 30)
        
        # Load CSV
        df = pd.read_csv(csv_file)
        
        if 'category' not in df.columns or 'type' not in df.columns:
            print(f"⚠️ Missing required columns (category, type)")
            continue
        
        # Count items that need fixing
        needs_fixing = df[df['category'].isin(['unknown', 'Unknown', 'Other Products'])]
        print(f"Items needing categorization: {len(needs_fixing)}")
        
        if len(needs_fixing) == 0:
            print("✅ No items need fixing")
            continue
        
        # Apply categorization
        fixed_count = 0
        category_changes = {}
        
        for idx, row in df.iterrows():
            if row['category'] in ['unknown', 'Unknown', 'Other Products']:
                original_category = row['category']
                new_category = categorize_product_type(row['type'])
                
                if new_category != 'Other Products':  # Only update if we found a better category
                    df.at[idx, 'category'] = new_category
                    fixed_count += 1
                    
                    # Track changes
                    key = f"{row['type']} → {new_category}"
                    category_changes[key] = category_changes.get(key, 0) + 1
        
        # Save updated CSV
        df.to_csv(csv_file, index=False)
        print(f"✅ Fixed {fixed_count} categories")
        total_fixed += fixed_count
        
        # Show top changes
        if category_changes:
            print("Top category changes:")
            for change, count in sorted(category_changes.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"  {change}: {count} items")
    
    print(f"\n🎉 SUMMARY")
    print(f"Total items fixed across all files: {total_fixed}")
    return total_fixed

def validate_results():
    """Validiere die Ergebnisse nach der Kategorisierung"""
    
    csv_files = [
        'Analysis/csv_exports/eye_level_analysis.csv',
        'Analysis/csv_exports/brand_classification.csv', 
        'Analysis/csv_exports/cjmore_private_brands.csv'
    ]
    
    print("\n📊 VALIDATION RESULTS")
    print("=" * 40)
    
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            continue
            
        print(f"\n📄 {csv_file}")
        print("-" * 30)
        
        df = pd.read_csv(csv_file)
        
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            print("Final category distribution:")
            for cat, count in category_counts.items():
                percentage = count / len(df) * 100
                print(f"  {cat}: {count} ({percentage:.1f}%)")
            
            # Check remaining Other Products
            other_products = df[df['category'] == 'Other Products']
            if len(other_products) > 0:
                print(f"\nRemaining Other Products: {len(other_products)}")
                if 'type' in df.columns:
                    remaining_types = other_products['type'].value_counts()
                    print("Remaining product types:")
                    for ptype, count in remaining_types.head(5).items():
                        print(f"  {ptype}: {count}")

if __name__ == "__main__":
    # First analyze current state
    analyze_csv_files()
    
    # Fix categories
    fixed_count = fix_csv_categories()
    
    # Validate results
    validate_results()
    
    print(f"\n🎯 MISSION ACCOMPLISHED!")
    print(f"Successfully improved categorization for {fixed_count} products!")
    print("All CSV files have been updated with better categories.")