#!/usr/bin/env python3
"""
Post-processing script to consolidate and optimize the enhanced categories
"""

import json
import pandas as pd
from collections import defaultdict
import re

def optimize_categories(input_path: str, output_path: str):
    """Consolidate and optimize the enhanced categories"""
    
    print("🔧 Optimizing category structure...")
    
    # Load enhanced data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Define category consolidation rules
    consolidation_rules = {
        # Beauty & Personal Care consolidation
        'Personal Care & Beauty': [
            'Hair Powder Products',
            'Facial Powder Products', 
            'Body Cream Products',
            'Hair Lip Products',
            'Lip Food Products',
            'Facial Serum Products',
            'Hair Conditioner Products',
            'Hair Hair Treatment Products',
            'Cream Face Products',
            'Body Body Wash Products',
            'Cream Skin Products',
            'Facial Cleanser Products',
            'Lotion Body Products',
            'Lip Product',
            'Care Skin Products',
            'Serum Gel Products',
            'Personal Care'
        ],
        
        # Food & Beverages consolidation  
        'Food & Beverages': [
            'Powder Sauce Products',
            'Chicken Sauce Products',
            'Frozen Sauce Products',
            'Sauce Lip Products',
            'Powder Seasoning Products',
            'Cream Ice Products',
            'Instant Soup Products',
            'Sauce Chili Sauce Products',
            'Jelly Chewy Products',
            'Instant Cup Products',
            'Flour Starch Products',
            'Flour Tapioca Products',
            'Cake Fish Cake Products',
            'Food & Snacks',
            'Beverages',
            'Frozen Chicken Products',
            'Frozen Fries Products'
        ],
        
        # Pet Care consolidation
        'Pet Care & Accessories': [
            'Cat Dog Products',
            'Cat Cat Food Products'
        ],
        
        # Keep existing major categories as-is
        'Household & Cleaning': ['Household & Cleaning'],
        'Baby & Kids': ['Baby & Kids'],
        'Health & Pharmacy': ['Health & Pharmacy'],
        'Electronics & Accessories': ['Electronics & Accessories'],
        'Stationery & Office': ['Stationery & Office'],
        'Home & Garden': ['Home & Garden'],
        
        # Services & Other
        'Services & Other': [
            'Store Retail Products',
            'Service Payment Products'
        ]
    }
    
    # Create reverse mapping
    category_mapping = {}
    for target_category, source_categories in consolidation_rules.items():
        for source_category in source_categories:
            category_mapping[source_category] = target_category
    
    # Apply consolidation
    print("🔄 Consolidating categories...")
    
    consolidation_stats = defaultdict(int)
    
    for idx, row in df.iterrows():
        original_category = row['category_display_name']
        
        if original_category in category_mapping:
            new_category = category_mapping[original_category]
            df.at[idx, 'category_display_name'] = new_category
            df.at[idx, 'main_category'] = new_category.lower().replace(' & ', '_').replace(' ', '_')
            consolidation_stats[f"{original_category} → {new_category}"] += 1
    
    # Generate final statistics
    final_counts = df['category_display_name'].value_counts()
    
    print("\n📊 FINAL CATEGORY DISTRIBUTION")
    print("=" * 50)
    
    for category, count in final_counts.items():
        percentage = count / len(df) * 100
        print(f"{category:<30} {count:>5} ({percentage:5.1f}%)")
    
    print(f"\nTotal categories: {len(final_counts)}")
    print(f"Total products: {len(df)}")
    
    # Show consolidation actions
    if consolidation_stats:
        print("\n🔄 CONSOLIDATION ACTIONS")
        print("=" * 30)
        for action, count in consolidation_stats.items():
            print(f"{action}: {count} products")
    
    # Save optimized results
    optimized_data = df.to_dict('records')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(optimized_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Optimized results saved to: {output_path}")
    
    return optimized_data, final_counts

def main():
    """Run category optimization"""
    
    input_path = "./enhanced_results_improved_categories.json"
    output_path = "./enhanced_results_final_categories.json"
    
    print("🚀 Category Structure Optimization")
    print("=" * 40)
    
    try:
        optimized_data, final_counts = optimize_categories(input_path, output_path)
        
        print("\n🎉 Category optimization complete!")
        print(f"📊 Reduced from 24 categories to {len(final_counts)} business-friendly categories")
        print("📈 Ready for business analytics and reporting!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()