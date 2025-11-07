#!/usr/bin/env python3
"""
Category Standardization System for CJMore vs Tops Daily Comparison
==================================================================

Maps Tops Daily categories to CJMore category schema for direct comparison.
This ensures both supermarkets use the same categorical framework.

CJMore Standard Categories:
1. Food & Snacks
2. Personal Care  
3. Beverages
4. Other Products
5. Household & Cleaning
6. Baby & Kids
7. Health & Pharmacy
8. Home & Garden
"""

import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CategoryStandardizer:
    """Standardizes categories across supermarkets for comparison"""
    
    def __init__(self):
        """Initialize category mapping system"""
        
        # CJMore standard categories (target schema)
        self.standard_categories = [
            'Food & Snacks',
            'Personal Care', 
            'Beverages',
            'Other Products',
            'Household & Cleaning',
            'Baby & Kids',
            'Health & Pharmacy',
            'Home & Garden'
        ]
        
        # Mapping rules: Tops Daily → CJMore categories
        self.category_mapping = {
            # Food categories → Food & Snacks
            'Food & Snacks': 'Food & Snacks',
            'Food & Beverages': 'Food & Snacks',  # Split this intelligently
            'Food': 'Food & Snacks',
            'Fresh Food': 'Food & Snacks',
            
            # Beverages stay separate
            'Beverages': 'Beverages',
            
            # Personal Care consolidation
            'Personal Care': 'Personal Care',
            'Personal Care & Beauty': 'Personal Care',
            
            # Health & Pharmacy
            'Health & Pharmacy': 'Health & Pharmacy',
            
            # Household & Cleaning
            'Household & Cleaning': 'Household & Cleaning',
            
            # Baby & Kids
            'Baby & Kids': 'Baby & Kids',
            
            # Home & Garden  
            'Home & Garden': 'Home & Garden',
            
            # Pet care → Other Products (CJMore doesn't have pet category)
            'Pet Care': 'Other Products',
            'Pet Supplies': 'Other Products',
            
            # Electronics → Other Products
            'Electronics & Accessories': 'Other Products',
            
            # Miscellaneous → Other Products
            'Other': 'Other Products',
            'Other Products': 'Other Products',
            'Services': 'Other Products',
            'Store Supplies': 'Other Products'
        }
        
        # File paths
        self.tops_file = Path('tops_daily_analysis_output/csv_exports/tops_daily_brand_classification_enhanced.csv')
        self.output_file = Path('tops_daily_analysis_output/csv_exports/tops_daily_brand_classification_standardized.csv')
        
    def intelligent_food_beverage_split(self, df):
        """Intelligently split 'Food & Beverages' into 'Food & Snacks' and 'Beverages'"""
        
        print("🔬 Intelligently splitting 'Food & Beverages' category...")
        
        # Get products in Food & Beverages category
        food_bev_products = df[df['category'] == 'Food & Beverages'].copy()
        
        if len(food_bev_products) == 0:
            return df
        
        print(f"   📊 Found {len(food_bev_products)} products in 'Food & Beverages'")
        
        # Define beverage keywords
        beverage_keywords = [
            'beer', 'wine', 'alcohol', 'juice', 'drink', 'beverage', 'water', 'soda',
            'coffee', 'tea', 'energy drink', 'soft drink', 'cola', 'sprite', 'fanta'
        ]
        
        # Check product types for beverage indicators
        beverage_count = 0
        food_count = 0
        
        for idx, row in food_bev_products.iterrows():
            product_type_lower = str(row['product_type']).lower()
            brand_lower = str(row['brand']).lower()
            combined_text = f"{product_type_lower} {brand_lower}"
            
            # Check if it's a beverage
            is_beverage = any(keyword in combined_text for keyword in beverage_keywords)
            
            if is_beverage:
                df.loc[idx, 'category'] = 'Beverages'
                beverage_count += 1
            else:
                df.loc[idx, 'category'] = 'Food & Snacks'
                food_count += 1
        
        print(f"   ✅ Split into: {beverage_count} beverages, {food_count} food items")
        return df
    
    def standardize_tops_daily_categories(self):
        """Apply category standardization to Tops Daily data"""
        
        print("🎯 Standardizing Tops Daily Categories to CJMore Schema")
        print("=" * 60)
        
        # Load Tops Daily data
        if not self.tops_file.exists():
            raise FileNotFoundError(f"Tops Daily file not found: {self.tops_file}")
        
        df = pd.read_csv(self.tops_file)
        print(f"📊 Loaded {len(df):,} Tops Daily products")
        print(f"📋 Current categories: {df['category'].nunique()}")
        
        # Show current category distribution
        print("\nCurrent Category Distribution:")
        current_cats = df['category'].value_counts()
        for cat, count in current_cats.items():
            percentage = (count / len(df)) * 100
            print(f"  {cat}: {count:,} ({percentage:.1f}%)")
        
        # Step 1: Intelligent Food & Beverages split
        df = self.intelligent_food_beverage_split(df)
        
        # Step 2: Apply category mappings
        print(f"\n🔄 Applying category mappings...")
        mapping_stats = {}
        
        for old_category, new_category in self.category_mapping.items():
            if old_category in df['category'].values:
                count = len(df[df['category'] == old_category])
                df.loc[df['category'] == old_category, 'category'] = new_category
                mapping_stats[old_category] = {'new_category': new_category, 'count': count}
                print(f"   {old_category} → {new_category}: {count:,} products")
        
        # Step 3: Validate standardization
        print(f"\n✅ Standardization Complete!")
        print(f"📊 Final categories: {df['category'].nunique()}")
        
        # Show final category distribution
        print("\nStandardized Category Distribution:")
        final_cats = df['category'].value_counts()
        for cat, count in final_cats.items():
            percentage = (count / len(df)) * 100
            print(f"  {cat}: {count:,} ({percentage:.1f}%)")
        
        # Check for any unmapped categories
        unmapped_categories = set(df['category'].unique()) - set(self.standard_categories)
        if unmapped_categories:
            print(f"\n⚠️  Unmapped categories found: {unmapped_categories}")
            print("   These will be mapped to 'Other Products'")
            for cat in unmapped_categories:
                df.loc[df['category'] == cat, 'category'] = 'Other Products'
        
        return df
    
    def save_standardized_data(self, df):
        """Save standardized dataset"""
        
        print(f"\n💾 Saving Standardized Dataset...")
        
        # Add standardization metadata
        df['standardization_version'] = 'v1.0_cjmore_schema'
        df['standardization_timestamp'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save standardized dataset
        df.to_csv(self.output_file, index=False)
        
        # Also update the original enhanced file for consistency
        df.to_csv(self.tops_file, index=False)
        
        print(f"   ✅ Saved: {self.output_file.name}")
        print(f"   ✅ Updated: {self.tops_file.name}")
        print(f"   📊 Dataset: {len(df):,} products with standardized categories")
    
    def create_comparison_report(self, df):
        """Create comparison report with CJMore"""
        
        print(f"\n📊 Creating CJMore vs Tops Daily Comparison...")
        
        # Load CJMore data
        cjmore_file = Path('Analysis/Supermarket_Analysis_Complete.xlsx')
        if cjmore_file.exists():
            df_cjmore = pd.read_excel(cjmore_file, sheet_name='Brand Classification')
            
            # Compare category distributions
            comparison_data = []
            
            for category in self.standard_categories:
                cjmore_count = len(df_cjmore[df_cjmore['category'] == category])
                cjmore_pct = (cjmore_count / len(df_cjmore)) * 100
                
                tops_count = len(df[df['category'] == category])
                tops_pct = (tops_count / len(df)) * 100
                
                comparison_data.append({
                    'Category': category,
                    'CJMore_Products': cjmore_count,
                    'CJMore_Percentage': cjmore_pct,
                    'Tops_Daily_Products': tops_count,
                    'Tops_Daily_Percentage': tops_pct,
                    'Difference_Products': tops_count - cjmore_count,
                    'Difference_Percentage': tops_pct - cjmore_pct
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            print("\nCategory Comparison (CJMore vs Tops Daily):")
            print("=" * 80)
            for _, row in comparison_df.iterrows():
                print(f"{row['Category']:20s} | CJMore: {row['CJMore_Products']:4d} ({row['CJMore_Percentage']:5.1f}%) | "
                      f"Tops: {row['Tops_Daily_Products']:4d} ({row['Tops_Daily_Percentage']:5.1f}%) | "
                      f"Diff: {row['Difference_Percentage']:+6.1f}%")
            
            # Save comparison
            comparison_file = Path('Market_Comparison_Analysis/category_comparison.csv')
            comparison_file.parent.mkdir(parents=True, exist_ok=True)
            comparison_df.to_csv(comparison_file, index=False)
            print(f"\n   ✅ Comparison saved: {comparison_file.name}")
            
        else:
            print("   ⚠️  CJMore data not found - skipping comparison")
    
    def run_standardization(self):
        """Execute complete category standardization process"""
        
        print("🚀 CATEGORY STANDARDIZATION SYSTEM")
        print("=" * 60)
        print("🎯 Mission: Align Tops Daily categories with CJMore schema")
        print("📋 Target: 8 standardized categories for direct comparison")
        print("🔄 Method: Intelligent mapping + beverage splitting")
        print()
        
        # Run standardization
        standardized_df = self.standardize_tops_daily_categories()
        
        # Save results
        self.save_standardized_data(standardized_df)
        
        # Create comparison
        self.create_comparison_report(standardized_df)
        
        print("\n🏆 STANDARDIZATION COMPLETE!")
        print("✅ Tops Daily categories now aligned with CJMore schema")
        print("📊 Ready for direct supermarket comparison analysis")
        print("🎯 Both datasets use identical category framework")
        print("=" * 60)

def main():
    """Main execution function"""
    
    try:
        # Create and run standardization system
        standardizer = CategoryStandardizer()
        standardizer.run_standardization()
        
    except Exception as e:
        print(f"❌ Standardization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())