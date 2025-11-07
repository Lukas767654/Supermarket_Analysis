#!/usr/bin/env python3
"""
🎨 SUPERMARKET GENERAL DATA VISUALIZATION SYSTEM
==============================================================================
Professional visualization system for general statistics and brand diversity
analysis of CJMore and Tops Daily supermarkets.

Features:
- Total product counts and portfolio overview
- Brand diversity and concentration analysis
- Top brands ranking and distribution
- Brand origin analysis (Local vs International)
- Product concentration metrics
- Comprehensive general statistics dashboard

Color Scheme:
- CJMore: #96B991 (Green)
- Tops Daily: #EF865B (Orange)
- Accent: #BFDAEF (Light Blue)
- Neutral: #F5F5F5 (Light Gray)
==============================================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import os
from collections import Counter
import matplotlib.patches as mpatches

class SupermarketGeneralDataVisualizer:
    def __init__(self):
        """Initialize the general data visualization system."""
        self.colors = {
            'cjmore': '#96B991',
            'tops_daily': '#EF865B', 
            'accent': '#BFDAEF',
            'neutral': '#F5F5F5',
            'dark_gray': '#666666',
            'light_gray': '#CCCCCC'
        }
        
        # Set professional style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create output directory
        self.output_dir = Path('final_results/comprehensive_analysis/general_data_visualizations')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data
        self.cjmore_data = None
        self.tops_daily_data = None
        
    def load_data(self):
        """Load and prepare datasets for visualization - same as create_visual_comparisons.py"""
        
        print("📂 Loading Supermarket Data for General Visualization...")
        
        # Data paths - same as create_visual_comparisons.py
        self.cjmore_file = Path('final_results/cjmore_data/CJMore_Complete_Analysis.xlsx')
        self.tops_file = Path('final_results/tops_daily_data/tops_daily_brand_classification_enhanced.csv')
        
        # Load CJMore data
        if self.cjmore_file.exists():
            self.cjmore_data = pd.read_excel(self.cjmore_file, sheet_name='Brand Classification')
            print(f"   ✅ CJMore: {len(self.cjmore_data):,} products loaded")
        else:
            raise FileNotFoundError(f"CJMore file not found: {self.cjmore_file}")
        
        # Load Tops Daily data
        if self.tops_file.exists():
            self.tops_daily_data = pd.read_csv(self.tops_file)
            print(f"   ✅ Tops Daily: {len(self.tops_daily_data):,} products loaded")
        else:
            raise FileNotFoundError(f"Tops Daily file not found: {self.tops_file}")
        
        # Add supermarket identifiers
        self.cjmore_data['supermarket'] = 'CJMore'
        self.tops_daily_data['supermarket'] = 'Tops Daily'
        
        # Standardize column names for both datasets
        for col in ['brand', 'product_type', 'category', 'origin']:
            if col not in self.cjmore_data.columns:
                self.cjmore_data[col] = 'Unknown'
            if col not in self.tops_daily_data.columns:
                self.tops_daily_data[col] = 'Unknown'
        
        # Fill missing values
        self.cjmore_data = self.cjmore_data.fillna('Unknown')
        self.tops_daily_data = self.tops_daily_data.fillna('Unknown')
        
        # Create standardized_category column for compatibility
        if 'standardized_category' not in self.cjmore_data.columns:
            self.cjmore_data['standardized_category'] = self.cjmore_data['category']
        if 'standardized_category' not in self.tops_daily_data.columns:
            self.tops_daily_data['standardized_category'] = self.tops_daily_data['category']
            
        return True
        
    def create_portfolio_overview(self):
        """Create comprehensive portfolio overview visualization."""
        print("📊 Creating Portfolio Overview...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🏪 SUPERMARKET PORTFOLIO OVERVIEW', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Total Product Count Comparison
        stores = ['CJMore', 'Tops Daily']
        product_counts = [len(self.cjmore_data), len(self.tops_daily_data)]
        colors = [self.colors['cjmore'], self.colors['tops_daily']]
        
        bars1 = ax1.bar(stores, product_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('📦 Total Product Portfolio', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=12)
        
        # Add value labels on bars
        for bar, count in zip(bars1, product_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, max(product_counts) * 1.2)
        
        # 2. Category Count Comparison
        cjmore_categories = self.cjmore_data['standardized_category'].nunique()
        tops_categories = self.tops_daily_data['standardized_category'].nunique()
        
        category_counts = [cjmore_categories, tops_categories]
        bars2 = ax2.bar(stores, category_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_title('🏷️ Category Coverage', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Categories', fontsize=12)
        
        for bar, count in zip(bars2, category_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, max(category_counts) * 1.3)
        
        # 3. Brand Count Comparison
        cjmore_brands = self.cjmore_data['brand'].nunique()
        tops_brands = self.tops_daily_data['brand'].nunique()
        
        brand_counts = [cjmore_brands, tops_brands]
        bars3 = ax3.bar(stores, brand_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_title('🏭 Brand Diversity', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Brands', fontsize=12)
        
        for bar, count in zip(bars3, brand_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim(0, max(brand_counts) * 1.2)
        
        # 4. Products per Brand Average
        cjmore_avg = len(self.cjmore_data) / cjmore_brands
        tops_avg = len(self.tops_daily_data) / tops_brands
        
        avg_products = [cjmore_avg, tops_avg]
        bars4 = ax4.bar(stores, avg_products, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_title('📈 Average Products per Brand', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Products per Brand', fontsize=12)
        
        for bar, avg in zip(bars4, avg_products):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{avg:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax4.grid(axis='y', alpha=0.3)
        ax4.set_ylim(0, max(avg_products) * 1.2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_portfolio_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Portfolio Overview saved")
        
    def create_brand_diversity_analysis(self):
        """Create comprehensive brand diversity visualization."""
        print("🏭 Creating Brand Diversity Analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('🏭 BRAND DIVERSITY & CONCENTRATION ANALYSIS', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Top 20 Brands by Product Count (Combined)
        all_brands = pd.concat([
            self.cjmore_data[['brand']].assign(store='CJMore'),
            self.tops_daily_data[['brand']].assign(store='Tops Daily')
        ])
        
        brand_counts = all_brands['brand'].value_counts().head(20)
        
        bars1 = ax1.barh(range(len(brand_counts)), brand_counts.values, 
                        color=self.colors['accent'], alpha=0.8, edgecolor='black')
        ax1.set_yticks(range(len(brand_counts)))
        ax1.set_yticklabels(brand_counts.index, fontsize=10)
        ax1.set_title('🏆 Top 20 Brands by Total Products', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Number of Products', fontsize=12)
        
        # Add value labels
        for i, v in enumerate(brand_counts.values):
            ax1.text(v + 1, i, str(v), va='center', fontweight='bold')
        
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # 2. Brand Distribution by Store
        cjmore_brand_counts = self.cjmore_data['brand'].value_counts().head(15)
        tops_brand_counts = self.tops_daily_data['brand'].value_counts().head(15)
        
        y_pos = np.arange(len(cjmore_brand_counts))
        
        ax2.barh(y_pos - 0.2, cjmore_brand_counts.values, 0.4, 
                label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        
        # For Tops Daily, we need to align with CJMore brands or show separate
        tops_aligned = []
        for brand in cjmore_brand_counts.index:
            if brand in tops_brand_counts.index:
                tops_aligned.append(tops_brand_counts[brand])
            else:
                tops_aligned.append(0)
        
        ax2.barh(y_pos + 0.2, tops_aligned, 0.4,
                label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(cjmore_brand_counts.index, fontsize=9)
        ax2.set_title('📊 Top Brands Comparison by Store', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Number of Products', fontsize=12)
        ax2.legend()
        ax2.grid(axis='x', alpha=0.3)
        ax2.invert_yaxis()
        
        # 3. Brand Concentration Analysis (Pareto Chart)
        cjmore_cumsum = (cjmore_brand_counts.cumsum() / len(self.cjmore_data) * 100)
        
        ax3_twin = ax3.twinx()
        
        bars3 = ax3.bar(range(len(cjmore_brand_counts)), cjmore_brand_counts.values,
                       color=self.colors['cjmore'], alpha=0.7, label='Product Count')
        
        line3 = ax3_twin.plot(range(len(cjmore_cumsum)), cjmore_cumsum.values,
                             'ro-', linewidth=2, markersize=6, label='Cumulative %')
        
        ax3.set_title('📈 CJMore Brand Concentration (Pareto)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Top Brands (Ranked)', fontsize=12)
        ax3.set_ylabel('Number of Products', fontsize=12, color=self.colors['cjmore'])
        ax3_twin.set_ylabel('Cumulative Percentage', fontsize=12, color='red')
        
        # 80/20 line
        ax3_twin.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Line')
        
        ax3.tick_params(axis='x', labelrotation=45)
        ax3.grid(alpha=0.3)
        
        # 4. Brand Origin Distribution (if available)
        # We'll create a mock analysis based on brand names
        def analyze_brand_origin(brands):
            """Simple heuristic to classify brand origins."""
            local_indicators = ['thai', 'thailand', 'siam', 'lotus', 'cp', 'tesco', 'big c']
            international_indicators = ['coca', 'pepsi', 'nestle', 'unilever', 'p&g', 'johnson']
            
            local_count = 0
            international_count = 0
            unknown_count = 0
            
            for brand in brands:
                brand_lower = brand.lower()
                if any(indicator in brand_lower for indicator in local_indicators):
                    local_count += 1
                elif any(indicator in brand_lower for indicator in international_indicators):
                    international_count += 1
                else:
                    unknown_count += 1
            
            return local_count, international_count, unknown_count
        
        cjmore_origins = analyze_brand_origin(self.cjmore_data['brand'].unique())
        tops_origins = analyze_brand_origin(self.tops_daily_data['brand'].unique())
        
        categories = ['Local/Regional', 'International', 'Unknown']
        cjmore_values = list(cjmore_origins)
        tops_values = list(tops_origins)
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars4_1 = ax4.bar(x - width/2, cjmore_values, width, 
                         label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars4_2 = ax4.bar(x + width/2, tops_values, width,
                         label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax4.set_title('🌍 Brand Origin Distribution', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Brands', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars4_1, bars4_2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                            f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_brand_diversity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Brand Diversity Analysis saved")
        
    def create_category_distribution_overview(self):
        """Create category distribution overview."""
        print("📋 Creating Category Distribution Overview...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📋 CATEGORY DISTRIBUTION OVERVIEW', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. Category product count - CJMore
        cjmore_categories = self.cjmore_data['standardized_category'].value_counts()
        
        bars1 = ax1.bar(range(len(cjmore_categories)), cjmore_categories.values,
                       color=self.colors['cjmore'], alpha=0.8, edgecolor='black')
        ax1.set_title('🛒 CJMore - Products per Category', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Categories', fontsize=12)
        ax1.set_ylabel('Number of Products', fontsize=12)
        ax1.set_xticks(range(len(cjmore_categories)))
        ax1.set_xticklabels(cjmore_categories.index, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on top of bars
        for bar, count in zip(bars1, cjmore_categories.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                    f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 2. Category product count - Tops Daily
        tops_categories = self.tops_daily_data['standardized_category'].value_counts()
        
        bars2 = ax2.bar(range(len(tops_categories)), tops_categories.values,
                       color=self.colors['tops_daily'], alpha=0.8, edgecolor='black')
        ax2.set_title('🛒 Tops Daily - Products per Category', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Categories', fontsize=12)
        ax2.set_ylabel('Number of Products', fontsize=12)
        ax2.set_xticks(range(len(tops_categories)))
        ax2.set_xticklabels(tops_categories.index, rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar, count in zip(bars2, tops_categories.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{count}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 3. Combined category comparison
        all_categories = set(cjmore_categories.index) | set(tops_categories.index)
        
        cjmore_aligned = [cjmore_categories.get(cat, 0) for cat in sorted(all_categories)]
        tops_aligned = [tops_categories.get(cat, 0) for cat in sorted(all_categories)]
        
        x = np.arange(len(all_categories))
        width = 0.35
        
        bars3_1 = ax3.bar(x - width/2, cjmore_aligned, width,
                         label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars3_2 = ax3.bar(x + width/2, tops_aligned, width,
                         label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax3.set_title('⚖️ Category Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Products', fontsize=12)
        ax3.set_xticks(x)
        ax3.set_xticklabels(sorted(all_categories), rotation=45, ha='right')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
        
        # 4. Category dominance (percentage)
        total_cjmore = len(self.cjmore_data)
        total_tops = len(self.tops_daily_data)
        
        cjmore_pct = [(count/total_cjmore)*100 for count in cjmore_aligned]
        tops_pct = [(count/total_tops)*100 for count in tops_aligned]
        
        bars4_1 = ax4.bar(x - width/2, cjmore_pct, width,
                         label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars4_2 = ax4.bar(x + width/2, tops_pct, width,
                         label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax4.set_title('📊 Category Share (%)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Percentage of Total Products', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(sorted(all_categories), rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_category_distribution_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Category Distribution Overview saved")
        
    def create_comprehensive_dashboard(self):
        """Create comprehensive general statistics dashboard."""
        print("📊 Creating Comprehensive General Dashboard...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('📊 COMPREHENSIVE SUPERMARKET DATA DASHBOARD', fontsize=24, fontweight='bold', y=0.95)
        
        # Key Statistics Summary (Top row)
        ax_summary = fig.add_subplot(gs[0, :])
        ax_summary.axis('off')
        
        # Create summary statistics
        cjmore_stats = {
            'Products': f"{len(self.cjmore_data):,}",
            'Categories': f"{self.cjmore_data['standardized_category'].nunique()}",
            'Brands': f"{self.cjmore_data['brand'].nunique():,}",
            'Avg Products/Brand': f"{len(self.cjmore_data)/self.cjmore_data['brand'].nunique():.1f}"
        }
        
        tops_stats = {
            'Products': f"{len(self.tops_daily_data):,}",
            'Categories': f"{self.tops_daily_data['standardized_category'].nunique()}",
            'Brands': f"{self.tops_daily_data['brand'].nunique():,}",
            'Avg Products/Brand': f"{len(self.tops_daily_data)/self.tops_daily_data['brand'].nunique():.1f}"
        }
        
        # Summary table
        summary_text = f"""
        ┌─────────────────────────────────────────────────────────────────────────────────────────┐
        │                           🏪 SUPERMARKET PORTFOLIO SUMMARY                               │
        ├─────────────────────────────────────────────────────────────────────────────────────────┤
        │                    CJMore                    │                Tops Daily                 │
        ├─────────────────────────────────────────────────────────────────────────────────────────┤
        │ Products:          {cjmore_stats['Products']:>10} │ Products:          {tops_stats['Products']:>10} │
        │ Categories:        {cjmore_stats['Categories']:>10} │ Categories:        {tops_stats['Categories']:>10} │
        │ Brands:            {cjmore_stats['Brands']:>10} │ Brands:            {tops_stats['Brands']:>10} │
        │ Avg Prod/Brand:    {cjmore_stats['Avg Products/Brand']:>10} │ Avg Prod/Brand:    {tops_stats['Avg Products/Brand']:>10} │
        └─────────────────────────────────────────────────────────────────────────────────────────┘
        """
        
        ax_summary.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center', 
                       fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", 
                       facecolor=self.colors['neutral'], alpha=0.8))
        
        # Product Portfolio Comparison
        ax1 = fig.add_subplot(gs[1, 0])
        stores = ['CJMore', 'Tops Daily']
        product_counts = [len(self.cjmore_data), len(self.tops_daily_data)]
        colors = [self.colors['cjmore'], self.colors['tops_daily']]
        
        bars = ax1.bar(stores, product_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_title('📦 Product Portfolio', fontweight='bold')
        ax1.set_ylabel('Products')
        
        for bar, count in zip(bars, product_counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Brand Diversity
        ax2 = fig.add_subplot(gs[1, 1])
        brand_counts = [self.cjmore_data['brand'].nunique(), self.tops_daily_data['brand'].nunique()]
        
        bars = ax2.bar(stores, brand_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.set_title('🏭 Brand Diversity', fontweight='bold')
        ax2.set_ylabel('Unique Brands')
        
        for bar, count in zip(bars, brand_counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 20,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # Category Coverage
        ax3 = fig.add_subplot(gs[1, 2])
        category_counts = [self.cjmore_data['standardized_category'].nunique(), 
                          self.tops_daily_data['standardized_category'].nunique()]
        
        bars = ax3.bar(stores, category_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax3.set_title('🏷️ Categories', fontweight='bold')
        ax3.set_ylabel('Category Count')
        
        for bar, count in zip(bars, category_counts):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # Portfolio Efficiency
        ax4 = fig.add_subplot(gs[1, 3])
        efficiency = [len(self.cjmore_data)/self.cjmore_data['brand'].nunique(),
                     len(self.tops_daily_data)/self.tops_daily_data['brand'].nunique()]
        
        bars = ax4.bar(stores, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_title('📈 Products/Brand', fontweight='bold')
        ax4.set_ylabel('Efficiency Ratio')
        
        for bar, eff in zip(bars, efficiency):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{eff:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Top Categories Combined
        ax5 = fig.add_subplot(gs[2, :2])
        
        all_categories = pd.concat([
            self.cjmore_data['standardized_category'].value_counts(),
            self.tops_daily_data['standardized_category'].value_counts()
        ], axis=1, keys=['CJMore', 'Tops Daily']).fillna(0)
        
        all_categories['Total'] = all_categories.sum(axis=1)
        top_categories = all_categories.sort_values('Total', ascending=False).head(8)
        
        x = np.arange(len(top_categories))
        width = 0.35
        
        bars1 = ax5.bar(x - width/2, top_categories['CJMore'], width,
                       label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars2 = ax5.bar(x + width/2, top_categories['Tops Daily'], width,
                       label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax5.set_title('🏆 Top Categories by Total Products', fontweight='bold')
        ax5.set_ylabel('Number of Products')
        ax5.set_xticks(x)
        ax5.set_xticklabels(top_categories.index, rotation=45, ha='right')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)
        
        # Top Brands Combined
        ax6 = fig.add_subplot(gs[2, 2:])
        
        all_brands = pd.concat([
            self.cjmore_data['brand'].value_counts(),
            self.tops_daily_data['brand'].value_counts()
        ], axis=1, keys=['CJMore', 'Tops Daily']).fillna(0)
        
        all_brands['Total'] = all_brands.sum(axis=1)
        top_brands = all_brands.sort_values('Total', ascending=False).head(10)
        
        x = np.arange(len(top_brands))
        
        bars1 = ax6.bar(x - width/2, top_brands['CJMore'], width,
                       label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars2 = ax6.bar(x + width/2, top_brands['Tops Daily'], width,
                       label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax6.set_title('🥇 Top Brands by Total Products', fontweight='bold')
        ax6.set_ylabel('Number of Products')
        ax6.set_xticks(x)
        ax6.set_xticklabels(top_brands.index, rotation=45, ha='right')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
        
        # Market Share Analysis
        ax7 = fig.add_subplot(gs[3, :2])
        
        total_products = len(self.cjmore_data) + len(self.tops_daily_data)
        market_share = [len(self.cjmore_data)/total_products*100, 
                       len(self.tops_daily_data)/total_products*100]
        
        wedges, texts, autotexts = ax7.pie(market_share, labels=stores, colors=colors,
                                          autopct='%1.1f%%', startangle=90, 
                                          textprops={'fontweight': 'bold', 'fontsize': 12})
        
        ax7.set_title('📊 Market Share by Products', fontweight='bold')
        
        # Diversity Index
        ax8 = fig.add_subplot(gs[3, 2:])
        
        # Calculate diversity metrics
        cjmore_diversity = self.cjmore_data['standardized_category'].nunique() / len(self.cjmore_data) * 1000
        tops_diversity = self.tops_daily_data['standardized_category'].nunique() / len(self.tops_daily_data) * 1000
        
        cjmore_brand_diversity = self.cjmore_data['brand'].nunique() / len(self.cjmore_data) * 1000
        tops_brand_diversity = self.tops_daily_data['brand'].nunique() / len(self.tops_daily_data) * 1000
        
        metrics = ['Category\nDiversity', 'Brand\nDiversity']
        cjmore_values = [cjmore_diversity, cjmore_brand_diversity]
        tops_values = [tops_diversity, tops_brand_diversity]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax8.bar(x - width/2, cjmore_values, width,
                       label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars2 = ax8.bar(x + width/2, tops_values, width,
                       label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax8.set_title('🎯 Diversity Index\n(per 1000 products)', fontweight='bold')
        ax8.set_ylabel('Diversity Score')
        ax8.set_xticks(x)
        ax8.set_xticklabels(metrics)
        ax8.legend()
        ax8.grid(axis='y', alpha=0.3)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax8.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.savefig(self.output_dir / '4_comprehensive_general_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Comprehensive General Dashboard saved")
        
    def generate_all_visualizations(self):
        """Generate all general data visualizations."""
        print("🎨 General Data Visualization System Initialized")
        print(f"📊 CJMore: {len(self.cjmore_data):,} products")
        print(f"📊 Tops Daily: {len(self.tops_daily_data):,} products")
        print("🚀 GENERAL DATA VISUALIZATION SUITE")
        print("=" * 78)
        print("🎨 Creating comprehensive general statistics and brand diversity charts")
        print(f"📊 Color Scheme: CJMore ({self.colors['cjmore']}) vs Tops Daily ({self.colors['tops_daily']})")
        print()
        
        # Generate all visualizations
        self.create_portfolio_overview()
        self.create_brand_diversity_analysis()
        self.create_category_distribution_overview()
        self.create_comprehensive_dashboard()
        
        print()
        print("=" * 78)
        print("🏆 GENERAL DATA VISUALIZATION SUITE COMPLETE!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print("📊 Generated Charts:")
        print("   1. Portfolio Overview (Total products, categories, brands)")
        print("   2. Brand Diversity Analysis (Top brands, concentration, origins)")
        print("   3. Category Distribution Overview (Category breakdowns)")
        print("   4. Comprehensive General Dashboard (Executive overview)")
        print()
        print("🎯 Ready for strategic presentation and portfolio analysis!")
        print("=" * 78)

def main():
    """Main execution function."""
    print("🎨 SUPERMARKET GENERAL DATA VISUALIZATION SYSTEM")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = SupermarketGeneralDataVisualizer()
    
    # Load data
    if not visualizer.load_data():
        print("❌ Failed to load data. Exiting.")
        return
    
    # Generate all visualizations
    visualizer.generate_all_visualizations()

if __name__ == "__main__":
    main()