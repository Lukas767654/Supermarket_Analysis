#!/usr/bin/env python3
"""
CJMore vs Tops Daily Competitive Market Analysis
==============================================

Comprehensive comparative analysis between CJMore and Tops Daily supermarkets
including market positioning, strategy differences, and competitive insights.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

class SupermarketCompetitiveAnalysis:
    """Comprehensive competitive analysis between CJMore and Tops Daily"""
    
    def __init__(self):
        # Paths for both supermarkets
        self.cjmore_dir = Path('cjmore_analysis_output/csv_exports')
        self.tops_dir = Path('tops_daily_analysis_output/csv_exports')
        self.output_dir = Path('Market_Comparison_Analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Modern color palette
        self.colors = {
            'cjmore': '#2E86AB',      # Blue for CJMore  
            'tops': '#A23B72',        # Purple for Tops Daily
            'accent1': '#F18F01',     # Orange
            'accent2': '#6A994E',     # Green
            'neutral1': '#577590',    # Gray-blue
            'neutral2': '#F2CC8F',    # Light orange
            'light': '#F8F9FA',       # Light gray
            'dark': '#495057'         # Dark gray
        }
        
        # Load data
        self.load_datasets()
    
    def load_datasets(self):
        """Load data from both supermarkets"""
        
        print("📊 Loading Competitive Analysis Datasets...")
        
        # Load CJMore data
        cjmore_path = self.cjmore_dir / 'cjmore_brand_classification_enhanced.csv'
        if cjmore_path.exists():
            self.df_cjmore = pd.read_csv(cjmore_path)
            print(f"   ✅ CJMore: {len(self.df_cjmore)} products, {self.df_cjmore['brand'].nunique()} brands")
        else:
            print(f"   ❌ CJMore data not found at {cjmore_path}")
            return
        
        # Load Tops Daily data  
        tops_path = self.tops_dir / 'tops_daily_brand_classification_enhanced.csv'
        if tops_path.exists():
            self.df_tops = pd.read_csv(tops_path)
            print(f"   ✅ Tops Daily: {len(self.df_tops)} products, {self.df_tops['brand'].nunique()} brands")
        else:
            print(f"   ❌ Tops Daily data not found at {tops_path}")
            return
        
        # Add supermarket identifier
        self.df_cjmore['supermarket'] = 'CJMore'
        self.df_tops['supermarket'] = 'Tops Daily'
        
        # Combine datasets for comparative analysis
        self.df_combined = pd.concat([self.df_cjmore, self.df_tops], ignore_index=True)
        
        print(f"   ✅ Combined Dataset: {len(self.df_combined)} total products")
    
    def create_market_overview_comparison(self):
        """Create market overview comparison between both supermarkets"""
        
        print("\n🏪 Creating Market Overview Comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CJMore vs Tops Daily: Market Overview Comparison', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Portfolio Size Comparison
        portfolio_stats = {
            'CJMore': len(self.df_cjmore),
            'Tops Daily': len(self.df_tops)
        }
        
        bars = ax1.bar(range(len(portfolio_stats)), list(portfolio_stats.values()),
                      color=[self.colors['cjmore'], self.colors['tops']])
        ax1.set_xticks(range(len(portfolio_stats)))
        ax1.set_xticklabels(list(portfolio_stats.keys()))
        ax1.set_ylabel('Number of Products')
        ax1.set_title('Portfolio Size Comparison', fontweight='bold', pad=20)
        ax1.grid(axis='y', alpha=0.3, color=self.colors['light'])
        
        # Add value labels
        for bar, (name, value) in zip(bars, portfolio_stats.items()):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(portfolio_stats.values()) * 0.01,
                    f'{value:,}', ha='center', fontweight='bold', fontsize=12)
        
        # 2. Brand Diversity Comparison
        brand_stats = {
            'CJMore': self.df_cjmore['brand'].nunique(),
            'Tops Daily': self.df_tops['brand'].nunique()
        }
        
        bars = ax2.bar(range(len(brand_stats)), list(brand_stats.values()),
                      color=[self.colors['cjmore'], self.colors['tops']])
        ax2.set_xticks(range(len(brand_stats)))
        ax2.set_xticklabels(list(brand_stats.keys()))
        ax2.set_ylabel('Number of Unique Brands')
        ax2.set_title('Brand Diversity Comparison', fontweight='bold', pad=20)
        ax2.grid(axis='y', alpha=0.3, color=self.colors['light'])
        
        # Add value labels
        for bar, (name, value) in zip(bars, brand_stats.items()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(brand_stats.values()) * 0.01,
                    f'{value:,}', ha='center', fontweight='bold', fontsize=12)
        
        # 3. Market Origin Strategy
        origin_comparison = self.df_combined.groupby(['supermarket', 'origin']).size().unstack(fill_value=0)
        origin_pct = origin_comparison.div(origin_comparison.sum(axis=1), axis=0) * 100
        
        x_pos = np.arange(len(origin_pct.columns))
        width = 0.35
        
        bars1 = ax3.bar(x_pos - width/2, origin_pct.iloc[0], width, label='CJMore', color=self.colors['cjmore'])
        bars2 = ax3.bar(x_pos + width/2, origin_pct.iloc[1], width, label='Tops Daily', color=self.colors['tops'])
        
        ax3.set_xlabel('Brand Origin')
        ax3.set_ylabel('Percentage of Products')
        ax3.set_title('Market Origin Strategy Comparison', fontweight='bold', pad=20)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(origin_pct.columns)
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3, color=self.colors['light'])
        
        # Add percentage labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # 4. Category Coverage Analysis
        category_comparison = self.df_combined.groupby(['supermarket', 'category']).size().unstack(fill_value=0, dropna=False)
        
        # Calculate category overlap
        cjmore_categories = set(self.df_cjmore['category'].unique())
        tops_categories = set(self.df_tops['category'].unique())
        overlap = len(cjmore_categories.intersection(tops_categories))
        cjmore_unique = len(cjmore_categories - tops_categories)
        tops_unique = len(tops_categories - cjmore_categories)
        
        coverage_data = {
            'Shared Categories': overlap,
            'CJMore Unique': cjmore_unique,
            'Tops Daily Unique': tops_unique
        }
        
        bars = ax4.bar(range(len(coverage_data)), list(coverage_data.values()),
                      color=[self.colors['accent2'], self.colors['cjmore'], self.colors['tops']])
        ax4.set_xticks(range(len(coverage_data)))
        ax4.set_xticklabels(list(coverage_data.keys()))
        ax4.set_ylabel('Number of Categories')
        ax4.set_title('Category Coverage Analysis', fontweight='bold', pad=20)
        ax4.grid(axis='y', alpha=0.3, color=self.colors['light'])
        
        # Add value labels
        for bar, (name, value) in zip(bars, coverage_data.items()):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(coverage_data.values()) * 0.02,
                    str(value), ha='center', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        
        # Save comparison
        report_path = self.output_dir / 'market_overview_comparison.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {report_path.name}")
    
    def create_competitive_positioning_analysis(self):
        """Analyze competitive positioning strategies"""
        
        print("\n🎯 Creating Competitive Positioning Analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CJMore vs Tops Daily: Competitive Positioning Analysis', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Market Positioning Map (Products vs Brands)
        positioning_data = {
            'CJMore': {
                'products': len(self.df_cjmore),
                'brands': self.df_cjmore['brand'].nunique(),
                'categories': self.df_cjmore['category'].nunique()
            },
            'Tops Daily': {
                'products': len(self.df_tops),
                'brands': self.df_tops['brand'].nunique(), 
                'categories': self.df_tops['category'].nunique()
            }
        }
        
        # Calculate products per brand (efficiency metric)
        cjmore_ppb = positioning_data['CJMore']['products'] / positioning_data['CJMore']['brands']
        tops_ppb = positioning_data['Tops Daily']['products'] / positioning_data['Tops Daily']['brands']
        
        ax1.scatter(positioning_data['CJMore']['brands'], positioning_data['CJMore']['products'],
                   s=300, color=self.colors['cjmore'], label='CJMore', alpha=0.8, edgecolors='white', linewidth=2)
        ax1.scatter(positioning_data['Tops Daily']['brands'], positioning_data['Tops Daily']['products'],
                   s=300, color=self.colors['tops'], label='Tops Daily', alpha=0.8, edgecolors='white', linewidth=2)
        
        # Add annotations
        ax1.annotate('CJMore\n(Premium Portfolio)', 
                    (positioning_data['CJMore']['brands'], positioning_data['CJMore']['products']),
                    xytext=(20, 20), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['cjmore'], alpha=0.7, color='white'),
                    fontweight='bold', color='white')
        
        ax1.annotate('Tops Daily\n(Value Focus)', 
                    (positioning_data['Tops Daily']['brands'], positioning_data['Tops Daily']['products']),
                    xytext=(-80, -30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor=self.colors['tops'], alpha=0.7, color='white'),
                    fontweight='bold', color='white')
        
        ax1.set_xlabel('Number of Unique Brands')
        ax1.set_ylabel('Total Products')
        ax1.set_title('Market Positioning Map', fontweight='bold', pad=20)
        ax1.grid(alpha=0.3, color=self.colors['light'])
        ax1.legend()
        
        # 2. Category Strategy Comparison
        # Top categories for each supermarket
        cjmore_top_cats = self.df_cjmore['category'].value_counts().head(5)
        tops_top_cats = self.df_tops['category'].value_counts().head(5)
        
        # Combined top categories
        all_top_cats = pd.concat([cjmore_top_cats, tops_top_cats]).groupby(level=0).sum().sort_values(ascending=False).head(8)
        
        # Get values for each supermarket for these categories
        cjmore_values = []
        tops_values = []
        
        for cat in all_top_cats.index:
            cjmore_val = len(self.df_cjmore[self.df_cjmore['category'] == cat])
            tops_val = len(self.df_tops[self.df_tops['category'] == cat])
            cjmore_values.append(cjmore_val)
            tops_values.append(tops_val)
        
        x_pos = np.arange(len(all_top_cats))
        width = 0.35
        
        bars1 = ax2.bar(x_pos - width/2, cjmore_values, width, label='CJMore', color=self.colors['cjmore'])
        bars2 = ax2.bar(x_pos + width/2, tops_values, width, label='Tops Daily', color=self.colors['tops'])
        
        ax2.set_xlabel('Categories')
        ax2.set_ylabel('Number of Products')
        ax2.set_title('Top Categories Strategy Comparison', fontweight='bold', pad=20)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([cat[:15] + '...' if len(str(cat)) > 15 else str(cat) 
                            for cat in all_top_cats.index], rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3, color=self.colors['light'])
        
        # 3. Brand Concentration Analysis
        # Calculate HHI (Herfindahl-Hirschman Index) for brand concentration
        def calculate_hhi(df):
            brand_counts = df['brand'].value_counts()
            market_shares = brand_counts / len(df)
            return (market_shares ** 2).sum()
        
        cjmore_hhi = calculate_hhi(self.df_cjmore)
        tops_hhi = calculate_hhi(self.df_tops)
        
        concentration_data = {
            'CJMore': cjmore_hhi,
            'Tops Daily': tops_hhi
        }
        
        bars = ax3.bar(range(len(concentration_data)), list(concentration_data.values()),
                      color=[self.colors['cjmore'], self.colors['tops']])
        ax3.set_xticks(range(len(concentration_data)))
        ax3.set_xticklabels(list(concentration_data.keys()))
        ax3.set_ylabel('Brand Concentration Index (HHI)')
        ax3.set_title('Brand Concentration Comparison', fontweight='bold', pad=20)
        ax3.grid(axis='y', alpha=0.3, color=self.colors['light'])
        
        # Add interpretation text
        ax3.text(0.5, 0.8, 'Higher HHI = More Concentrated\nLower HHI = More Diverse', 
                ha='center', va='center', transform=ax3.transAxes,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # Add value labels
        for bar, (name, value) in zip(bars, concentration_data.items()):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(concentration_data.values()) * 0.02,
                    f'{value:.4f}', ha='center', fontweight='bold', fontsize=10)
        
        # 4. Market Efficiency Analysis
        efficiency_metrics = {
            'Products per Brand': [cjmore_ppb, tops_ppb],
            'Categories per 100 Products': [
                (positioning_data['CJMore']['categories'] / positioning_data['CJMore']['products']) * 100,
                (positioning_data['Tops Daily']['categories'] / positioning_data['Tops Daily']['products']) * 100
            ]
        }
        
        x_pos = np.arange(len(efficiency_metrics))
        width = 0.35
        
        bars1 = ax4.bar(x_pos - width/2, [efficiency_metrics['Products per Brand'][0], 
                                         efficiency_metrics['Categories per 100 Products'][0]], 
                       width, label='CJMore', color=self.colors['cjmore'])
        bars2 = ax4.bar(x_pos + width/2, [efficiency_metrics['Products per Brand'][1],
                                         efficiency_metrics['Categories per 100 Products'][1]], 
                       width, label='Tops Daily', color=self.colors['tops'])
        
        ax4.set_xlabel('Efficiency Metrics')
        ax4.set_ylabel('Metric Value')
        ax4.set_title('Market Efficiency Comparison', fontweight='bold', pad=20)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(['Products\nper Brand', 'Categories\nper 100 Products'])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3, color=self.colors['light'])
        
        plt.tight_layout()
        
        # Save analysis
        report_path = self.output_dir / 'competitive_positioning_analysis.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {report_path.name}")
    
    def create_brand_strategy_comparison(self):
        """Compare brand strategies between supermarkets"""
        
        print("\n🏷️ Creating Brand Strategy Comparison...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CJMore vs Tops Daily: Brand Strategy Comparison', fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Top Brand Performance
        cjmore_top_brands = self.df_cjmore['brand'].value_counts().head(8)
        tops_top_brands = self.df_tops['brand'].value_counts().head(8)
        
        # Plot CJMore top brands
        bars1 = ax1.barh(range(len(cjmore_top_brands)), cjmore_top_brands.values,
                        color=self.colors['cjmore'])
        ax1.set_yticks(range(len(cjmore_top_brands)))
        ax1.set_yticklabels([brand[:20] + '...' if len(str(brand)) > 20 else str(brand) 
                            for brand in cjmore_top_brands.index], fontsize=9)
        ax1.set_xlabel('Number of Products')
        ax1.set_title('CJMore: Top Performing Brands', fontweight='bold', pad=20)
        ax1.grid(axis='x', alpha=0.3, color=self.colors['light'])
        
        # Plot Tops Daily top brands  
        bars2 = ax2.barh(range(len(tops_top_brands)), tops_top_brands.values,
                        color=self.colors['tops'])
        ax2.set_yticks(range(len(tops_top_brands)))
        ax2.set_yticklabels([brand[:20] + '...' if len(str(brand)) > 20 else str(brand) 
                            for brand in tops_top_brands.index], fontsize=9)
        ax2.set_xlabel('Number of Products')
        ax2.set_title('Tops Daily: Top Performing Brands', fontweight='bold', pad=20)
        ax2.grid(axis='x', alpha=0.3, color=self.colors['light'])
        
        # 3. Brand Overlap Analysis
        cjmore_brands = set(self.df_cjmore['brand'].unique())
        tops_brands = set(self.df_tops['brand'].unique())
        
        overlap_brands = cjmore_brands.intersection(tops_brands)
        cjmore_unique = cjmore_brands - tops_brands
        tops_unique = tops_brands - cjmore_brands
        
        overlap_data = {
            'Shared Brands': len(overlap_brands),
            'CJMore Exclusive': len(cjmore_unique),
            'Tops Daily Exclusive': len(tops_unique)
        }
        
        wedges, texts, autotexts = ax3.pie(overlap_data.values(), labels=overlap_data.keys(),
                                          autopct='%1.1f%%', colors=[self.colors['accent2'], 
                                                                     self.colors['cjmore'], 
                                                                     self.colors['tops']],
                                          startangle=90)
        ax3.set_title('Brand Portfolio Overlap', fontweight='bold', pad=20)
        
        for autotext in autotexts:
            autotext.set_fontweight('bold')
            autotext.set_color('white')
        
        # 4. Market Share Distribution
        # Analyze top 5 shared brands
        if len(overlap_brands) > 0:
            shared_brand_performance = {}
            
            for brand in list(overlap_brands)[:5]:  # Top 5 shared brands
                cjmore_count = len(self.df_cjmore[self.df_cjmore['brand'] == brand])
                tops_count = len(self.df_tops[self.df_tops['brand'] == brand])
                shared_brand_performance[brand] = [cjmore_count, tops_count]
            
            if shared_brand_performance:
                brands = list(shared_brand_performance.keys())
                cjmore_values = [shared_brand_performance[brand][0] for brand in brands]
                tops_values = [shared_brand_performance[brand][1] for brand in brands]
                
                x_pos = np.arange(len(brands))
                width = 0.35
                
                bars1 = ax4.bar(x_pos - width/2, cjmore_values, width, label='CJMore', color=self.colors['cjmore'])
                bars2 = ax4.bar(x_pos + width/2, tops_values, width, label='Tops Daily', color=self.colors['tops'])
                
                ax4.set_xlabel('Shared Brands')
                ax4.set_ylabel('Number of Products')
                ax4.set_title('Shared Brand Performance', fontweight='bold', pad=20)
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels([brand[:12] + '...' if len(str(brand)) > 12 else str(brand) 
                                    for brand in brands], rotation=45, ha='right')
                ax4.legend()
                ax4.grid(axis='y', alpha=0.3, color=self.colors['light'])
            else:
                ax4.text(0.5, 0.5, 'No Shared Brands\nDetected', ha='center', va='center',
                        fontsize=12, transform=ax4.transAxes, color=self.colors['dark'])
                ax4.set_title('Shared Brand Analysis', fontweight='bold', pad=20)
        else:
            ax4.text(0.5, 0.5, 'No Shared Brands\nBetween Supermarkets', ha='center', va='center',
                    fontsize=12, transform=ax4.transAxes, color=self.colors['dark'])
            ax4.set_title('Shared Brand Analysis', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save comparison
        report_path = self.output_dir / 'brand_strategy_comparison.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"   ✅ Saved: {report_path.name}")
    
    def create_strategic_insights_report(self):
        """Create comprehensive strategic insights and recommendations"""
        
        print("\n📊 Creating Strategic Insights Report...")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Calculate key comparative metrics
        cjmore_stats = {
            'products': len(self.df_cjmore),
            'brands': self.df_cjmore['brand'].nunique(),
            'categories': self.df_cjmore['category'].nunique(),
            'thai_pct': (len(self.df_cjmore[self.df_cjmore['origin'] == 'thai']) / len(self.df_cjmore)) * 100,
            'intl_pct': (len(self.df_cjmore[self.df_cjmore['origin'] == 'international']) / len(self.df_cjmore)) * 100
        }
        
        tops_stats = {
            'products': len(self.df_tops),
            'brands': self.df_tops['brand'].nunique(),
            'categories': self.df_tops['category'].nunique(),
            'thai_pct': (len(self.df_tops[self.df_tops['origin'] == 'thai']) / len(self.df_tops)) * 100,
            'intl_pct': (len(self.df_tops[self.df_tops['origin'] == 'international']) / len(self.df_tops)) * 100
        }
        
        # Brand overlap analysis
        cjmore_brands = set(self.df_cjmore['brand'].unique())
        tops_brands = set(self.df_tops['brand'].unique())
        overlap_brands = cjmore_brands.intersection(tops_brands)
        
        # Category overlap analysis
        cjmore_categories = set(self.df_cjmore['category'].unique())
        tops_categories = set(self.df_tops['category'].unique())
        overlap_categories = cjmore_categories.intersection(tops_categories)
        
        strategic_report = f"""
# CJMore vs Tops Daily: Strategic Market Analysis

**Analysis Date:** {timestamp}  
**Scope:** Comprehensive Competitive Intelligence  
**Methodology:** Enhanced ML + AI Categorization Analysis  

## 🎯 Executive Summary

### Market Positioning Overview
- **CJMore Strategy:** Premium lifestyle supermarket with comprehensive product portfolio
- **Tops Daily Strategy:** Value-focused international brands with efficient operations
- **Market Differentiation:** Clear positioning in different customer segments

## 📊 Comparative Market Metrics

### Portfolio Scale Analysis
| Metric | CJMore | Tops Daily | Advantage |
|--------|--------|------------|-----------|
| **Total Products** | {cjmore_stats['products']:,} | {tops_stats['products']:,} | {'CJMore' if cjmore_stats['products'] > tops_stats['products'] else 'Tops Daily'} |
| **Unique Brands** | {cjmore_stats['brands']:,} | {tops_stats['brands']:,} | {'CJMore' if cjmore_stats['brands'] > tops_stats['brands'] else 'Tops Daily'} |
| **Categories** | {cjmore_stats['categories']} | {tops_stats['categories']} | {'CJMore' if cjmore_stats['categories'] > tops_stats['categories'] else 'Tops Daily'} |
| **Products per Brand** | {cjmore_stats['products']/cjmore_stats['brands']:.1f} | {tops_stats['products']/tops_stats['brands']:.1f} | {'CJMore' if (cjmore_stats['products']/cjmore_stats['brands']) > (tops_stats['products']/tops_stats['brands']) else 'Tops Daily'} |

### Market Origin Strategy
| Origin Focus | CJMore | Tops Daily | Strategic Insight |
|--------------|--------|------------|-------------------|
| **Thai Products** | {cjmore_stats['thai_pct']:.1f}% | {tops_stats['thai_pct']:.1f}% | {'CJMore more Thai-focused' if cjmore_stats['thai_pct'] > tops_stats['thai_pct'] else 'Tops Daily more Thai-focused'} |
| **International Products** | {cjmore_stats['intl_pct']:.1f}% | {tops_stats['intl_pct']:.1f}% | {'CJMore more international' if cjmore_stats['intl_pct'] > tops_stats['intl_pct'] else 'Tops Daily more international'} |

### Brand & Category Overlap
- **Shared Brands:** {len(overlap_brands):,} ({len(overlap_brands)/min(len(cjmore_brands), len(tops_brands))*100:.1f}% overlap)
- **Shared Categories:** {len(overlap_categories)} ({len(overlap_categories)/min(len(cjmore_categories), len(tops_categories))*100:.1f}% overlap)
- **Market Differentiation Level:** {'High' if len(overlap_brands)/min(len(cjmore_brands), len(tops_brands)) < 0.3 else 'Moderate' if len(overlap_brands)/min(len(cjmore_brands), len(tops_brands)) < 0.6 else 'Low'}

## 🏆 Competitive Advantages Analysis

### CJMore Competitive Strengths
1. **Premium Market Leadership**
   - Comprehensive product portfolio ({cjmore_stats['products']:,} products)
   - Diverse brand ecosystem ({cjmore_stats['brands']:,} unique brands)
   - {"Extensive" if cjmore_stats['categories'] >= 10 else "Focused"} category coverage

2. **Lifestyle Positioning**
   - {"Premium international focus" if cjmore_stats['intl_pct'] > 60 else "Balanced origin strategy"}
   - High-variety product selection
   - Brand diversity advantage

3. **Market Coverage**
   - {"Superior" if cjmore_stats['products'] > tops_stats['products'] else "Comparable"} product breadth
   - {"Advanced" if cjmore_stats['categories'] > tops_stats['categories'] else "Efficient"} category strategy

### Tops Daily Competitive Strengths
1. **Value Market Leadership**
   - Efficient brand portfolio ({tops_stats['products']/tops_stats['brands']:.1f} products per brand)
   - {"International-focused" if tops_stats['intl_pct'] > 50 else "Thai-focused"} value strategy
   - Streamlined operations

2. **Operational Efficiency**
   - {"Higher" if (tops_stats['products']/tops_stats['brands']) > (cjmore_stats['products']/cjmore_stats['brands']) else "Comparable"} brand efficiency
   - {"Focused" if tops_stats['categories'] < 10 else "Comprehensive"} category approach
   - Cost-effective portfolio management

3. **Market Accessibility**
   - Value-oriented positioning
   - {"Strong" if tops_stats['intl_pct'] > 45 else "Moderate"} international brand access
   - Everyday essentials focus

## 🎯 Strategic Market Opportunities

### For CJMore
1. **Market Expansion Opportunities**
   - Leverage premium positioning for category leadership
   - Expand {"international" if cjmore_stats['intl_pct'] < 50 else "Thai"} product offerings
   - Enhance private label penetration

2. **Competitive Differentiation**
   - Strengthen lifestyle and premium categories
   - Develop exclusive brand partnerships
   - Advanced merchandising and customer experience

### For Tops Daily  
1. **Value Market Domination**
   - Strengthen everyday essentials positioning
   - Expand private brand portfolio (My Choice, Smart-r, Love The Value)
   - Enhance international value brands

2. **Operational Excellence**
   - Leverage efficient brand management model
   - Optimize high-volume category performance
   - Develop cost leadership advantages

## 📈 Market Implications & Forecasts

### Competitive Dynamics
- **Market Segmentation:** Clear differentiation enables coexistence
- **Growth Vectors:** Different customer segments and value propositions  
- **Innovation Focus:** CJMore on premium experience, Tops Daily on value optimization

### Strategic Recommendations

#### Immediate Actions (0-3 months)
1. **CJMore:** Strengthen premium category leadership and lifestyle positioning
2. **Tops Daily:** Enhance value proposition and operational efficiency
3. **Both:** Leverage enhanced analytics for competitive intelligence

#### Medium-term Strategy (3-12 months)
1. **Market Position Reinforcement:** Each supermarket should double down on core strengths
2. **Technology Adoption:** Use ML/AI insights for competitive advantage
3. **Customer Segmentation:** Develop targeted strategies for distinct market segments

#### Long-term Vision (1+ years)
1. **Market Leadership:** Establish category dominance in respective segments
2. **Innovation Integration:** Leverage data analytics for continuous competitive advantage
3. **Ecosystem Development:** Build comprehensive retail ecosystems aligned with positioning

## 💡 Key Success Factors

### Critical Success Elements
1. **Clear Market Positioning:** Maintain distinct value propositions
2. **Operational Excellence:** Optimize for target customer segments  
3. **Data-Driven Decision Making:** Leverage enhanced analytics capabilities
4. **Category Management:** Focus on high-performing product areas
5. **Brand Strategy Optimization:** Align portfolio with market positioning

### Competitive Monitoring KPIs
- Portfolio efficiency ratios
- Category market share trends  
- Brand performance metrics
- Customer segment penetration
- Operational cost advantages

---

**Conclusion:** CJMore and Tops Daily represent complementary market strategies with clear differentiation. Success depends on strengthening core positioning while leveraging advanced analytics for competitive advantage.

*Analysis powered by Enhanced ML + AI Categorization System with Computer Vision and Gemini Classification.*
"""
        
        # Save strategic report
        report_path = self.output_dir / 'strategic_insights_report.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(strategic_report.strip())
        
        print(f"   ✅ Saved: {report_path.name}")
    
    def run_competitive_analysis(self):
        """Execute complete competitive analysis"""
        
        print("🚀 Starting Comprehensive Competitive Market Analysis")
        print("=" * 70)
        
        # Create all comparative analyses
        self.create_market_overview_comparison()
        self.create_competitive_positioning_analysis()
        self.create_brand_strategy_comparison()
        self.create_strategic_insights_report()
        
        print("\n" + "=" * 70)
        print("✅ Comprehensive Competitive Analysis Complete!")
        print(f"📁 Reports Location: {self.output_dir}")
        print("📊 Generated Analysis:")
        print("   - Market Overview Comparison")
        print("   - Competitive Positioning Analysis")
        print("   - Brand Strategy Comparison") 
        print("   - Strategic Insights Report")
        print("🎯 Ready for strategic competitive intelligence!")
        print("=" * 70)

def main():
    """Main execution function"""
    
    # Create competitive analyzer and run analysis
    analyzer = SupermarketCompetitiveAnalysis()
    analyzer.run_competitive_analysis()

if __name__ == "__main__":
    main()