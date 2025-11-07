#!/usr/bin/env python3
"""
Comprehensive Visual Comparison: CJMore vs Tops Daily
===================================================

Creates professional visual charts comparing product portfolios, variety,
and categories between the two supermarkets. All visualizations designed
for business presentation with clear comparative insights.

Color Scheme:
- Accent/Basic: #BFDAEF (Light Blue)
- CJMore: #96B991 (Green)  
- Tops Daily: #EF865B (Orange)
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify  # For treemap
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

class SupermarketVisualComparison:
    """Professional visual comparison system for supermarket analysis"""
    
    def __init__(self):
        """Initialize visualization system with color scheme and data paths"""
        
        # Professional color scheme
        self.colors = {
            'accent': '#BFDAEF',      # Light Blue (Basic/Accent)
            'cjmore': '#96B991',      # Green (CJMore)
            'tops_daily': '#EF865B',  # Orange (Tops Daily)
            'light_accent': '#E8F2F8', # Lighter blue for backgrounds
            'dark_accent': '#7FB3D3',  # Darker blue for emphasis
            'neutral': '#F5F5F5'      # Light gray for neutrals
        }
        
        # Data paths
        self.cjmore_file = Path('final_results/cjmore_data/CJMore_Complete_Analysis.xlsx')
        self.tops_file = Path('final_results/tops_daily_data/tops_daily_brand_classification_enhanced.csv')
        self.output_dir = Path('final_results/comprehensive_analysis/visual_comparisons')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load datasets
        self.load_data()
        
        # Set style
        plt.style.use('default')
        sns.set_palette([self.colors['cjmore'], self.colors['tops_daily'], self.colors['accent']])
        
        print("🎨 Visual Comparison System Initialized")
        print(f"📊 CJMore: {len(self.df_cjmore):,} products")
        print(f"📊 Tops Daily: {len(self.df_tops):,} products")
    
    def load_data(self):
        """Load and prepare datasets for visualization"""
        
        print("📂 Loading Supermarket Data for Visualization...")
        
        # Load CJMore data
        if self.cjmore_file.exists():
            self.df_cjmore = pd.read_excel(self.cjmore_file, sheet_name='Brand Classification')
            print(f"   ✅ CJMore: {len(self.df_cjmore):,} products loaded")
        else:
            raise FileNotFoundError(f"CJMore file not found: {self.cjmore_file}")
        
        # Load Tops Daily data
        if self.tops_file.exists():
            self.df_tops = pd.read_csv(self.tops_file)
            print(f"   ✅ Tops Daily: {len(self.df_tops):,} products loaded")
        else:
            raise FileNotFoundError(f"Tops Daily file not found: {self.tops_file}")
        
        # Add supermarket identifiers
        self.df_cjmore['supermarket'] = 'CJMore'
        self.df_tops['supermarket'] = 'Tops Daily'
        
        # Standardize column names
        for col in ['brand', 'product_type', 'category', 'origin']:
            if col not in self.df_cjmore.columns:
                self.df_cjmore[col] = 'Unknown'
            if col not in self.df_tops.columns:
                self.df_tops[col] = 'Unknown'
        
        # Fill missing values
        self.df_cjmore = self.df_cjmore.fillna('Unknown')
        self.df_tops = self.df_tops.fillna('Unknown')
    
    def create_product_portfolio_comparison(self):
        """Bar Chart: Product Portfolio & Variety Comparison"""
        
        print("\n📊 Creating Product Portfolio Comparison...")
        
        # Get category counts for both supermarkets
        cjmore_categories = self.df_cjmore['category'].value_counts()
        tops_categories = self.df_tops['category'].value_counts()
        
        # Get all unique categories
        all_categories = list(set(cjmore_categories.index) | set(tops_categories.index))
        
        # Prepare data for comparison
        comparison_data = []
        for category in all_categories:
            cjmore_count = cjmore_categories.get(category, 0)
            tops_count = tops_categories.get(category, 0)
            
            comparison_data.append({
                'Category': category,
                'CJMore': cjmore_count,
                'Tops Daily': tops_count
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        df_comparison = df_comparison.sort_values('CJMore', ascending=False).head(10)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(df_comparison))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_comparison['CJMore'], width, 
                      label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars2 = ax.bar(x + width/2, df_comparison['Tops Daily'], width,
                      label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        # Customize chart
        ax.set_xlabel('Product Categories', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Products', fontsize=12, fontweight='bold')
        ax.set_title('Product Portfolio Comparison: CJMore vs Tops Daily\nNumber of Products by Category', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([cat[:15] + '...' if len(cat) > 15 else cat 
                           for cat in df_comparison['Category']], rotation=45, ha='right')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                           f'{int(height):,}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=9)
        
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, color=self.colors['accent'])
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '1_product_portfolio_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("   ✅ Product Portfolio Comparison saved")
    
    def create_category_distribution_pie_charts(self):
        """Pie Charts: Category Distribution for Both Supermarkets"""
        
        print("\n🥧 Creating Category Distribution Pie Charts...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # CJMore distribution
        cjmore_categories = self.df_cjmore['category'].value_counts().head(8)
        
        wedges1, texts1, autotexts1 = ax1.pie(cjmore_categories.values, 
                                             labels=[cat[:12] + '...' if len(cat) > 12 else cat 
                                                    for cat in cjmore_categories.index],
                                             autopct='%1.1f%%', startangle=90,
                                             colors=plt.cm.Set3(np.linspace(0, 1, len(cjmore_categories))))
        
        ax1.set_title('CJMore Category Distribution\n3,694 Total Products', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Tops Daily distribution
        tops_categories = self.df_tops['category'].value_counts().head(8)
        
        wedges2, texts2, autotexts2 = ax2.pie(tops_categories.values,
                                             labels=[cat[:12] + '...' if len(cat) > 12 else cat 
                                                    for cat in tops_categories.index],
                                             autopct='%1.1f%%', startangle=90,
                                             colors=plt.cm.Set3(np.linspace(0, 1, len(tops_categories))))
        
        ax2.set_title('Tops Daily Category Distribution\n1,961 Total Products', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Style improvements
        for autotexts in [autotexts1, autotexts2]:
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_fontsize(9)
                autotext.set_color('white')
        
        plt.suptitle('Category Distribution Comparison: CJMore vs Tops Daily', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '2_category_distribution_pie_charts.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("   ✅ Category Distribution Pie Charts saved")
    
    def create_private_label_comparison(self):
        """Bar Chart: Private Label Comparison"""
        
        print("\n🏷️ Creating Private Label Comparison...")
        
        # Define private label brands
        cjmore_private_brands = ['UNO', 'NINE BEAUTY', 'uno', 'nine beauty']
        tops_private_brands = ['My Choice', 'Tops', 'Smart-r', 'Love The Value', 'my choice']
        
        # Calculate private label percentages
        cjmore_private_count = self.df_cjmore[
            self.df_cjmore['brand'].str.contains('|'.join(cjmore_private_brands), case=False, na=False)
        ].shape[0]
        
        tops_private_count = self.df_tops[
            self.df_tops['brand'].str.contains('|'.join(tops_private_brands), case=False, na=False)
        ].shape[0]
        
        cjmore_private_pct = (cjmore_private_count / len(self.df_cjmore)) * 100
        tops_private_pct = (tops_private_count / len(self.df_tops)) * 100
        
        # Create comparison data
        private_label_data = {
            'Supermarket': ['CJMore', 'Tops Daily'],
            'Private Label %': [cjmore_private_pct, tops_private_pct],
            'National Brands %': [100 - cjmore_private_pct, 100 - tops_private_pct]
        }
        
        df_private = pd.DataFrame(private_label_data)
        
        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df_private))
        width = 0.6
        
        bars1 = ax.bar(x, df_private['Private Label %'], width,
                      label='Private Label', color=self.colors['accent'], alpha=0.8)
        bars2 = ax.bar(x, df_private['National Brands %'], width,
                      bottom=df_private['Private Label %'],
                      label='National Brands', color=self.colors['dark_accent'], alpha=0.8)
        
        # Customize chart
        ax.set_xlabel('Supermarket', fontsize=12, fontweight='bold')
        ax.set_ylabel('Percentage of Product Portfolio (%)', fontsize=12, fontweight='bold')
        ax.set_title('Private Label vs National Brands Comparison\nProduct Portfolio Distribution', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(df_private['Supermarket'])
        ax.set_ylim(0, 100)
        
        # Add percentage labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            private_pct = df_private.iloc[i]['Private Label %']
            national_pct = df_private.iloc[i]['National Brands %']
            
            # Private label percentage
            ax.text(bar1.get_x() + bar1.get_width()/2., private_pct/2,
                   f'{private_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=11, color='black')
            
            # National brands percentage
            ax.text(bar2.get_x() + bar2.get_width()/2., private_pct + national_pct/2,
                   f'{national_pct:.1f}%', ha='center', va='center', 
                   fontweight='bold', fontsize=11, color='white')
        
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, color=self.colors['accent'])
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '3_private_label_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("   ✅ Private Label Comparison saved")
    
    def create_treemap_visualization(self):
        """Treemap: Product Distribution Visualization"""
        
        print("\n🗺️ Creating Treemap Visualizations...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # CJMore treemap
        cjmore_categories = self.df_cjmore['category'].value_counts().head(8)
        
        squarify.plot(sizes=cjmore_categories.values,
                     label=[f'{cat}\n{count:,} products' for cat, count in cjmore_categories.items()],
                     alpha=0.8, ax=ax1, color=plt.cm.Set3(np.linspace(0, 1, len(cjmore_categories))))
        
        ax1.set_title('CJMore Product Distribution\nTreemap by Category', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.axis('off')
        
        # Tops Daily treemap
        tops_categories = self.df_tops['category'].value_counts().head(8)
        
        squarify.plot(sizes=tops_categories.values,
                     label=[f'{cat}\n{count:,} products' for cat, count in tops_categories.items()],
                     alpha=0.8, ax=ax2, color=plt.cm.Set3(np.linspace(0, 1, len(tops_categories))))
        
        ax2.set_title('Tops Daily Product Distribution\nTreemap by Category', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.axis('off')
        
        plt.suptitle('Product Distribution Treemap: Category Proportions Visualization', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '4_product_distribution_treemap.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("   ✅ Treemap Visualizations saved")
    
    def create_side_by_side_category_comparison(self):
        """Side-by-Side Bar Chart: Specific Category Comparison"""
        
        print("\n📊 Creating Side-by-Side Category Comparison...")
        
        # Focus on key categories for comparison
        key_categories = ['Food & Snacks', 'Beverages', 'Personal Care', 'Household & Cleaning', 
                         'Health & Pharmacy', 'Baby & Kids']
        
        # Prepare comparison data
        comparison_data = []
        for category in key_categories:
            cjmore_count = len(self.df_cjmore[self.df_cjmore['category'] == category])
            tops_count = len(self.df_tops[self.df_tops['category'] == category])
            
            # Calculate percentages
            cjmore_pct = (cjmore_count / len(self.df_cjmore)) * 100
            tops_pct = (tops_count / len(self.df_tops)) * 100
            
            comparison_data.append({
                'Category': category,
                'CJMore Count': cjmore_count,
                'Tops Daily Count': tops_count,
                'CJMore %': cjmore_pct,
                'Tops Daily %': tops_pct
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create side-by-side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        # Product count comparison
        x = np.arange(len(df_comparison))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, df_comparison['CJMore Count'], width,
                       label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, df_comparison['Tops Daily Count'], width,
                       label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax1.set_xlabel('Categories', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('Product Count by Category', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([cat.replace(' & ', '\n& ') for cat in df_comparison['Category']], 
                           rotation=45, ha='right', fontsize=9)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
                           f'{int(height)}', ha='center', va='bottom', 
                           fontweight='bold', fontsize=8)
        
        # Percentage comparison
        bars3 = ax2.bar(x - width/2, df_comparison['CJMore %'], width,
                       label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars4 = ax2.bar(x + width/2, df_comparison['Tops Daily %'], width,
                       label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax2.set_xlabel('Categories', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Percentage of Portfolio (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Portfolio Share by Category', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([cat.replace(' & ', '\n& ') for cat in df_comparison['Category']], 
                           rotation=45, ha='right', fontsize=9)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for bars in [bars3, bars4]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                           f'{height:.1f}%', ha='center', va='bottom', 
                           fontweight='bold', fontsize=8)
        
        plt.suptitle('Category Comparison: CJMore vs Tops Daily\nProduct Count & Portfolio Share', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '5_side_by_side_category_comparison.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("   ✅ Side-by-Side Category Comparison saved")
    
    def create_top_categories_ranked_chart(self):
        """Ranked Bar Chart: Top Categories by Product Count"""
        
        print("\n🏆 Creating Top Categories Ranked Chart...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # CJMore top categories
        cjmore_top = self.df_cjmore['category'].value_counts().head(8)
        
        bars1 = ax1.barh(range(len(cjmore_top)), cjmore_top.values, 
                        color=self.colors['cjmore'], alpha=0.8)
        
        ax1.set_yticks(range(len(cjmore_top)))
        ax1.set_yticklabels([f"{i+1}. {cat}" for i, cat in enumerate(cjmore_top.index)])
        ax1.set_xlabel('Number of Products', fontsize=11, fontweight='bold')
        ax1.set_title('CJMore: Top Categories by Product Count', fontsize=12, fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3, color=self.colors['accent'])
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, cjmore_top.values)):
            ax1.text(value + 10, bar.get_y() + bar.get_height()/2,
                    f'{value:,} ({value/len(self.df_cjmore)*100:.1f}%)', 
                    va='center', fontweight='bold', fontsize=9)
        
        # Tops Daily top categories
        tops_top = self.df_tops['category'].value_counts().head(8)
        
        bars2 = ax2.barh(range(len(tops_top)), tops_top.values, 
                        color=self.colors['tops_daily'], alpha=0.8)
        
        ax2.set_yticks(range(len(tops_top)))
        ax2.set_yticklabels([f"{i+1}. {cat}" for i, cat in enumerate(tops_top.index)])
        ax2.set_xlabel('Number of Products', fontsize=11, fontweight='bold')
        ax2.set_title('Tops Daily: Top Categories by Product Count', fontsize=12, fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, color=self.colors['accent'])
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, tops_top.values)):
            ax2.text(value + 5, bar.get_y() + bar.get_height()/2,
                    f'{value:,} ({value/len(self.df_tops)*100:.1f}%)', 
                    va='center', fontweight='bold', fontsize=9)
        
        plt.suptitle('Top Categories Ranking: Product Count & Portfolio Share', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / '6_top_categories_ranked_chart.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("   ✅ Top Categories Ranked Chart saved")
    
    def create_summary_dashboard(self):
        """Create comprehensive summary dashboard"""
        
        print("\n📋 Creating Summary Dashboard...")
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Key metrics comparison
        ax1 = fig.add_subplot(gs[0, :])
        
        metrics = ['Total Products', 'Unique Brands', 'Categories', 'Product Types']
        cjmore_values = [
            len(self.df_cjmore),
            self.df_cjmore['brand'].nunique(),
            self.df_cjmore['category'].nunique(),
            self.df_cjmore['product_type'].nunique()
        ]
        tops_values = [
            len(self.df_tops),
            self.df_tops['brand'].nunique(), 
            self.df_tops['category'].nunique(),
            self.df_tops['product_type'].nunique()
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, cjmore_values, width, 
                       label='CJMore', color=self.colors['cjmore'], alpha=0.8)
        bars2 = ax1.bar(x + width/2, tops_values, width,
                       label='Tops Daily', color=self.colors['tops_daily'], alpha=0.8)
        
        ax1.set_xlabel('Key Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax1.set_title('Portfolio Overview: Key Performance Indicators', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend(fontsize=11)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(max(cjmore_values), max(tops_values)) * 0.01,
                        f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Category distribution (simplified)
        ax2 = fig.add_subplot(gs[1, 0])
        cjmore_top3 = self.df_cjmore['category'].value_counts().head(3)
        ax2.pie(cjmore_top3.values, labels=cjmore_top3.index, autopct='%1.1f%%',
               colors=[self.colors['cjmore'], self.colors['accent'], self.colors['light_accent']])
        ax2.set_title('CJMore\nTop 3 Categories', fontweight='bold')
        
        ax3 = fig.add_subplot(gs[1, 1])
        tops_top3 = self.df_tops['category'].value_counts().head(3)
        ax3.pie(tops_top3.values, labels=tops_top3.index, autopct='%1.1f%%',
               colors=[self.colors['tops_daily'], self.colors['accent'], self.colors['light_accent']])
        ax3.set_title('Tops Daily\nTop 3 Categories', fontweight='bold')
        
        # Brand efficiency comparison
        ax4 = fig.add_subplot(gs[1, 2])
        efficiency_metrics = ['Products\nper Brand', 'Brands\nper Category']
        cjmore_eff = [
            len(self.df_cjmore) / self.df_cjmore['brand'].nunique(),
            self.df_cjmore['brand'].nunique() / self.df_cjmore['category'].nunique()
        ]
        tops_eff = [
            len(self.df_tops) / self.df_tops['brand'].nunique(),
            self.df_tops['brand'].nunique() / self.df_tops['category'].nunique()
        ]
        
        x_eff = np.arange(len(efficiency_metrics))
        ax4.bar(x_eff - 0.2, cjmore_eff, 0.4, label='CJMore', color=self.colors['cjmore'])
        ax4.bar(x_eff + 0.2, tops_eff, 0.4, label='Tops Daily', color=self.colors['tops_daily'])
        ax4.set_xticks(x_eff)
        ax4.set_xticklabels(efficiency_metrics)
        ax4.set_title('Efficiency Metrics', fontweight='bold')
        ax4.legend()
        
        # Market positioning summary
        ax5 = fig.add_subplot(gs[2, :])
        ax5.text(0.5, 0.8, 'Market Positioning Summary', ha='center', fontsize=16, fontweight='bold', 
                transform=ax5.transAxes)
        
        summary_text = f"""
CJMore Strategic Position:
• Premium Portfolio: {len(self.df_cjmore):,} products across {self.df_cjmore['category'].nunique()} categories
• Brand Diversity: {self.df_cjmore['brand'].nunique():,} unique brands ({len(self.df_cjmore)/self.df_cjmore['brand'].nunique():.1f} products per brand)
• Market Focus: Comprehensive lifestyle supermarket with broad category coverage

Tops Daily Strategic Position:
• Efficient Portfolio: {len(self.df_tops):,} products across {self.df_tops['category'].nunique()} categories  
• Brand Focus: {self.df_tops['brand'].nunique():,} unique brands ({len(self.df_tops)/self.df_tops['brand'].nunique():.1f} products per brand)
• Market Focus: Value-oriented supermarket with concentrated category strategy

Competitive Analysis:
• Portfolio Scale Advantage: CJMore ({(len(self.df_cjmore)/len(self.df_tops)):.1f}x larger product range)
• Brand Diversity Advantage: CJMore ({(self.df_cjmore['brand'].nunique()/self.df_tops['brand'].nunique()):.1f}x more brands)
• Operational Efficiency: Tops Daily (more focused brand and category management)
        """
        
        ax5.text(0.05, 0.6, summary_text, ha='left', va='top', fontsize=11, 
                transform=ax5.transAxes, bbox=dict(boxstyle='round,pad=0.5', 
                facecolor=self.colors['light_accent'], alpha=0.8))
        ax5.axis('off')
        
        plt.suptitle('Comprehensive Market Analysis Dashboard: CJMore vs Tops Daily', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        plt.savefig(self.output_dir / '7_comprehensive_dashboard.png', 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("   ✅ Summary Dashboard saved")
    
    def run_complete_visualization_suite(self):
        """Execute all visualization components"""
        
        print("🚀 COMPREHENSIVE VISUAL COMPARISON SUITE")
        print("=" * 80)
        print("🎨 Creating professional visualizations for CJMore vs Tops Daily")
        print(f"📊 Color Scheme: CJMore ({self.colors['cjmore']}) vs Tops Daily ({self.colors['tops_daily']})")
        print()
        
        # Create all visualizations
        self.create_product_portfolio_comparison()
        self.create_category_distribution_pie_charts()
        self.create_private_label_comparison()
        self.create_treemap_visualization()
        self.create_side_by_side_category_comparison()
        self.create_top_categories_ranked_chart()
        self.create_summary_dashboard()
        
        print("\n" + "=" * 80)
        print("🏆 VISUAL COMPARISON SUITE COMPLETE!")
        print(f"📁 All visualizations saved to: {self.output_dir}")
        print("📊 Generated Charts:")
        print("   1. Product Portfolio Comparison (Bar Chart)")
        print("   2. Category Distribution (Pie Charts)")
        print("   3. Private Label Comparison (Stacked Bar)")
        print("   4. Product Distribution (Treemap)")
        print("   5. Side-by-Side Category Comparison")
        print("   6. Top Categories Ranking")
        print("   7. Comprehensive Dashboard")
        print()
        print("🎯 Ready for executive presentation and strategic analysis!")
        print("=" * 80)

def main():
    """Main execution function"""
    
    try:
        # Create and run visualization suite
        visualizer = SupermarketVisualComparison()
        visualizer.run_complete_visualization_suite()
        
    except Exception as e:
        print(f"❌ Visualization creation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())