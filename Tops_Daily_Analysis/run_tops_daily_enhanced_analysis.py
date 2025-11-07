#!/usr/bin/env python3
"""
Tops Daily Supermarket Analysis Script
=====================================

Runs the same enhanced analysis as CJMore but configured for Tops Daily.
Generates visualizations and reports for Tops Daily supermarket data.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

def run_tops_daily_analysis():
    """Run enhanced analysis for Tops Daily"""
    
    print("🚀 Starting Tops Daily Enhanced Analysis")
    print("=" * 60)
    
    try:
        # Import the enhanced analysis from Analysis folder
        sys.path.append(str(Path(__file__).parent.parent / 'Analysis'))
        from enhanced_supermarket_analysis import SupermarketAnalyzer
        
        # Configuration for Tops Daily
        INPUT_FOLDER = "../tops_daily_analysis_output"  # Where pipeline results are
        OUTPUT_FOLDER = "."  # Current directory (Tops_Daily_Analysis)
        
        # Create analyzer and run analysis
        analyzer = SupermarketAnalyzer(INPUT_FOLDER, OUTPUT_FOLDER)
        
        # Override title for Tops Daily
        print("📊 Running analysis for: Tops Daily Supermarket")
        
        # Run complete analysis
        analyzer.run_complete_analysis()
        
        print("\n" + "=" * 60)
        print("✅ Tops Daily Enhanced Analysis Complete!")
        print("📁 Results in Tops_Daily_Analysis/")
        print("📊 Visualizations: visualizations/")
        print("📋 Reports: reports/")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure to run the Tops Daily pipeline first!")
        return False
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        return False
    
    return True

def main():
    """Main execution function"""
    run_tops_daily_analysis()

if __name__ == "__main__":
    main()