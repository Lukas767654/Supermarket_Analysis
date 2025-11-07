#!/usr/bin/env python3
"""
Tops Daily Supermarket Brand Analysis Runner
===========================================

Simple script to run the complete brand analysis for Tops Daily supermarket.
Uses the same pipeline as CJMore but with Tops Daily configuration.
"""

import sys
import os
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def run_tops_daily_analysis():
    """Run complete Tops Daily brand analysis"""
    
    print("🚀 Starting Tops Daily Brand Analysis")
    print("=" * 60)
    
    try:
        # Import the enhanced pipeline
        from enhanced_brand_pipeline import main
        
        # Run with Tops Daily configuration
        main('tops_daily')
        
        print("\n" + "=" * 60)
        print("✅ Tops Daily Analysis Complete!")
        print("📁 Results saved in: tops_daily_analysis_output/")
        print("📊 Excel Report: tops_daily_brand_analysis.xlsx")
        print("=" * 60)
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure enhanced_brand_pipeline.py is in the same directory")
        return False
        
    except Exception as e:
        print(f"❌ Analysis Error: {e}")
        return False
    
    return True

def run_comparison_analysis():
    """Run both CJMore and Tops Daily for comparison"""
    
    print("🔄 Running Comparison Analysis: CJMore vs Tops Daily")
    print("=" * 80)
    
    # Import the pipeline
    from enhanced_brand_pipeline import main
    
    try:
        print("\n1️⃣ Running CJMore Analysis...")
        main('cjmore')
        print("✅ CJMore analysis complete")
        
        print("\n2️⃣ Running Tops Daily Analysis...")  
        main('tops_daily')
        print("✅ Tops Daily analysis complete")
        
        print("\n" + "=" * 80)
        print("🎉 Comparison Analysis Complete!")
        print("📊 CJMore Results: brand_analysis_output/")
        print("📊 Tops Daily Results: tops_daily_analysis_output/")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Comparison Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'compare':
            run_comparison_analysis()
        elif sys.argv[1] == 'help':
            print("Tops Daily Analysis Runner")
            print("Usage:")
            print("  python run_tops_daily_analysis.py          # Run Tops Daily only")
            print("  python run_tops_daily_analysis.py compare  # Run both CJMore and Tops Daily")
            print("  python run_tops_daily_analysis.py help     # Show this help")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use 'help' to see available options")
    else:
        # Default: Run Tops Daily analysis only
        run_tops_daily_analysis()