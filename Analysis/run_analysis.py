#!/usr/bin/env python3
"""
Supermarket Analysis Runner
==========================
Quick execution script for the comprehensive supermarket analysis
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from supermarket_analysis import SupermarketAnalyzer

def main():
    print("🏪 Supermarket Product Portfolio Analysis")
    print("=" * 50)
    
    # Paths
    input_folder = Path("../brand_analysis_output")
    output_folder = Path(".")
    
    # Verify input data exists
    if not input_folder.exists():
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    excel_file = input_folder / "enhanced_brand_analysis.xlsx"
    if not excel_file.exists():
        print(f"❌ Excel file not found: {excel_file}")
        return
    
    print(f"✅ Input folder: {input_folder}")
    print(f"✅ Output folder: {output_folder}")
    print("")
    
    # Run analysis
    analyzer = SupermarketAnalyzer(str(input_folder), str(output_folder))
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main()