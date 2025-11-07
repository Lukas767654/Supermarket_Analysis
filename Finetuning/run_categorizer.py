#!/usr/bin/env python3
"""
Simple runner for the enhanced categorization system
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_categorizer import ProductTypeCategorizer
import logging

def main():
    """Simple runner for category enhancement"""
    
    # Setup simple logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    print("🚀 Enhanced Product Category Classification")
    print("=" * 50)
    
    # Check paths
    input_path = "../brand_analysis_output/enhanced_results.json"
    output_path = "./enhanced_results_improved_categories.json"
    
    if not Path(input_path).exists():
        print(f"❌ Input file not found: {input_path}")
        print("Please make sure the enhanced_results.json file exists")
        return
    
    print(f"📥 Input: {input_path}")
    print(f"📤 Output: {output_path}")
    print()
    
    # Create and run categorizer
    try:
        categorizer = ProductTypeCategorizer()
        enhanced_data = categorizer.enhance_categorization(input_path, output_path)
        
        print()
        print("🎉 Success! Enhanced categorization complete!")
        print(f"📊 Check the reports in the Finetuning folder")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()