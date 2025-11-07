#!/usr/bin/env python3
"""
🚀 Complete Supermarket Brand Analysis - Main Launcher
====================================================
Starts the complete pipeline for analyzing supermarket shelf images:
- ✅ Full-image analysis (no segmentation)
- 🇹🇭 Thai vs International brand classification  
- 👁️ Eye-level detection for shelf positioning
- 📊 Multi-level Excel reports
- 🔍 Enhanced product categorization
"""

import sys
import os
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('supermarket_analysis.log')
    ]
)

logger = logging.getLogger(__name__)

def check_requirements():
    """Check if all requirements are met."""
    
    print("🔧 Checking System Requirements...")
    print("=" * 50)
    
    required_packages = [
        'requests', 'PIL', 'pandas', 'numpy', 
        'cv2', 'sklearn', 'openpyxl', 'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                from PIL import Image
            elif package == 'cv2':
                import cv2
            elif package == 'sklearn':
                from sklearn.cluster import DBSCAN
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING!")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {missing_packages}")
        print("Run: pip install " + " ".join(missing_packages))
        return False
    
    return True

def check_configuration():
    """Check configuration files."""
    
    print("\n⚙️  Checking Configuration...")
    print("=" * 50)
    
    config_files = [
        'config_brand_analysis.py',
        'supermarket_catalog.py', 
        'product_categories.py',
        'enhanced_brand_pipeline.py'
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ {config_file}")
        else:
            print(f"❌ {config_file} - MISSING!")
            return False
    
    # Test configuration import
    try:
        from config_brand_analysis import GOOGLE_API_KEY, IMAGE_FOLDER, OUTPUT_FOLDER
        
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
            print("⚠️  Google API Key nicht gesetzt!")
            return False
        
        print(f"✅ Google API Key configured")
        print(f"✅ Image Folder: {IMAGE_FOLDER}")
        print(f"✅ Output Folder: {OUTPUT_FOLDER}")
        
    except ImportError as e:
        print(f"❌ Configuration import failed: {e}")
        return False
    
    return True

def check_images():
    """Check for images to analyze."""
    
    print("\n📷 Checking Images...")
    print("=" * 50)
    
    from config_brand_analysis import IMAGE_FOLDER, SUPPORTED_EXTENSIONS
    
    if not IMAGE_FOLDER.exists():
        print(f"❌ Images folder not found: {IMAGE_FOLDER}")
        return False
    
    # Count images
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(list(IMAGE_FOLDER.glob(f"*{ext}")))
        image_files.extend(list(IMAGE_FOLDER.glob(f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        print(f"❌ No images found in {IMAGE_FOLDER}")
        print(f"   Please add images with extensions: {SUPPORTED_EXTENSIONS}")
        return False
    
    print(f"✅ Found {len(image_files)} images:")
    for img in image_files[:5]:  # Show first 5
        print(f"   📸 {img.name}")
    
    if len(image_files) > 5:
        print(f"   ... and {len(image_files) - 5} more")
    
    return True

def run_analysis():
    """Run the complete analysis pipeline."""
    
    print("\n🚀 Starting Supermarket Brand Analysis...")
    print("=" * 50)
    
    try:
        # Import and run enhanced pipeline
        from enhanced_brand_pipeline import main as run_enhanced_pipeline
        
        print("📊 Running Enhanced Brand Analysis Pipeline...")
        results_path = run_enhanced_pipeline()
        
        print(f"\n✅ Analysis Complete!")
        print(f"📄 Results saved: {results_path}")
        
        # Show output structure
        from config_brand_analysis import OUTPUT_FOLDER
        
        print(f"\n📁 Generated Files:")
        output_files = list(OUTPUT_FOLDER.rglob("*"))
        for file_path in sorted(output_files):
            if file_path.is_file():
                size_kb = file_path.stat().st_size / 1024
                print(f"  📄 {file_path.name} ({size_kb:.1f} KB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\n❌ Analysis failed: {e}")
        return False

def main():
    """Main launcher function."""
    
    print("🏪 SUPERMARKET BRAND ANALYSIS PIPELINE")
    print("=" * 50)
    print("🎯 Features:")
    print("  • Full-image analysis (no segmentation)")
    print("  • Thai vs International brand classification")
    print("  • Eye-level shelf position detection")
    print("  • Enhanced product categorization")
    print("  • Multi-level Excel reports")
    print("=" * 50)
    
    # Step 1: Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        return False
    
    # Step 2: Check configuration
    if not check_configuration():
        print("\n❌ Configuration check failed!")
        return False
    
    # Step 3: Check images
    if not check_images():
        print("\n❌ Images check failed!")
        print("\n💡 Add images to the 'images' folder and try again.")
        return False
    
    # Step 4: Run analysis
    success = run_analysis()
    
    if success:
        print(f"\n🎉 SUCCESS! Supermarket analysis completed.")
        print(f"💡 Check the output folder for detailed results.")
        
        # Show next steps
        print(f"\n📋 Next Steps:")
        print(f"  1. Open the Excel files to view analysis results")
        print(f"  2. Add more images to analyze additional shelves")
        print(f"  3. Adjust brand lists in config_brand_analysis.py")
        
    else:
        print(f"\n💥 FAILED! Check the logs for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)