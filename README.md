# Supermarket Analysis - AI-Driven Retail Audit Pipeline

This project implements a comprehensive AI-driven retail audit pipeline that analyzes supermarket photos to extract merchandise and visual merchandising insights.

## Features

- **Multi-format Image Support**: Handles HEIC, JPG, PNG images with automatic HEIC conversion
- **Google Vision API Integration**: Object detection, OCR (Thai + English), and logo detection
- **Gemini 2.5 Pro Consolidation**: Structured JSON output with confidence scoring
- **Duplicate Detection**: Advanced near-duplicate detection using pHash, SSIM, OCR Jaccard similarity
- **Comprehensive Excel Reports**: Detailed findings and aggregated KPIs
- **Thai Language Support**: Full bilingual support with proper text normalization

## Architecture

### Pipeline Steps

1. **Image Ingestion**: Load and preprocess images, compute quality metrics and fingerprints
2. **Object Localization**: Detect products, shelves, signage using Vision API
3. **OCR & Logo Detection**: Extract text and brand logos from crops
4. **Gemini Consolidation**: Structured analysis with confidence scoring and validation
5. **Duplicate Detection**: Cluster near-duplicates using Union-Find algorithm
6. **Excel Export**: Generate comprehensive reports with KPIs and category analysis

### Key Components

- `src/image_utils.py`: Image processing, HEIC conversion, quality assessment
- `src/vision_api.py`: Google Cloud Vision API integration
- `src/text_utils.py`: Thai/English text normalization and size extraction
- `src/gemini_consolidation.py`: Gemini 2.5 Pro structured analysis
- `src/duplicate_detection.py`: Multi-metric duplicate detection with Union-Find clustering
- `src/excel_export.py`: Comprehensive Excel report generation
- `main.py`: Pipeline orchestrator with CLI interface

## Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup Google Cloud Credentials**:
   - Create a service account with Vision API access
   - Download the service account key JSON file
   - Get a Gemini API key from Google AI for Developers

3. **Configure Environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

## Configuration

Edit `.env` file:

```env
# Google Cloud Credentials
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
GOOGLE_API_KEY=your_gemini_api_key

# Model Configuration
GEMINI_MODEL=gemini-2.5-pro

# Processing Settings
MAX_IMAGES=0
LOGO_MIN_CONF=0.75
ACCEPT_CONF=0.75

# Duplicate Detection Thresholds
PHASH_MAX_DIST=6
SSIM_MIN=0.92
JACCARD_MIN=0.80
```

## Usage

### Basic Usage
```bash
python main.py --assets ./assets --out ./outputs
```

### Advanced Options
```bash
# Process limited number of images
python main.py --assets ./assets --out ./outputs --max_images 50

# Strict mode (fail on any errors)
python main.py --assets ./assets --out ./outputs --strict

# Vision-only dry run (skip Gemini)
python main.py --assets ./assets --out ./outputs --no_gemini

# Debug logging
python main.py --assets ./assets --out ./outputs --log_level DEBUG
```

## Output Structure

```
outputs/
├── excel/
│   └── audit.xlsx          # Main Excel report
├── crops/
│   └── *.jpg              # Product/signage crops
├── thumbs/
│   └── *_thumb.jpg        # Thumbnail images
└── json/
    ├── *_metadata.json    # Image metadata
    ├── *_context.json     # OCR/logo results
    ├── *_consolidated.json # Gemini analysis
    └── duplicate_results.json # Duplicate detection
```

## Excel Report Sheets

### 1. All Findings
Complete dataset with every detected crop including duplicates:
- Product details (bilingual)
- Category classification
- Placement analysis
- Confidence scores
- File references

### 2. Deduped Summary
Category-level aggregations from canonical images only:
- SKU counts and variety metrics
- Private label analysis
- Category roles (Destination/Routine/Impulse)
- Visual merchandising KPIs

### 3. KPIs
High-level summary metrics:
- Total images and duplicates
- Private label share
- Top categories by facings
- Promotional rates

### 4. Issues
Items requiring manual review:
- Low confidence detections
- Consistency conflicts
- Review queue

## Key Features

### Thai Language Support
- Proper Unicode normalization (NFC)
- Thai digit conversion (๐-๙ → 0-9)
- Bilingual field extraction
- Size pattern recognition for Thai units

### Duplicate Detection
Uses multi-criteria approach requiring ≥2 conditions:
- pHash Hamming distance ≤ 6
- SSIM ≥ 0.92  
- OCR Jaccard similarity ≥ 0.80
- Embedding cosine similarity ≥ 0.98 (placeholder)

### Confidence Scoring
```
conf_overall = 0.40×conf_logo + 0.35×conf_ocr + 0.15×conf_detector + 0.10×embedding
```

Items with `conf_overall < 0.75` are flagged for review.

### Category Role Classification
- **Destination**: High facings, category headers, low promo rate
- **Routine**: Moderate facings, regular distribution
- **Impulse**: Many endcaps/checkout, high promo rate, small packs

## Error Handling

- Exponential backoff for API retries
- Graceful failure handling with continuation
- Comprehensive logging and intermediate file saves
- Validation with Pydantic models

## Performance Notes

- Processes ~100 images efficiently with batch operations
- API rate limiting with delays between requests
- Memory-efficient image processing with streaming
- Intermediate JSON saves for fault tolerance

## Requirements

- Python 3.8+
- Google Cloud Vision API access
- Google AI Gemini API access
- ~2GB RAM for typical workloads
- Internet connectivity for API calls

## Cost Estimation

For ~100 images:
- Vision API: ~$0.50-2.00 (depending on crop count)
- Gemini 2.5 Pro: ~$0.10-0.50 (compact JSON outputs)
- Total: Under $3 for typical retail audit

## Troubleshooting

### Common Issues

1. **HEIC Images Not Loading**:
   - Ensure `pillow-heif` is installed
   - Check file permissions

2. **Vision API Errors**:
   - Verify service account permissions
   - Check quota limits

3. **Gemini JSON Parsing**:
   - Use `--strict` mode to catch errors
   - Check API key validity

4. **Memory Issues**:
   - Reduce `--max_images`
   - Use smaller batch sizes

### Debug Mode
```bash
python main.py --log_level DEBUG --max_images 5
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality  
4. Submit pull request

## Support

For questions or issues:
1. Check the logs in `retail_audit.log`
2. Review intermediate JSON files in `outputs/json/`
3. Use debug mode for detailed tracing