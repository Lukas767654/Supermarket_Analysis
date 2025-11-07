"""
Gemini 2.5 Pro integration for consolidating crop analysis into structured JSON.
"""
import json
import logging
import time
from typing import Optional, Dict, Any
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

from .config import config
from .models import CropContext, ConsolidatedResult
from .text_utils import text_normalizer

logger = logging.getLogger(__name__)

class GeminiConsolidator:
    """Gemini 2.5 Pro consolidation service."""
    
    def __init__(self):
        """Initialize Gemini client."""
        try:
            genai.configure(api_key=config.GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=config.GEMINI_MODEL,
                generation_config={
                    "temperature": 0.1,  # Low temperature for consistent structured output
                    "top_p": 0.8,
                    "top_k": 40,
                    "max_output_tokens": 2048,
                    "response_mime_type": "application/json"  # Force JSON output
                },
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            logger.info(f"Gemini {config.GEMINI_MODEL} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.model = None
    
    def create_consolidation_prompt(self, context: CropContext) -> str:
        """
        Create structured prompt for Gemini consolidation.
        
        Args:
            context: Crop context with OCR and logo data
            
        Returns:
            Formatted prompt string
        """
        # Extract relevant information
        ocr_text = context.ocr_raw or ""
        tokens = ", ".join(context.ocr_tokens) if context.ocr_tokens else "None"
        logo_info = ""
        if context.logo_candidates:
            logo_info = "; ".join([f"{logo['brand']} (conf: {logo['score']:.2f})" 
                                 for logo in context.logo_candidates])
        else:
            logo_info = "None"
        
        # Extract sizes using text normalizer
        sizes = text_normalizer.extract_sizes(ocr_text)
        size_info = "; ".join([f"{s['value']} {s['unit']}" for s in sizes]) if sizes else "None"
        
        prompt = f"""
You are an expert retail analyst. Analyze this supermarket product crop and output ONLY valid JSON with the exact schema provided.

CROP ANALYSIS DATA:
- Image ID: {context.image_id}
- Crop ID: {context.crop_id}
- Detected Type: {context.type}
- OCR Text: "{ocr_text}"
- OCR Tokens: {tokens}
- Logo Candidates: {logo_info}
- Detected Sizes: {size_info}
- Confidence Scores: OCR={context.scores.get('ocr_conf', 0):.2f}, Logo={context.scores.get('logo_conf', 0):.2f}, Detector={context.scores.get('det_conf', 0):.2f}

ANALYSIS RULES:
1. NEVER translate brand names - keep them exactly as detected
2. Support Thai + English text (provide both _th and _en fields where applicable)
3. Extract product info: brand, product name, variant, size, category
4. Determine if this is a private label product (store brand vs national brand)
5. Assess placement (EyeLevel if crop center is 40-60% of image height, Endcap if wide/prominent, Checkout if small items)
6. Identify signage type (CategoryHeader, Promo, Claim, Price, or None)
7. Set review_needed=true if logo/brand/size/category conflict OR overall confidence < 0.75
8. Estimate facings (number of same SKU visible in shelf row)

CONFIDENCE CALCULATION:
- conf_overall = 0.40*conf_logo + 0.35*conf_ocr + 0.15*conf_detector + 0.10*embedding_match (set embedding_match=0)
- Use actual scores provided above

OUTPUT ONLY THIS JSON SCHEMA (no other text):
{{
  "image_id": "{context.image_id}",
  "crop_id": "{context.crop_id}",
  "brand_th": "string|null",
  "brand_en": "string|null", 
  "product_th": "string|null",
  "product_en": "string|null",
  "variant_th": "string|null",
  "variant_en": "string|null",
  "size_raw": "string|null",
  "size_normalized": "string|null",
  "category_th": "string|null",
  "category_en": "string|null",
  "subcategory_th": "string|null", 
  "subcategory_en": "string|null",
  "signage_type": "CategoryHeader|Promo|Claim|Price|None",
  "private_label": "yes|no|unknown",
  "facings_hint": "integer|null",
  "placement": "EyeLevel|Endcap|Checkout|None",
  "conf_logo": {context.scores.get('logo_conf', 0):.3f},
  "conf_ocr": {context.scores.get('ocr_conf', 0):.3f},
  "conf_detector": {context.scores.get('det_conf', 0):.3f},
  "conf_overall": 0.000,
  "review_needed": false,
  "notes": "string|null"
}}
"""
        return prompt
    
    def consolidate_crop(self, context: CropContext, 
                        retry_count: int = 1) -> Optional[ConsolidatedResult]:
        """
        Consolidate crop analysis using Gemini 2.5 Pro.
        
        Args:
            context: Crop context data
            retry_count: Number of retries for failed requests
            
        Returns:
            ConsolidatedResult or None if failed
        """
        if not self.model:
            logger.error("Gemini model not initialized")
            return None
        
        prompt = self.create_consolidation_prompt(context)
        
        for attempt in range(retry_count + 1):
            try:
                # Generate response
                response = self.model.generate_content(prompt)
                
                if not response or not response.text:
                    logger.error(f"Empty response from Gemini for crop {context.crop_id}")
                    continue
                
                # Parse JSON response
                try:
                    result_dict = json.loads(response.text)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from Gemini for crop {context.crop_id}: {e}")
                    logger.error(f"Response text: {response.text[:500]}...")
                    
                    # Try to extract JSON from response if wrapped in markdown
                    json_match = self._extract_json_from_response(response.text)
                    if json_match:
                        try:
                            result_dict = json.loads(json_match)
                        except json.JSONDecodeError:
                            if attempt < retry_count:
                                logger.warning(f"Retrying Gemini request for crop {context.crop_id}")
                                time.sleep(1)  # Brief delay before retry
                                continue
                            else:
                                logger.error(f"Failed to parse JSON after {retry_count} retries")
                                return None
                    else:
                        if attempt < retry_count:
                            logger.warning(f"Retrying Gemini request for crop {context.crop_id}")
                            time.sleep(1)
                            continue
                        else:
                            return None
                
                # Validate and create ConsolidatedResult
                try:
                    result = ConsolidatedResult(**result_dict)
                    
                    # Post-process the result
                    result = self._post_process_result(result, context)
                    
                    logger.debug(f"Successfully consolidated crop {context.crop_id}")
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed to validate Gemini response for crop {context.crop_id}: {e}")
                    logger.error(f"Response dict: {result_dict}")
                    
                    if attempt < retry_count:
                        logger.warning(f"Retrying Gemini request for crop {context.crop_id}")
                        time.sleep(1)
                        continue
                    else:
                        return None
                
            except Exception as e:
                logger.error(f"Gemini API error for crop {context.crop_id}: {e}")
                
                if attempt < retry_count:
                    logger.warning(f"Retrying Gemini request for crop {context.crop_id}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return None
        
        return None
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """
        Extract JSON from response that might be wrapped in markdown or have extra text.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Extracted JSON string or None
        """
        import re
        
        # Try to find JSON block in markdown
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # Try to find JSON block without markdown
        json_match = re.search(r'(\{.*?\})', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        return None
    
    def _post_process_result(self, result: ConsolidatedResult, 
                           context: CropContext) -> ConsolidatedResult:
        """
        Post-process the consolidated result for consistency and quality.
        
        Args:
            result: Initial consolidated result
            context: Original crop context
            
        Returns:
            Post-processed result
        """
        # Normalize size using text utils
        if result.size_raw:
            sizes = text_normalizer.extract_sizes(result.size_raw)
            if sizes:
                normalized_size = text_normalizer.normalize_size(sizes)
                if normalized_size:
                    result.size_normalized = normalized_size
        
        # Determine brand family
        if result.brand_en:
            brand_family = config.BRAND_FAMILIES.get(result.brand_en)
            if brand_family:
                # Store in notes for now (can be added to model later)
                brand_note = f"Brand family: {brand_family}"
                if result.notes:
                    result.notes = f"{result.notes}; {brand_note}"
                else:
                    result.notes = brand_note
        
        # Validate confidence score calculation
        embedding_match = 0  # Set to 0 as specified
        calculated_conf = (
            0.40 * result.conf_logo + 
            0.35 * result.conf_ocr + 
            0.15 * result.conf_detector + 
            0.10 * embedding_match
        )
        result.conf_overall = round(calculated_conf, 3)
        
        # Update review_needed based on confidence
        if result.conf_overall < config.ACCEPT_CONF:
            result.review_needed = True
        
        # Consistency checks
        consistency_issues = []
        
        # Check logo vs brand consistency
        if context.logo_candidates and result.brand_en:
            logo_brands = [logo['brand'].lower() for logo in context.logo_candidates]
            if result.brand_en.lower() not in logo_brands:
                consistency_issues.append("Logo-brand mismatch")
        
        # Check if private label determination makes sense
        if result.private_label == "yes" and result.brand_en:
            # Common private label indicators
            pl_indicators = ['great value', 'kirkland', 'store brand', 'own brand']
            if not any(indicator in result.brand_en.lower() for indicator in pl_indicators):
                # This might need review
                consistency_issues.append("Private label unclear")
        
        # Add consistency issues to notes
        if consistency_issues:
            issues_note = f"Issues: {'; '.join(consistency_issues)}"
            if result.notes:
                result.notes = f"{result.notes}; {issues_note}"
            else:
                result.notes = issues_note
            result.review_needed = True
        
        return result
    
    def batch_consolidate(self, contexts: list[CropContext], 
                         batch_size: int = 5) -> Dict[str, Optional[ConsolidatedResult]]:
        """
        Process multiple crops in batches to manage API rate limits.
        
        Args:
            contexts: List of crop contexts
            batch_size: Number of contexts to process concurrently
            
        Returns:
            Dictionary mapping crop_id to ConsolidatedResult
        """
        results = {}
        
        for i in range(0, len(contexts), batch_size):
            batch = contexts[i:i + batch_size]
            
            logger.info(f"Processing consolidation batch {i//batch_size + 1}/{(len(contexts)-1)//batch_size + 1}")
            
            for context in batch:
                result = self.consolidate_crop(context)
                results[context.crop_id] = result
                
                # Brief delay to respect rate limits
                time.sleep(0.1)
        
        return results

# Global instance
gemini_consolidator = GeminiConsolidator()