"""
Text normalization utilities for Thai and English OCR processing.
Handles Thai digit conversion, size extraction, and text cleaning.
"""
import re
import unicodedata
from typing import List, Dict, Optional, Tuple
import logging

from .config import config

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Text normalization and processing utilities."""
    
    def __init__(self):
        self.thai_digits = config.THAI_DIGITS
        self.size_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in config.SIZE_PATTERNS]
        
        # Additional Thai text patterns
        self.thai_units = {
            'มล.': 'ml',
            'มิลลิลิตร': 'ml', 
            'ลิตร': 'l',
            'กรัม': 'g',
            'กิโลกรัม': 'kg',
            'แพ็ค': 'pack',
            'ออนซ์': 'oz',
            'ปอนด์': 'lb'
        }
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode text to NFC form.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            # Normalize to NFC (canonical composition)
            normalized = unicodedata.normalize('NFC', text)
            return normalized.strip()
        except Exception as e:
            logger.error(f"Unicode normalization failed: {e}")
            return text.strip()
    
    def convert_thai_digits(self, text: str) -> str:
        """
        Convert Thai digits to Arabic numerals.
        
        Args:
            text: Text containing Thai digits
            
        Returns:
            Text with Arabic numerals
        """
        if not text:
            return ""
        
        result = text
        for thai_digit, arabic_digit in self.thai_digits.items():
            result = result.replace(thai_digit, arabic_digit)
        
        return result
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = self.normalize_unicode(text)
        
        # Convert Thai digits
        text = self.convert_thai_digits(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_sizes(self, text: str) -> List[Dict[str, str]]:
        """
        Extract size information from text using regex patterns.
        
        Args:
            text: Input text
            
        Returns:
            List of size dictionaries with 'value', 'unit', 'raw' keys
        """
        if not text:
            return []
        
        sizes = []
        cleaned_text = self.clean_text(text)
        
        # Try each size pattern
        for pattern in self.size_patterns:
            matches = pattern.findall(cleaned_text)
            for match in matches:
                if isinstance(match, tuple):
                    # If pattern has groups, take the first group
                    value = match[0] if match else ""
                else:
                    # Single match
                    value = match
                
                if value:
                    # Extract full match for context
                    full_match = pattern.search(cleaned_text)
                    raw_text = full_match.group(0) if full_match else value
                    
                    # Determine unit from the raw text
                    unit = self._extract_unit_from_text(raw_text)
                    
                    sizes.append({
                        'value': value.replace(',', '.'),  # Normalize decimal separator
                        'unit': unit,
                        'raw': raw_text
                    })
        
        # Remove duplicates while preserving order
        unique_sizes = []
        seen = set()
        for size in sizes:
            size_key = f"{size['value']}_{size['unit']}"
            if size_key not in seen:
                seen.add(size_key)
                unique_sizes.append(size)
        
        return unique_sizes
    
    def _extract_unit_from_text(self, text: str) -> str:
        """Extract unit from text fragment."""
        text_lower = text.lower()
        
        # Check for Thai units first
        for thai_unit, english_unit in self.thai_units.items():
            if thai_unit in text_lower:
                return english_unit
        
        # Check for English units
        if 'ml' in text_lower:
            return 'ml'
        elif 'l' in text_lower and 'ml' not in text_lower:
            return 'l'
        elif 'kg' in text_lower:
            return 'kg'
        elif 'g' in text_lower and 'kg' not in text_lower:
            return 'g'
        elif 'oz' in text_lower:
            return 'oz'
        elif 'lb' in text_lower:
            return 'lb'
        elif 'pack' in text_lower or 'แพ็ค' in text_lower:
            return 'pack'
        elif 'pcs' in text_lower or 'pieces' in text_lower:
            return 'pcs'
        else:
            return 'unknown'
    
    def normalize_size(self, sizes: List[Dict[str, str]]) -> Optional[str]:
        """
        Normalize size to standard format.
        
        Args:
            sizes: List of extracted sizes
            
        Returns:
            Normalized size string or None
        """
        if not sizes:
            return None
        
        # Take the first valid size
        for size in sizes:
            try:
                value = float(size['value'])
                unit = size['unit']
                
                # Convert to standard units
                if unit == 'l':
                    value = value * 1000  # Convert to ml
                    unit = 'ml'
                elif unit == 'kg':
                    value = value * 1000  # Convert to g
                    unit = 'g'
                
                # Format value
                if value == int(value):
                    value_str = str(int(value))
                else:
                    value_str = f"{value:.1f}"
                
                return f"{value_str}{unit}"
                
            except ValueError:
                continue
        
        # Return first raw size if normalization fails
        return sizes[0]['raw']
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text for comparison and analysis.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        cleaned = self.clean_text(text)
        
        # Simple tokenization - split by whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', cleaned, re.UNICODE)
        
        # Convert to lowercase for comparison
        tokens = [token.lower() for token in tokens if len(token) > 1]
        
        return tokens
    
    def calculate_jaccard_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """
        Calculate Jaccard similarity between two token lists.
        
        Args:
            tokens1: First token list
            tokens2: Second token list
            
        Returns:
            Jaccard similarity (0.0 to 1.0)
        """
        if not tokens1 and not tokens2:
            return 1.0
        
        if not tokens1 or not tokens2:
            return 0.0
        
        set1 = set(tokens1)
        set2 = set(tokens2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def extract_brand_candidates(self, text: str) -> List[str]:
        """
        Extract potential brand names from text.
        
        Args:
            text: Input text
            
        Returns:
            List of potential brand names
        """
        if not text:
            return []
        
        candidates = []
        cleaned = self.clean_text(text)
        
        # Look for capitalized words (potential brand names)
        words = cleaned.split()
        for word in words:
            # Skip short words and numbers
            if len(word) < 2 or word.isdigit():
                continue
            
            # Look for words with capital letters
            if any(c.isupper() for c in word):
                candidates.append(word)
        
        return candidates
    
    def handle_dehyphenation(self, text: str) -> str:
        """
        Handle de-hyphenation of words broken across lines.
        
        Args:
            text: Input text potentially with line breaks
            
        Returns:
            Text with de-hyphenated words
        """
        if not text:
            return ""
        
        # Handle common OCR line break patterns
        # Remove hyphens at end of lines followed by newlines
        text = re.sub(r'-\s*\n\s*', '', text)
        
        # Handle soft hyphens and similar characters
        text = re.sub(r'[\u00AD\u2010\u2011]-?\s*\n\s*', '', text)
        
        return text
    
    def classify_text_type(self, text: str) -> str:
        """
        Classify the type of text (product name, price, size, etc.).
        
        Args:
            text: Input text
            
        Returns:
            Text classification
        """
        if not text:
            return "unknown"
        
        cleaned = text.lower().strip()
        
        # Check for price patterns
        if re.search(r'[\฿$€£¥]\s*\d+', cleaned) or re.search(r'\d+\s*[\฿$€£¥บาท]', cleaned):
            return "price"
        
        # Check for size patterns
        sizes = self.extract_sizes(text)
        if sizes:
            return "size"
        
        # Check for promotional text
        promo_keywords = ['sale', 'offer', 'discount', 'promo', 'special', 
                         'ลด', 'พิเศษ', 'โปรโมชั่น', 'เซล']
        if any(keyword in cleaned for keyword in promo_keywords):
            return "promotion"
        
        # Check for category headers
        if len(cleaned.split()) <= 3 and any(c.isupper() for c in text):
            return "category"
        
        return "product"

# Global instance
text_normalizer = TextNormalizer()