"""
Main entry point for the Vietnamese Comment Preprocessing Pipeline.
Provides an easy-to-use API that wraps the decoder, filter, and normalizer.
"""
import os
from .config import EMOJI_MAPPING_PATH, ABBREVIATIONS_PATH
from .decoder import Decoder
from .filter import Filter
from .normalizer import Normalizer
from .word_segmentor import WordSegmentor
from .vietnamese_typing_normalizer import VietnameseNormalizer

class VietnameseCommentPreprocessor:
    def __init__(self, 
                 emoji_mapping_path: str = None, 
                 abbrev_mapping_path: str = None):
        """Initialize the pipeline with appropriate mapping paths."""
        self.emoji_path = emoji_mapping_path or EMOJI_MAPPING_PATH
        self.abbrev_path = abbrev_mapping_path or ABBREVIATIONS_PATH
        
        # We need to rewrite paths if they point to the old "comment/data" folder
        # Now they should point to "k:/fb_crawl/mappings"
        if not os.path.exists(self.emoji_path):
            current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.emoji_path = os.path.join(current_dir, "mappings", "emoji_vi.json")
            self.abbrev_path = os.path.join(current_dir, "mappings", "abbreviations.json")
        
        self.decoder = Decoder(self.emoji_path)
        self.filter = Filter()
        
        self.normalizer = Normalizer(self.abbrev_path)
        self.vn_typing_normalizer = VietnameseNormalizer()
        # Initialize the segmentor with VnCoreNLP backend using the pre-downloaded model
        # The java wrapper appends "/models" internally, so we just point to the root models dir
        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.segmentor_dir = os.path.join(current_dir, "models")
        
        # In case the user's manual download extracted strangely, let's verify where VnCoreNLP-1.2.jar lives
        if not os.path.exists(os.path.join(self.segmentor_dir, "VnCoreNLP-1.2.jar")):
            # Fallback to current dir if models wasn't correct
            self.segmentor_dir = current_dir
            
        self.segmentor = WordSegmentor(backend="vncorenlp", vncorenlp_dir=self.segmentor_dir, auto_download=False)

    def process_comment(self, text: str, 
                        use_decoder: bool = True, 
                        use_filter: bool = True, 
                        use_normalizer: bool = True,
                        use_segmentor: bool = True) -> dict:
        """
        Process a single comment through the pipeline.
        Returns a dictionary containing:
        - raw_text: Original text passed in.
        - cleaned_text: Fully decoded and normalized text.
        - filter_reason: Why it was dropped (or "ok" if kept).
        - is_valid: True if it passes the filters, False otherwise.
        """
        if not text or not text.strip():
            return {"raw_text": text, "cleaned_text": "", "filter_reason": "empty", "is_valid": False}

        # 1. Decode
        current_text = text
        if use_decoder:
            try:
                current_text = self.decoder.decode(current_text)
            except Exception as e:
                return {"raw_text": text, "cleaned_text": "", "filter_reason": f"decode_error: {str(e)}", "is_valid": False}

        # 2. Filter (Before Normalization)
        reason = "ok"
        if use_filter:
            keep, reason = self.filter.filter_comment(current_text, raw_text=text)
            if not keep:
                return {
                    "raw_text": text,
                    "cleaned_text": current_text,
                    "filter_reason": reason,
                    "is_valid": False
                }

        # 3. Normalize
        if use_normalizer:
            try:
                current_text = self.normalizer.normalize(current_text)
            except Exception as e:
                return {"raw_text": text, "cleaned_text": current_text, "filter_reason": f"normalize_error: {str(e)}", "is_valid": False}

        # Final check if normalization produced empty text
        if use_normalizer and len(current_text.strip()) == 0:
             return {"raw_text": text, "cleaned_text": "", "filter_reason": "empty_after_normalize", "is_valid": False}

        # 3.5. Vietnamese tone placement normalization (dấu thanh kiểu mới)
        if use_normalizer:
            try:
                current_text = self.vn_typing_normalizer.normalize(current_text)
            except Exception:
                pass  # Non-critical: if it fails, just use the text as-is

        # 4. Segment (VnCoreNLP)
        if use_segmentor:
            try:
                segmented = self.segmentor.segment(current_text)
                if segmented:
                    current_text = segmented
            except Exception as e:
                # If segment fails, just use the non-segmented text to avoid data loss
                pass

        return {
            "raw_text": text, 
            "cleaned_text": current_text, 
            "filter_reason": reason, 
            "is_valid": True
        }
