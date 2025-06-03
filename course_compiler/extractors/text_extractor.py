"""
Plain text file extractor implementation.
"""
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
import chardet
from ..logging_utils import get_logger
from .base import BaseExtractor, ExtractionError

logger = get_logger(__name__)

class TextExtractor(BaseExtractor):
    """Extracts content from plain text files with automatic encoding detection."""

    def __init__(self, file_path: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(file_path, config)
        self.encoding = self.config.get('encoding', 'utf-8')
        self.chunk_size = self.config.get('chunk_size', 8192)  # For encoding detection

    def detect_encoding(self) -> str:
        """
        Detect the encoding of the text file.

        Returns:
            Detected encoding (defaults to 'utf-8' if detection fails)
        """
        try:
            # Read a chunk of the file for detection
            with open(self.file_path, 'rb') as f:
                raw_data = f.read(self.chunk_size)

            result = chardet.detect(raw_data)
            confidence = result.get('confidence', 0)
            encoding = result.get('encoding', 'utf-8').lower()

            # Only use detected encoding if confidence is high enough
            if confidence > 0.7:
                logger.debug(f"Detected encoding '{encoding}' with confidence {confidence:.2f}")
                return encoding

            logger.debug(f"Low confidence ({confidence:.2f}) in detected encoding '{encoding}'. Using default 'utf-8'")
            return 'utf-8'

        except Exception as e:
            logger.warning(f"Error detecting file encoding: {e}. Using default 'utf-8'")
            return 'utf-8'

    def extract(self) -> Dict[str, Any]:
        """
        Extract content from the text file.

        Returns:
            Dictionary containing the file content and metadata
        """
        try:
            metadata = self.get_metadata()

            # Detect encoding if not specified
            encoding = self.encoding
            if encoding == 'auto':
                encoding = self.detect_encoding()

            # Read the file with detected encoding
            with open(self.file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()

            # Split into lines for easier processing
            lines = content.splitlines()

            # Simple paragraph detection (group consecutive non-empty lines)
            paragraphs = []
            current_para = []

            for line in lines:
                line = line.strip()
                if line:  # Non-empty line
                    current_para.append(line)
                elif current_para:  # Empty line after content
                    paragraphs.append(' '.join(current_para))
                    current_para = []

            # Add the last paragraph if there's any content
            if current_para:
                paragraphs.append(' '.join(current_para))

            return {
                'metadata': metadata,
                'content': content,
                'paragraphs': paragraphs,
                'line_count': len(lines),
                'word_count': sum(len(line.split()) for line in lines),
                'file_type': 'text',
                'encoding': encoding
            }

        except UnicodeDecodeError as e:
            error_msg = f"Failed to decode file with encoding '{encoding}': {e}"
            logger.error(error_msg)
            raise ExtractionError(error_msg) from e

        except Exception as e:
            error_msg = f"Error extracting content from {self.file_path}: {e}"
            logger.error(error_msg)
            raise ExtractionError(error_msg) from e

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata including basic file info and text statistics.
        """
        metadata = super().get_metadata()

        try:
            # Add text-specific metadata
            with open(self.file_path, 'rb') as f:
                content = f.read()

            # Try to decode for text stats
            try:
                text = content.decode(self.encoding if self.encoding != 'auto' else 'utf-8', errors='replace')
                lines = text.splitlines()
                words = [word for line in lines for word in line.split()]

                metadata.update({
                    'line_count': len(lines),
                    'word_count': len(words),
                    'char_count': len(text),
                    'is_binary': False
                })
            except (UnicodeDecodeError, LookupError):
                # If we can't decode, it's likely a binary file
                metadata.update({
                    'is_binary': True,
                    'size_bytes': len(content)
                })

        except Exception as e:
            logger.warning(f"Error collecting text metadata: {e}")

        return metadata
