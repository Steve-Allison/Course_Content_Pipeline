"""
Extractors package for handling different file types.

This package contains modules for extracting content from various file formats.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, Type, List, Tuple
import mimetypes
import os

from ..config import config
from ..logging_utils import get_logger

# Import extractors
from .base import BaseExtractor, ExtractionError
from .pptx_extractor import PptxExtractor
from .docx_extractor import DocxExtractor
from .pdf_extractor import PdfExtractor
from .text_extractor import TextExtractor
from .xlsx_extractor import XlsxExtractor

logger = get_logger(__name__)

# Map file extensions to their respective extractors
DEFAULT_EXTRACTORS = {
    # PowerPoint formats
    '.pptx': PptxExtractor,
    '.ppt': PptxExtractor,  # For older PowerPoint formats

    # Word document formats
    '.docx': DocxExtractor,
    '.doc': DocxExtractor,  # For older Word formats

    # PDF documents
    '.pdf': PdfExtractor,

    # Excel files
    '.xlsx': XlsxExtractor,
    '.xlsm': XlsxExtractor,  # Excel with macros
    '.xltx': XlsxExtractor,  # Excel template
    '.xls': XlsxExtractor,   # Older Excel format (will have limited support)

    # Text files
    '.txt': TextExtractor,
    '.md': TextExtractor,   # Markdown files
    '.rst': TextExtractor,  # reStructuredText
    '.csv': TextExtractor,  # Comma-separated values
    '.json': TextExtractor, # JSON files
    '.xml': TextExtractor,  # XML files
    '.html': TextExtractor, # HTML files
    '.htm': TextExtractor,  # HTML files (short extension)
}

# Additional MIME type mappings
MIME_TYPE_MAP = {
    # PowerPoint
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': PptxExtractor,
    'application/vnd.ms-powerpoint': PptxExtractor,
    
    # Word
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': DocxExtractor,
    'application/msword': DocxExtractor,
    
    # PDF
    'application/pdf': PdfExtractor,
    
    # Excel
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': XlsxExtractor,
    'application/vnd.ms-excel': XlsxExtractor,
    'application/vnd.ms-excel.sheet.macroEnabled.12': XlsxExtractor,
    'application/vnd.ms-excel.template.macroEnabled.12': XlsxExtractor,
    'application/vnd.ms-excel.sheet.binary.macroEnabled.12': XlsxExtractor,
    
    # Text formats
    'text/plain': TextExtractor,
    'text/markdown': TextExtractor,
    'text/x-markdown': TextExtractor,
    'text/csv': TextExtractor,
    'application/json': TextExtractor,
    'application/xml': TextExtractor,
    'text/xml': TextExtractor,
    'text/html': TextExtractor,
}

# Register custom extractors if any
CUSTOM_EXTRACTORS = {}

def register_extractor(extensions: Union[str, List[str]], extractor_class: Type[BaseExtractor]) -> None:
    """
    Register a custom extractor for the given file extensions.

    Args:
        extensions: File extension(s) to register (with leading dot)
        extractor_class: Extractor class to use for these extensions
    """
    if isinstance(extensions, str):
        extensions = [extensions]

    for ext in extensions:
        ext = ext.lower()
        if not ext.startswith('.'):
            ext = f'.{ext}'
        CUSTOM_EXTRACTORS[ext] = extractor_class
        logger.debug(f"Registered extractor {extractor_class.__name__} for extension {ext}")

def get_file_mime_type(file_path: Union[str, Path]) -> Tuple[Optional[str], Optional[str]]:
    """
    Get the MIME type and encoding of a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Tuple of (mime_type, encoding)
    """
    # First try python-magic if available
    try:
        import magic
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(str(file_path))
            if mime_type and mime_type != 'application/octet-stream':
                return mime_type, None
        except Exception as e:
            logger.debug(f"python-magic error: {e}")
    except ImportError:
        logger.debug("python-magic not available, falling back to mimetypes")
    
    # Fallback to mimetypes
    mime_type, encoding = mimetypes.guess_type(file_path)
    
    # Additional manual MIME type detection for common cases
    if not mime_type or mime_type == 'application/octet-stream':
        ext = Path(file_path).suffix.lower()
        if ext == '.xlsx':
            return 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', None
        elif ext == '.docx':
            return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', None
        elif ext == '.pptx':
            return 'application/vnd.openxmlformats-officedocument.presentationml.presentation', None
        elif ext == '.pdf':
            return 'application/pdf', None
    
    return mime_type, encoding

def get_extractor(file_path: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> Optional[BaseExtractor]:
    """
    Get the appropriate extractor for the given file.

    Args:
        file_path: Path to the file to extract content from
        config: Optional configuration dictionary for the extractor

    Returns:
        An instance of the appropriate extractor class, or None if no extractor is available
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")

    # Check custom extractors first
    ext = file_path.suffix.lower()
    if ext in CUSTOM_EXTRACTORS:
        return CUSTOM_EXTRACTORS[ext](file_path, config)

    # Check default extractors
    if ext in DEFAULT_EXTRACTORS:
        return DEFAULT_EXTRACTORS[ext](file_path, config)

    # Try to determine by MIME type
    mime_type, _ = get_file_mime_type(file_path)
    if mime_type and mime_type in MIME_TYPE_MAP:
        return MIME_TYPE_MAP[mime_type](file_path, config)

    # If we still don't have an extractor, try to use the text extractor for text files
    if mime_type and mime_type.startswith('text/'):
        return TextExtractor(file_path, config)

    return None

def extract_content(file_path: Union[str, Path], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Extract content from a file using the appropriate extractor.

    Args:
        file_path: Path to the file to extract content from
        config: Optional configuration dictionary for the extractor

    Returns:
        Dictionary containing the extracted content and metadata

    Raises:
        ValueError: If no extractor is available for the file type
        ExtractionError: If there's an error during extraction
    """
    extractor = get_extractor(file_path, config)
    if not extractor:
        raise ValueError(f"No extractor available for file: {file_path}")

    return extractor.extract()

# Initialize MIME types
mimetypes.init()

# Add some common MIME types that might be missing
mimetypes.add_type('application/vnd.openxmlformats-officedocument.presentationml.presentation', '.pptx')
mimetypes.add_type('application/vnd.openxmlformats-officedocument.wordprocessingml.document', '.docx')
mimetypes.add_type('application/vnd.ms-powerpoint', '.ppt')
mimetypes.add_type('application/msword', '.doc')
mimetypes.add_type('text/markdown', '.md')
mimetypes.add_type('text/x-markdown', '.md')
