"""
PDF file extractor implementation.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import fitz  # PyMuPDF
import io
import os
import logging
from PIL import Image

from .base import BaseExtractor, ExtractionError
from ..logging_utils import get_logger

logger = get_logger(__name__)

class PdfExtractor(BaseExtractor):
    """Extracts content from PDF files including text, tables, and images."""

    def __init__(self, file_path: Path, config: Optional[Dict[str, Any]] = None):
        super().__init__(file_path, config)
        self.document = None
        self.images_dir = self.config.get('images_dir', 'images')
        self.extract_tables = self.config.get('extract_tables', True)
        self.extract_images = self.config.get('extract_images', True)

    def load(self) -> None:
        """Load the PDF document."""
        try:
            self.document = fitz.open(self.file_path)
        except Exception as e:
            raise ExtractionError(f"Failed to load PDF file: {e}") from e

    def extract(self) -> Dict[str, Any]:
        """
        Extract content from the PDF file.

        Returns:
            Dictionary containing pages, text, tables, and metadata
        """
        if self.document is None:
            self.load()

        try:
            metadata = self.get_metadata()

            # Create output directory for images if it doesn't exist
            if self.extract_images:
                os.makedirs(self.images_dir, exist_ok=True)

            content = {
                'metadata': metadata,
                'pages': [],
                'tables': [],
                'images': [],
                'file_type': 'pdf',
                'page_count': len(self.document)
            }

            # Process each page
            for page_num in range(len(self.document)):
                page = self.document.load_page(page_num)
                page_content = self._process_page(page, page_num)
                content['pages'].append(page_content)

                # Extract tables if enabled
                if self.extract_tables:
                    tables = self._extract_tables(page, page_num)
                    content['tables'].extend(tables)

                # Extract images if enabled
                if self.extract_images:
                    images = self._extract_images(page, page_num)
                    content['images'].extend(images)

            return content

        except Exception as e:
            logger.error(f"Error extracting content from {self.file_path}: {e}")
            raise ExtractionError(f"Failed to extract content: {e}") from e
        finally:
            if self.document:
                self.document.close()

    def _process_page(self, page, page_num: int) -> Dict[str, Any]:
        """Extract text and basic information from a page."""
        try:
            # Get page text with formatting and position information
            text_blocks = []
            blocks = page.get_text("dict", flags=11)["blocks"]  # flags=11 preserves layout

            for b in blocks:
                if "lines" in b:  # Text block
                    for line in b["lines"]:
                        for span in line["spans"]:
                            text_blocks.append({
                                'text': span['text'],
                                'font': span['font'],
                                'size': span['size'],
                                'color': span['color'],
                                'bbox': span['bbox'],
                                'block_num': b['number'],
                                'line_num': line['wmode']
                            })

            # Get page dimensions and rotation
            page_info = {
                'page_number': page_num + 1,
                'dimensions': (page.rect.width, page.rect.height),
                'rotation': page.rotation,
                'text_blocks': text_blocks,
                'plain_text': page.get_text("text")
            }

            return page_info

        except Exception as e:
            logger.warning(f"Failed to process page {page_num}: {e}")
            return {'page_number': page_num + 1, 'error': str(e)}

    def _extract_tables(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract tables from a page using PyMuPDF's table functionality."""
        tables = []
        try:
            # This is a simplified version - for production, consider using camelot or tabula
            # for more accurate table extraction
            tabs = page.find_tables()

            if not tabs.tables:
                return []

            for i, table in enumerate(tabs.tables):
                try:
                    rows = table.extract()
                    if rows:
                        tables.append({
                            'page': page_num + 1,
                            'table_num': i + 1,
                            'rows': rows,
                            'bbox': table.bbox,
                            'dimensions': (len(rows), len(rows[0])) if rows else (0, 0)
                        })
                except Exception as e:
                    logger.warning(f"Failed to extract table {i} from page {page_num}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error extracting tables from page {page_num}: {e}")

        return tables

    def _extract_images(self, page, page_num: int) -> List[Dict[str, Any]]:
        """Extract images from a page."""
        images = []
        try:
            image_list = page.get_images(full=True)

            for img_index, img in enumerate(image_list, start=1):
                try:
                    xref = img[0]
                    base_image = self.document.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Generate a unique filename
                    image_name = f"page_{page_num + 1}_img_{img_index}.{base_image['ext']}"
                    image_path = os.path.join(self.images_dir, image_name)

                    # Save the image
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)

                    # Get image dimensions
                    with Image.open(io.BytesIO(image_bytes)) as img_pil:
                        width, height = img_pil.size

                    images.append({
                        'page': page_num + 1,
                        'index': img_index,
                        'path': image_path,
                        'name': image_name,
                        'width': width,
                        'height': height,
                        'format': base_image['ext'],
                        'size': len(image_bytes)
                    })

                except Exception as e:
                    logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
                    continue

        except Exception as e:
            logger.warning(f"Error extracting images from page {page_num}: {e}")

        return images

    def get_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the PDF document."""
        try:
            if self.document is None:
                self.load()

            metadata = super().get_metadata()
            pdf_metadata = self.document.metadata

            # Add PDF-specific metadata
            metadata.update({
                'pdf_version': self.document.pdf_version,
                'is_encrypted': self.document.is_encrypted,
                'permissions': self._get_permissions(),
                'title': pdf_metadata.get('title', ''),
                'author': pdf_metadata.get('author', ''),
                'subject': pdf_metadata.get('subject', ''),
                'keywords': pdf_metadata.get('keywords', ''),
                'creator': pdf_metadata.get('creator', ''),
                'producer': pdf_metadata.get('producer', ''),
                'creation_date': pdf_metadata.get('creationDate', ''),
                'modification_date': pdf_metadata.get('modDate', '')
            })

            return metadata

        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
            return super().get_metadata()

    def _get_permissions(self) -> Dict[str, bool]:
        """Get PDF permissions as a dictionary."""
        if self.document is None or not hasattr(self.document, 'permissions'):
            return {}

        perms = {}
        perm_names = [
            'print', 'modify', 'copy', 'annotate', 'form', 'accessibility',
            'assemble', 'print_highres'
        ]

        for perm in perm_names:
            perms[perm] = bool(getattr(self.document, f'has_{perm}', lambda: False)())

        return perms
