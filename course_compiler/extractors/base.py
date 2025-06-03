"""
Base extractor class that defines the interface for all file extractors.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

class BaseExtractor(ABC):
    """Abstract base class for all file extractors."""

    def __init__(self, file_path: Path, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor with a file path and optional configuration.

        Args:
            file_path: Path to the file to extract content from
            config: Optional configuration dictionary
        """
        self.file_path = Path(file_path)
        self.config = config or {}
        self._validate_file()

    def _validate_file(self) -> None:
        """Validate that the file exists and is accessible."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")
        if not self.file_path.is_file():
            raise IsADirectoryError(f"Path is a directory, not a file: {self.file_path}")

    @abstractmethod
    def extract(self) -> Dict[str, Any]:
        """
        Extract content from the file.

        Returns:
            Dictionary containing the extracted content and metadata

        Raises:
            ExtractionError: If there's an error during extraction
        """
        pass

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get basic file metadata.

        Returns:
            Dictionary containing file metadata
        """
        stat = self.file_path.stat()
        return {
            "file_name": self.file_path.name,
            "file_size": stat.st_size,
            "last_modified": stat.st_mtime,
            "created": stat.st_ctime,
            "absolute_path": str(self.file_path.resolve())
        }

    def __str__(self) -> str:
        """String representation of the extractor."""
        return f"{self.__class__.__name__}(file_path={self.file_path!r})"


class ExtractionError(Exception):
    """Base exception for extraction errors."""
    pass
