"""Base parser interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, file_path: str | Path) -> str:
        """Parse a document file and return its text content.

        Args:
            file_path: Path to the document file.

        Returns:
            The extracted text content.
        """
        ...

    @classmethod
    def supports(cls, file_path: str | Path) -> bool:
        """Check if this parser supports the given file."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.supported_extensions()

    @classmethod
    @abstractmethod
    def supported_extensions(cls) -> list[str]:
        """Return list of supported file extensions (e.g. ['.pdf'])."""
        ...
