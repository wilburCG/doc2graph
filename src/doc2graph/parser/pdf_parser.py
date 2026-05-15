"""PDF document parser."""

from __future__ import annotations

from pathlib import Path

from doc2graph.parser.base import BaseParser


class PdfParser(BaseParser):
    """Parse PDF files using PyMuPDF (fitz) with PyPDF2 fallback."""

    def parse(self, file_path: str | Path) -> str:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"PDF file not found: {p}")

        # Try PyMuPDF first (faster, better quality)
        try:
            return self._parse_with_mupdf(p)
        except ImportError:
            pass

        # Fallback to PyPDF2
        return self._parse_with_pypdf2(p)

    def _parse_with_mupdf(self, p: Path) -> str:
        import fitz  # pymupdf

        doc = fitz.open(str(p))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        return "\n\n".join(pages)

    def _parse_with_pypdf2(self, p: Path) -> str:
        from PyPDF2 import PdfReader

        reader = PdfReader(str(p))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return [".pdf"]
