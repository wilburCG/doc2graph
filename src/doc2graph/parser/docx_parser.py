"""DOCX (Word) document parser."""

from __future__ import annotations

from pathlib import Path

from doc2graph.parser.base import BaseParser


class DocxParser(BaseParser):
    """Parse Microsoft Word (.docx) files using python-docx."""

    def parse(self, file_path: str | Path) -> str:
        from docx import Document as DocxDocument

        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"DOCX file not found: {p}")

        doc = DocxDocument(str(p))
        paragraphs = []
        for para in doc.paragraphs:
            if para.text.strip():
                paragraphs.append(para.text)
        return "\n\n".join(paragraphs)

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return [".docx"]
