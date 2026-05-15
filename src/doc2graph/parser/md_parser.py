"""Markdown document parser."""

from __future__ import annotations

from pathlib import Path

from doc2graph.parser.base import BaseParser


class MdParser(BaseParser):
    """Parse Markdown files, preserving structure as plain text."""

    def parse(self, file_path: str | Path) -> str:
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Markdown file not found: {p}")
        return p.read_text(encoding="utf-8")

    @classmethod
    def supported_extensions(cls) -> list[str]:
        return [".md", ".markdown"]
