"""Document metadata model."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Metadata for a processed document."""

    path: str
    name: str
    file_type: str  # pdf, md, txt, docx
    content: str = ""
    source_doc_id: str = ""
    processed_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    entity_count: int = 0
    relation_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_path(cls, file_path: str | Path) -> "Document":
        """Create a Document instance from a file path."""
        p = Path(file_path)
        file_type = p.suffix.lstrip(".").lower()
        # Map common extensions
        ext_map = {
            "pdf": "pdf",
            "md": "md",
            "markdown": "md",
            "txt": "txt",
            "text": "txt",
            "docx": "docx",
        }
        file_type = ext_map.get(file_type, file_type)
        return cls(
            path=str(p),
            name=p.stem,
            file_type=file_type,
            source_doc_id=p.stem,
        )
