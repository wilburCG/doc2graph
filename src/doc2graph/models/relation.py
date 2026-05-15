"""Relation data model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Relation(BaseModel):
    """Represents a relationship between two entities."""

    source: str
    target: str
    type: str = "RELATED_TO"
    properties: dict[str, Any] = Field(default_factory=dict)
    source_doc: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def model_dump_for_neo4j(self) -> dict[str, Any]:
        """Convert to a dict suitable for Neo4j relationship properties."""
        props = {
            "source_doc": self.source_doc,
            "created_at": self.created_at,
        }
        if self.properties.get("description"):
            props["description"] = self.properties["description"]
        return props
