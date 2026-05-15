"""Entity data model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Represents a knowledge graph entity extracted from a document."""

    name: str
    type: str = "Concept"
    properties: dict[str, Any] = Field(default_factory=dict)
    source_doc: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def model_dump_for_neo4j(self) -> dict[str, Any]:
        """Convert to a dict suitable for Neo4j node properties."""
        props = {
            "name": self.name,
            "type": self.type,
            "source_doc": self.source_doc,
            "created_at": self.created_at,
        }
        if self.properties.get("description"):
            props["description"] = self.properties["description"]
        return props

    def neo4j_label(self) -> str:
        """Return the Neo4j label for this entity type."""
        return f"Entity_{self.type}"

    def __hash__(self) -> int:
        return hash((self.name, self.type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.name == other.name and self.type == other.type
