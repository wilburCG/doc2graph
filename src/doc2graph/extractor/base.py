"""Base extractor interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from doc2graph.models.entity import Entity
from doc2graph.models.relation import Relation


class BaseExtractor(ABC):
    """Abstract base class for entity/relation extractors."""

    @abstractmethod
    def extract(self, text: str, source_doc: str = "") -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from text.

        Args:
            text: The document text to extract from.
            source_doc: Source document identifier.

        Returns:
            Tuple of (entities, relations).
        """
        ...
