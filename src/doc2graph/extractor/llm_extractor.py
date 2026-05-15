"""LLM-based entity and relation extractor using OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

from doc2graph.extractor.base import BaseExtractor
from doc2graph.extractor.prompt import EXTRACTION_SYSTEM_PROMPT, EXTRACTION_USER_TEMPLATE
from doc2graph.models.entity import Entity
from doc2graph.models.relation import Relation

logger = logging.getLogger(__name__)


class LlmExtractor(BaseExtractor):
    """Extract entities and relations using an OpenAI-compatible LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        api_base: str = "https://api.openai.com/v1",
        max_entities: int = 50,
        max_relations: int = 100,
        temperature: float = 0.1,
    ):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.temperature = temperature

    def extract(self, text: str, source_doc: str = "") -> tuple[list[Entity], list[Relation]]:
        """Extract entities and relations from text using LLM."""
        prompt = EXTRACTION_USER_TEMPLATE.format(
            text=text[:8000],  # Truncate to avoid token limit issues
            max_entities=self.max_entities,
            max_relations=self.max_relations,
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or ""
        return self._parse_response(content, source_doc)

    def _parse_response(self, content: str, source_doc: str) -> tuple[list[Entity], list[Relation]]:
        """Parse LLM JSON response into Entity and Relation objects."""
        try:
            # Strip markdown code fences if present
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()

            data = json.loads(content)
        except (json.JSONDecodeError, IndexError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response: {content[:500]}")
            return [], []

        entities = []
        for e in data.get("entities", []):
            entities.append(Entity(
                name=e.get("name", ""),
                type=e.get("type", "Concept"),
                properties={"description": e.get("description", "")},
                source_doc=source_doc,
            ))

        relations = []
        for r in data.get("relations", []):
            relations.append(Relation(
                source=r.get("source", ""),
                target=r.get("target", ""),
                type=r.get("type", "RELATED_TO"),
                properties={"description": r.get("description", "")},
                source_doc=source_doc,
            ))

        logger.info(f"Extracted {len(entities)} entities and {len(relations)} relations from '{source_doc}'")
        return entities, relations
