"""Graph merge — entity resolution and relationship fusion across documents."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from doc2graph.graph.neo4j_client import Neo4jClient
from doc2graph.models.entity import Entity
from doc2graph.models.relation import Relation

logger = logging.getLogger(__name__)


class GraphMerger:
    """Merge entities and relations from multiple documents, handling duplicates."""

    def __init__(self, client: Neo4jClient, strategy: str = "name_type", threshold: float = 0.85):
        self.client = client
        self.strategy = strategy
        self.threshold = threshold

    def merge_entities(self, entities: list[Entity]) -> list[Entity]:
        """Deduplicate entities based on strategy.

        - name_type: same name + same type → merge
        - fuzzy: fuzzy string match on name
        """
        if self.strategy == "fuzzy":
            return self._fuzzy_merge(entities)
        return self._name_type_merge(entities)

    def merge_relations(self, relations: list[Relation]) -> list[Relation]:
        """Deduplicate relations. Same source/target/type → keep all source markers."""
        seen: dict[tuple, Relation] = {}
        for rel in relations:
            key = (rel.source, rel.target, rel.type)
            if key in seen:
                # Merge properties: append source docs
                existing = seen[key]
                if rel.source_doc and rel.source_doc not in existing.source_doc:
                    existing.source_doc = f"{existing.source_doc}, {rel.source_doc}"
            else:
                seen[key] = rel
        return list(seen.values())

    def merge_all(
        self,
        all_entities: list[Entity],
        all_relations: list[Relation],
    ) -> tuple[list[Entity], list[Relation]]:
        """Merge a complete set of entities and relations."""
        merged_entities = self.merge_entities(all_entities)
        merged_relations = self.merge_relations(all_relations)
        logger.info(f"Merged {len(all_entities)}→{len(merged_entities)} entities, "
                     f"{len(all_relations)}→{len(merged_relations)} relations")
        return merged_entities, merged_relations

    def merge_in_neo4j(self) -> dict[str, int]:
        """Perform entity resolution directly in Neo4j using Cypher.

        Finds entities with same name and type across different source documents
        and merges them.
        """
        # Merge entities with same name and type
        merge_query = """
        MATCH (n)
        WHERE n.name IS NOT NULL AND n.type IS NOT NULL
        WITH n.type as type, n.name as name, collect(n) as nodes
        WHERE size(nodes) > 1
        UNWIND nodes[1..] as duplicate
        WITH nodes[0] as keep, duplicate
        // Copy properties
        SET keep.source_doc = keep.source_doc + ", " + duplicate.source_doc
        // Redirect relationships
        MATCH (duplicate)-[r]->(other)
        WHERE id(other) <> id(keep)
        MERGE (keep)-[new_r:RELATED_TO]->(other)
        SET new_r += properties(r)
        DELETE r, duplicate
        RETURN count(duplicate) as merged
        """
        result = self.client.run_write(merge_query)
        return {"merged": 0}

    def _name_type_merge(self, entities: list[Entity]) -> list[Entity]:
        """Merge entities with identical name and type."""
        groups: dict[tuple, list[Entity]] = defaultdict(list)
        for e in entities:
            key = (e.name, e.type)
            groups[key].append(e)

        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Merge into first entity
                base = group[0]
                source_docs = set()
                for e in group:
                    source_docs.add(e.source_doc)
                    # Merge properties (keep non-empty values)
                    for k, v in e.properties.items():
                        if v and k not in base.properties:
                            base.properties[k] = v
                base.source_doc = ", ".join(sorted(source_docs))
                merged.append(base)

        return merged

    def _fuzzy_merge(self, entities: list[Entity]) -> list[Entity]:
        """Simple fuzzy merge using name similarity."""
        # Group by type first
        by_type: dict[str, list[Entity]] = defaultdict(list)
        for e in entities:
            by_type[e.type].append(e)

        merged = []
        for entity_type, group in by_type.items():
            merged.extend(self._fuzzy_merge_group(group))

        return merged

    def _fuzzy_merge_group(self, entities: list[Entity]) -> list[Entity]:
        """Fuzzy merge within a single entity type group."""
        merged: list[Entity] = []
        used = set()

        for i, e1 in enumerate(entities):
            if i in used:
                continue
            merged_entity = e1
            source_docs = {e1.source_doc}

            for j, e2 in enumerate(entities[i + 1:], i + 1):
                if j in used:
                    continue
                if self._similarity(e1.name, e2.name) >= self.threshold:
                    used.add(j)
                    source_docs.add(e2.source_doc)
                    for k, v in e2.properties.items():
                        if v and k not in merged_entity.properties:
                            merged_entity.properties[k] = v

            merged_entity.source_doc = ", ".join(sorted(source_docs))
            merged.append(merged_entity)

        return merged

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Simple character-level Jaccard similarity."""
        if not a or not b:
            return 0.0
        a_lower = a.lower()
        b_lower = b.lower()
        if a_lower == b_lower:
            return 1.0
        set_a = set(a_lower)
        set_b = set(b_lower)
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0
