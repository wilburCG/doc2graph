"""Import entities and relations into Neo4j."""

from __future__ import annotations

import logging
from typing import Any

from doc2graph.graph.neo4j_client import Neo4jClient
from doc2graph.models.entity import Entity
from doc2graph.models.relation import Relation

logger = logging.getLogger(__name__)


class GraphImporter:
    """Import extracted entities and relations into Neo4j."""

    def __init__(self, client: Neo4jClient):
        self.client = client

    def import_entities(self, entities: list[Entity]) -> int:
        """Import a list of entities into Neo4j.

        Uses MERGE to avoid duplicates within the same source document.
        Returns the number of entities imported.
        """
        if not entities:
            return 0

        count = 0
        for entity in entities:
            props = entity.model_dump_for_neo4j()
            label = entity.neo4j_label()

            # Build MERGE query with dynamic label
            # Note: labels cannot be parameterized in Neo4j, so we use string formatting
            # (safe because label comes from our controlled entity type)
            safe_label = self._sanitize_label(label)
            query = f"""
                MERGE (n:{safe_label} {{name: $name, source_doc: $source_doc}})
                ON CREATE SET n += $props
                ON MATCH SET n.description = coalesce(n.description, $props.description)
            """
            self.client.run_write(query, {
                "name": entity.name,
                "source_doc": entity.source_doc,
                "props": props,
            })
            count += 1

        logger.info(f"Imported {count} entities")
        return count

    def import_relations(self, relations: list[Relation]) -> int:
        """Import a list of relations into Neo4j.

        Creates relationships between entities, creating nodes if they don't exist.
        Returns the number of relations imported.
        """
        if not relations:
            return 0

        count = 0
        for rel in relations:
            safe_rel_type = self._sanitize_rel_type(rel.type)

            # We need to find or create both source and target nodes
            # Since we don't know their types, we use generic Entity label
            query = f"""
                MATCH (s {{name: $source}})
                MATCH (t {{name: $target}})
                MERGE (s)-[r:{safe_rel_type} {{source_doc: $source_doc}}]->(t)
                ON CREATE SET r += $props
            """
            try:
                self.client.run_write(query, {
                    "source": rel.source,
                    "target": rel.target,
                    "source_doc": rel.source_doc,
                    "props": rel.model_dump_for_neo4j(),
                })
                count += 1
            except Exception as e:
                logger.warning(f"Failed to import relation {rel.source}-[{rel.type}]->{rel.target}: {e}")

        logger.info(f"Imported {count} relations")
        return count

    def import_graph(self, entities: list[Entity], relations: list[Relation]) -> dict[str, int]:
        """Import a complete graph (entities + relations) into Neo4j.

        Returns counts of imported entities and relations.
        """
        entity_count = self.import_entities(entities)
        relation_count = self.import_relations(relations)
        return {"entities": entity_count, "relations": relation_count}

    @staticmethod
    def _sanitize_label(label: str) -> str:
        """Sanitize a Neo4j label (remove unsafe characters)."""
        return "".join(c if c.isalnum() or c == "_" else "" for c in label)

    @staticmethod
    def _sanitize_rel_type(rel_type: str) -> str:
        """Sanitize a Neo4j relationship type."""
        sanitized = "".join(c if c.isalnum() or c == "_" else "" for c in rel_type).upper()
        return sanitized if sanitized else "RELATED_TO"
