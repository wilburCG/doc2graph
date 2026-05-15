"""Graph query utilities."""

from __future__ import annotations

import logging
from typing import Any

from doc2graph.graph.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphQuery:
    """Common graph query operations."""

    def __init__(self, client: Neo4jClient):
        self.client = client

    def list_entities(self, limit: int = 20) -> list[dict]:
        """List all entities in the graph."""
        return self.client.run_query(
            "MATCH (n) RETURN n.name as name, n.type as type, n.source_doc as source LIMIT $limit",
            {"limit": limit},
        )

    def list_relations(self, limit: int = 20) -> list[dict]:
        """List all relations in the graph."""
        return self.client.run_query(
            "MATCH (a)-[r]->(b) RETURN a.name as source, type(r) as type, b.name as target LIMIT $limit",
            {"limit": limit},
        )

    def find_entity(self, name: str) -> list[dict]:
        """Find an entity by name."""
        return self.client.run_query(
            "MATCH (n {name: $name}) RETURN n.name as name, n.type as type, n.description as description, n.source_doc as source",
            {"name": name},
        )

    def find_related(self, name: str, depth: int = 1) -> list[dict]:
        """Find entities related to a given entity."""
        pattern = "-[*1.." + str(depth) + "]-"
        return self.client.run_query(
            f"MATCH (a {{name: $name}}){pattern}(b) RETURN DISTINCT b.name as name, b.type as type",
            {"name": name},
        )

    def export_json(self) -> dict:
        """Export the entire graph as JSON."""
        entities = self.client.run_query("MATCH (n) RETURN n")
        relations = self.client.run_query("MATCH (a)-[r]->(b) RETURN a.name as source, type(r) as type, b.name as target, r")
        return {
            "entities": entities,
            "relations": relations,
        }

    def export_cypher(self) -> str:
        """Export the graph as Cypher CREATE statements."""
        entities = self.client.run_query("MATCH (n) RETURN n")
        relations = self.client.run_query("MATCH (a)-[r]->(b) RETURN a.name as source, type(r) as type, b.name as target")

        lines = []
        for e in entities:
            node = e.get("n", {})
            label = f"Entity_{node.get('type', 'Concept')}"
            props = ", ".join(f'{k}: "{v}"' for k, v in node.items() if v)
            lines.append(f"CREATE (:{label} {{{props}}});")

        for r in relations:
            src = r.get("source", "")
            tgt = r.get("target", "")
            rel_type = r.get("type", "RELATED_TO")
            lines.append(f'MATCH (a {{name: "{src}"}}), (b {{name: "{tgt}"}}) MERGE (a)-[:{rel_type}]->(b);')

        return "\n".join(lines)
