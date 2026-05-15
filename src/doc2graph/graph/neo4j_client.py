"""Neo4j database client."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Session

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Manages connection to Neo4j and provides query execution."""

    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self._driver = None

    def connect(self) -> None:
        """Establish connection to Neo4j."""
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )
        # Verify connectivity
        self._driver.verify_connectivity()
        logger.info(f"Connected to Neo4j at {self.uri}")

    def close(self) -> None:
        """Close the connection."""
        if self._driver:
            self._driver.close()
            logger.info("Neo4j connection closed")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for Neo4j sessions."""
        if not self._driver:
            self.connect()
        s = self._driver.session(database=self.database)
        try:
            yield s
        finally:
            s.close()

    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict]:
        """Execute a read query and return results."""
        with self.session() as s:
            result = s.run(query, parameters or {})
            return [record.data() for record in result]

    def run_write(self, query: str, parameters: dict[str, Any] | None = None) -> None:
        """Execute a write query."""
        with self.session() as s:
            s.run(query, parameters or {})

    def clear_all(self) -> None:
        """Delete all nodes and relationships (use with caution!)."""
        self.run_write("MATCH (n) DETACH DELETE n")
        logger.warning("All data cleared from Neo4j")

    def get_stats(self) -> dict:
        """Get basic graph statistics."""
        node_count = self.run_query("MATCH (n) RETURN count(n) as count")
        rel_count = self.run_query("MATCH ()-[r]->() RETURN count(r) as count")
        return {
            "nodes": node_count[0]["count"] if node_count else 0,
            "relationships": rel_count[0]["count"] if rel_count else 0,
        }
