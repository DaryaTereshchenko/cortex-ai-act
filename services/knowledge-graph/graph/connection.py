"""Neo4j connection management with health checks and session helpers."""

from __future__ import annotations

import logging
import os

from neo4j import Driver, GraphDatabase

log = logging.getLogger(__name__)


class Neo4jConnection:
    """Thin wrapper around the Neo4j Python driver."""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "changeme")
        self.database = database or os.getenv("NEO4J_DATABASE", "neo4j")
        self._driver: Driver | None = None

    # -- lifecycle -----------------------------------------------------------

    def connect(self) -> None:
        """Create the driver and verify connectivity."""
        if self._driver is not None:
            return
        self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self._driver.verify_connectivity()
        log.info("Connected to Neo4j at %s (database=%s)", self.uri, self.database)

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None
            log.info("Neo4j connection closed")

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self.connect()
        return self._driver  # type: ignore[return-value]

    # -- helpers -------------------------------------------------------------

    def execute_read(self, query: str, parameters: dict | None = None) -> list[dict]:
        """Run a read transaction and return records as dicts."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(self, query: str, parameters: dict | None = None) -> list[dict]:
        """Run a write transaction and return records as dicts."""
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def is_healthy(self) -> bool:
        """Return True if the driver can reach the server."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception:
            return False
