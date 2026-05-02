"""
Ports (interfaces) — Dependency Inversion Principle.

High-level modules (Builder, Traversal, MCP) depend on these abstractions,
not on concrete storage or parser implementations.  This keeps the domain
layer free of infrastructure concerns and makes every component independently
testable via simple fakes.
"""


from typing import Protocol, runtime_checkable

from jcode.domain.models import (
    BlastRadiusResult,
    Edge,
    IndexSnapshot,
    Node,
    NodeId,
    TraversalResult,
)

# Storage ports
@runtime_checkable
class ObjectStorePort(Protocol):
    """Write and read content-addressable node objects (git-blob style)."""

    def put(self, node: Node) -> NodeId:
        """Serialise *node* and store it; return its deterministic NodeId."""
        ...

    def get(self, node_id: NodeId) -> Node:
        """Retrieve and deserialise a node by id. Raises KeyError if absent."""
        ...

    def exists(self, node_id: NodeId) -> bool:
        """Return True if the object is already stored (avoids redundant writes)."""
        ...

@runtime_checkable
class GraphReaderPort(Protocol):
    """Read-only view of the graph — used by traversal and MCP."""

    def get_node(self, node_id: NodeId) -> Node | None: ...

    def search_nodes(self, query: str, limit: int = 20) -> list[Node]:
        """Full-text search over node titles."""
        ...

    def successors(self, node_id: NodeId) -> list[Edge]:
        """Return all edges where *node_id* is the source (forward edges)."""
        ...

    def predecessors(self, node_id: NodeId) -> list[Edge]:
        """Return all edges where *node_id* is the target (reverse edges)."""
        ...

    def all_nodes(self) -> list[Node]: ...

    def get_snapshot(self) -> IndexSnapshot | None: ...

@runtime_checkable
class GraphWriterPort(Protocol):
    """Write side — used only by the Builder."""

    def upsert_node(self, node: Node) -> None: ...

    def upsert_edge(self, edge: Edge) -> None: ...

    def save_snapshot(self, snapshot: IndexSnapshot) -> None: ...

    def clear(self) -> None:
        """Drop all data — used before a full re-index."""
        ...
    # Incremental indexing
    def get_file_hashes(self) -> dict[str, str]:
        """Return stored manifest: {rel_file_path -> content_hash}."""
        ...

    def put_file_hash(self, file_path: str, content_hash: str) -> None:
        """Upsert a file's content hash in the manifest."""
        ...

    def delete_file_hash(self, file_path: str) -> None:
        """Remove a file from the manifest."""
        ...

    def delete_nodes_for_file(self, file_path: str) -> None:
        """Remove all nodes (and their outgoing edges) for the given file."""
        ...

    def count_edges(self) -> int:
        """Total edge count — for snapshot stats."""
        ...
# Parser port
@runtime_checkable
class LanguageParserPort(Protocol):
    """
    Parses a single source file and emits (nodes, edges).

    Open/Closed: add Go, TypeScript etc. by implementing this protocol —
    the Builder never needs to change.
    """

    @property
    def language(self) -> str:
        """e.g. 'python', 'typescript'"""
        ...

    @property
    def file_extensions(self) -> frozenset[str]:
        """e.g. frozenset({'.py'})"""
        ...

    def parse_file(
        self,
        file_path: str,
        source: str,
        repo_root: str,
    ) -> tuple[list[Node], list[Edge]]:
        """
        Parse *source* and return all nodes and edges found.
        *file_path* and *repo_root* are used to build relative paths / titles.
        """
        ...
# Traversal port
@runtime_checkable
class TraversalPort(Protocol):
    """Graph traversal operations — consumed by the MCP tool layer."""

    def context(
        self,
        entry_node_id: NodeId,
        max_depth: int = 3,
    ) -> TraversalResult:
        """DFS forward from *entry_node_id* — gather surrounding context."""
        ...

    def blast_radius(
        self,
        changed_node_id: NodeId,
        max_depth: int = 5,
    ) -> BlastRadiusResult:
        """BFS reverse from *changed_node_id* — find everything that could break."""
        ...
