"""
SQLite-backed graph store with FTS5 search and semantic vector search.

Schema
------
nodes            — one row per Node
edges            — directed adjacency list
nodes_fts        — FTS5 virtual table on node titles
node_embeddings  — sentence-transformer vectors keyed by node_id
file_hashes      — content-hash manifest for incremental indexing
snapshots        — single-row latest IndexSnapshot
"""


import sqlite3
import struct
import time
from contextlib import contextmanager
from pathlib import Path

from jcode.domain.models import (
    Edge,
    IndexSnapshot,
    Node,
    NodeId,
    NodeType,
)
from jcode.domain.ports import GraphReaderPort, GraphWriterPort

# DDL
_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    node_type   TEXT NOT NULL,
    name        TEXT NOT NULL,
    title       TEXT NOT NULL,
    file_path   TEXT NOT NULL,
    line_start  INTEGER NOT NULL,
    line_end    INTEGER NOT NULL,
    signature   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
    USING fts5(id UNINDEXED, title, content='nodes', content_rowid='rowid');

CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, id, title) VALUES (new.rowid, new.id, new.title);
END;
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title) VALUES ('delete', old.rowid, old.id, old.title);
    INSERT INTO nodes_fts(rowid, id, title) VALUES (new.rowid, new.id, new.title);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title) VALUES ('delete', old.rowid, old.id, old.title);
END;

CREATE TABLE IF NOT EXISTS node_embeddings (
    node_id     TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    vector      BLOB NOT NULL,
    dims        INTEGER NOT NULL,
    embedded_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS file_hashes (
    file_path    TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    indexed_at   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS snapshots (
    id              INTEGER PRIMARY KEY CHECK (id = 1),
    snapshot_hash   TEXT NOT NULL,
    indexed_at      REAL NOT NULL,
    file_count      INTEGER NOT NULL,
    node_count      INTEGER NOT NULL,
    edge_count      INTEGER NOT NULL,
    root_path       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key     TEXT PRIMARY KEY,
    value   TEXT NOT NULL
);
"""
# Helpers
def _row_to_node(row: sqlite3.Row) -> Node:
    return Node(
        id=NodeId(row["id"]),
        node_type=NodeType(row["node_type"]),
        name=row["name"],
        title=row["title"],
        file_path=row["file_path"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        signature=row["signature"],
    )

def _row_to_edge(row: sqlite3.Row) -> Edge:
    return Edge(
        source_id=NodeId(row["source_id"]),
        target_id=NodeId(row["target_id"]),
        edge_type=row["edge_type"],
    )

def _pack_vector(v: list[float]) -> bytes:
    return struct.pack(f"{len(v)}f", *v)

def _unpack_vector(b: bytes, dims: int) -> list[float]:
    return list(struct.unpack(f"{dims}f", b))

def _cosine(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b, strict=False))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    return dot / (na * nb + 1e-9)
# GraphDB
class GraphDB(GraphReaderPort, GraphWriterPort):

    def __init__(self, jcode_dir: Path) -> None:
        self._db_path = jcode_dir / "graph.db"
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
    # GraphWriterPort
    def upsert_node(self, node: Node) -> None:
        sql = """
            INSERT INTO nodes (id, node_type, name, title, file_path, line_start, line_end, signature)
            VALUES (:id, :node_type, :name, :title, :file_path, :line_start, :line_end, :signature)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, signature=excluded.signature, line_end=excluded.line_end
        """
        with self._tx():
            self._conn.execute(sql, {
                "id": node.id.hex, "node_type": node.node_type.value,
                "name": node.name, "title": node.title,
                "file_path": node.file_path, "line_start": node.line_start,
                "line_end": node.line_end, "signature": node.signature,
            })

    def upsert_edge(self, edge: Edge) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type) VALUES (?,?,?)",
                (edge.source_id.hex, edge.target_id.hex, edge.edge_type),
            )

    def save_snapshot(self, snapshot: IndexSnapshot) -> None:
        sql = """
            INSERT INTO snapshots (id,snapshot_hash,indexed_at,file_count,node_count,edge_count,root_path)
            VALUES (1,:snapshot_hash,:indexed_at,:file_count,:node_count,:edge_count,:root_path)
            ON CONFLICT(id) DO UPDATE SET
                snapshot_hash=excluded.snapshot_hash, indexed_at=excluded.indexed_at,
                file_count=excluded.file_count, node_count=excluded.node_count,
                edge_count=excluded.edge_count, root_path=excluded.root_path
        """
        with self._tx():
            self._conn.execute(sql, {
                "snapshot_hash": snapshot.snapshot_hash, "indexed_at": snapshot.indexed_at,
                "file_count": snapshot.file_count, "node_count": snapshot.node_count,
                "edge_count": snapshot.edge_count, "root_path": snapshot.root_path,
            })

    def clear(self) -> None:
        with self._tx():
            self._conn.executescript("""
                DELETE FROM node_embeddings;
                DELETE FROM edges;
                DELETE FROM nodes;
                DELETE FROM file_hashes;
                DELETE FROM snapshots;
                INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild');
            """)
    # Incremental indexing helpers
    def get_file_hashes(self) -> dict[str, str]:
        """Return stored manifest: {rel_file_path → content_hash}."""
        rows = self._conn.execute(
            "SELECT file_path, content_hash FROM file_hashes"
        ).fetchall()
        return {r["file_path"]: r["content_hash"] for r in rows}

    def put_file_hash(self, file_path: str, content_hash: str) -> None:
        """Upsert a file's content hash in the manifest."""
        with self._tx():
            self._conn.execute(
                """INSERT INTO file_hashes (file_path, content_hash, indexed_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(file_path) DO UPDATE SET
                       content_hash=excluded.content_hash,
                       indexed_at=excluded.indexed_at""",
                (file_path, content_hash, time.time()),
            )

    def delete_file_hash(self, file_path: str) -> None:
        """Remove a file from the manifest (deleted or being re-indexed)."""
        with self._tx():
            self._conn.execute(
                "DELETE FROM file_hashes WHERE file_path = ?", (file_path,)
            )

    def delete_nodes_for_file(self, file_path: str) -> None:
        """
        Remove all nodes belonging to file_path, plus:
          - all outgoing edges from those nodes
          - their embeddings (embeddings table has ON DELETE CASCADE,
            but we also delete explicitly for the non-CASCADE path)

        Incoming edges from OTHER files pointing TO these nodes are kept —
        they will become stale only if the callee was renamed/removed, which
        is acceptable for incremental accuracy.
        """
        rows = self._conn.execute(
            "SELECT id FROM nodes WHERE file_path = ?", (file_path,)
        ).fetchall()
        node_ids = [r["id"] for r in rows]
        if not node_ids:
            return

        placeholders = ",".join("?" * len(node_ids))
        with self._tx():
            # Outgoing edges (structure + call edges FROM this file)
            self._conn.execute(
                f"DELETE FROM edges WHERE source_id IN ({placeholders})", node_ids
            )
            # Embeddings (belt-and-suspenders alongside CASCADE)
            self._conn.execute(
                f"DELETE FROM node_embeddings WHERE node_id IN ({placeholders})", node_ids
            )
            # Nodes (FTS triggers fire automatically via nodes_ad)
            self._conn.execute(
                "DELETE FROM nodes WHERE file_path = ?", (file_path,)
            )

    def all_edges(self) -> list[Edge]:
        """Return every edge in the graph — primarily for testing and inspection."""
        rows = self._conn.execute(
            "SELECT source_id, target_id, edge_type FROM edges"
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def count_edges(self) -> int:
        """Total edge count — used for snapshot stats after incremental runs."""
        return self._conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]
    # Embedding write/read
    def put_embedding(self, node_id: NodeId, vector: list[float]) -> None:
        with self._tx():
            self._conn.execute(
                """INSERT INTO node_embeddings (node_id, vector, dims, embedded_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(node_id) DO UPDATE SET
                       vector=excluded.vector, dims=excluded.dims,
                       embedded_at=excluded.embedded_at""",
                (node_id.hex, _pack_vector(vector), len(vector), time.time()),
            )

    def search_semantic(
        self,
        query_vector: list[float],
        limit: int = 10,
        scope: str | None = None,
    ) -> list[tuple[Node, float]]:
        """
        Cosine similarity search over stored embeddings (numpy-accelerated).
        scope: file_path prefix filter, e.g. "comments/" — None means full graph.
        Returns list of (Node, score) sorted by score descending.
        """
        if scope:
            rows = self._conn.execute(
                """SELECT e.node_id, e.vector, e.dims
                   FROM node_embeddings e
                   JOIN nodes n ON e.node_id = n.id
                   WHERE n.file_path LIKE ?""",
                (scope.rstrip("/") + "/%",),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT node_id, vector, dims FROM node_embeddings"
            ).fetchall()

        if not rows:
            return []

        try:
            import numpy as np
            dims = rows[0]["dims"]
            # Stack all vectors into one matrix (N, dims) — single allocation
            matrix = np.frombuffer(
                b"".join(r["vector"] for r in rows), dtype=np.float32
            ).reshape(len(rows), dims)
            q = np.array(query_vector, dtype=np.float32)
            # Batch cosine similarity: dot(matrix, q) / (||row|| * ||q||)
            norms = np.linalg.norm(matrix, axis=1) * (np.linalg.norm(q) + 1e-9) + 1e-9
            scores = (matrix @ q) / norms
            # Top-k via argpartition (O(n)) instead of full sort (O(n log n))
            top_k = min(limit, len(rows))
            if top_k == len(rows):
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argpartition(scores, -top_k)[-top_k:]
                top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            node_ids = [rows[int(i)]["node_id"] for i in top_indices]
            top_scores = scores[top_indices].tolist()
        except ImportError:
            # Pure-Python fallback (no numpy installed)
            scored: list[tuple[float, str]] = []
            for row in rows:
                vec = _unpack_vector(row["vector"], row["dims"])
                scored.append((_cosine(query_vector, vec), row["node_id"]))
            scored.sort(reverse=True)
            node_ids = [nid for _, nid in scored[:limit]]
            top_scores = [s for s, _ in scored[:limit]]

        if not node_ids:
            return []

        # Batch-fetch all result nodes in ONE SQL query instead of N individual calls
        placeholders = ",".join("?" * len(node_ids))
        node_rows = self._conn.execute(
            f"SELECT * FROM nodes WHERE id IN ({placeholders})", node_ids
        ).fetchall()
        node_map = {r["id"]: _row_to_node(r) for r in node_rows}

        return [
            (node_map[nid], float(score))
            for nid, score in zip(node_ids, top_scores)
            if nid in node_map
        ]

    def embeddings_count(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM node_embeddings"
        ).fetchone()[0]
    # GraphReaderPort
    def get_node(self, node_id: NodeId) -> Node | None:
        row = self._conn.execute(
            "SELECT * FROM nodes WHERE id = ?", (node_id.hex,)
        ).fetchone()
        return _row_to_node(row) if row else None

    def search_nodes(
        self,
        query: str,
        limit: int = 20,
        scope: str | None = None,
    ) -> list[Node]:
        """FTS5 search with optional scope (file_path prefix filter)."""
        scope_sql  = "AND n.file_path LIKE ?" if scope else ""
        scope_arg  = (scope.rstrip("/") + "/%",) if scope else ()
        try:
            rows = self._conn.execute(
                f"""SELECT n.* FROM nodes n
                    JOIN nodes_fts f ON n.id = f.id
                    WHERE nodes_fts MATCH ? {scope_sql}
                    ORDER BY rank LIMIT ?""",
                (query, *scope_arg, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            rows = self._conn.execute(
                f"SELECT * FROM nodes WHERE title LIKE ? {scope_sql} LIMIT ?",
                (f"%{query}%", *scope_arg, limit),
            ).fetchall()
        return [_row_to_node(r) for r in rows]

    def successors(self, node_id: NodeId) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE source_id = ?", (node_id.hex,)
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def predecessors(self, node_id: NodeId) -> list[Edge]:
        rows = self._conn.execute(
            "SELECT * FROM edges WHERE target_id = ?", (node_id.hex,)
        ).fetchall()
        return [_row_to_edge(r) for r in rows]

    def all_nodes(self) -> list[Node]:
        rows = self._conn.execute("SELECT * FROM nodes").fetchall()
        return [_row_to_node(r) for r in rows]

    def get_snapshot(self) -> IndexSnapshot | None:
        row = self._conn.execute(
            "SELECT * FROM snapshots WHERE id = 1"
        ).fetchone()
        if row is None:
            return None
        return IndexSnapshot(
            snapshot_hash=row["snapshot_hash"],
            indexed_at=row["indexed_at"],
            file_count=row["file_count"],
            node_count=row["node_count"],
            edge_count=row["edge_count"],
            root_path=row["root_path"],
        )
