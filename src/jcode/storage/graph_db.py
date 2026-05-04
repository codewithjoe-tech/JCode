"""
SQLite-backed graph store with FTS5 search and semantic vector search.

Schema
------
nodes            — one row per Node
edges            — directed adjacency list
nodes_fts        — FTS5 virtual table on node titles + keywords
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
    signature   TEXT NOT NULL DEFAULT '',
    keywords    TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edges (
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    PRIMARY KEY (source_id, target_id, edge_type)
);

CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts
    USING fts5(id UNINDEXED, title, keywords, content='nodes', content_rowid='rowid');

CREATE TRIGGER IF NOT EXISTS nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, id, title, keywords) VALUES (new.rowid, new.id, new.title, new.keywords);
END;
CREATE TRIGGER IF NOT EXISTS nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title, keywords) VALUES ('delete', old.rowid, old.id, old.title, old.keywords);
    INSERT INTO nodes_fts(rowid, id, title, keywords) VALUES (new.rowid, new.id, new.title, new.keywords);
END;
CREATE TRIGGER IF NOT EXISTS nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title, keywords) VALUES ('delete', old.rowid, old.id, old.title, old.keywords);
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

_MIGRATION_KEYWORDS = """
-- Add keywords column to existing databases that predate this field
ALTER TABLE nodes ADD COLUMN keywords TEXT NOT NULL DEFAULT '';
"""

_MIGRATION_FTS_KEYWORDS = """
-- Rebuild FTS table to include keywords column
DROP TRIGGER IF EXISTS nodes_ai;
DROP TRIGGER IF EXISTS nodes_au;
DROP TRIGGER IF EXISTS nodes_ad;
DROP TABLE IF EXISTS nodes_fts;
CREATE VIRTUAL TABLE nodes_fts
    USING fts5(id UNINDEXED, title, keywords, content='nodes', content_rowid='rowid');
CREATE TRIGGER nodes_ai AFTER INSERT ON nodes BEGIN
    INSERT INTO nodes_fts(rowid, id, title, keywords) VALUES (new.rowid, new.id, new.title, new.keywords);
END;
CREATE TRIGGER nodes_au AFTER UPDATE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title, keywords) VALUES ('delete', old.rowid, old.id, old.title, old.keywords);
    INSERT INTO nodes_fts(rowid, id, title, keywords) VALUES (new.rowid, new.id, new.title, new.keywords);
END;
CREATE TRIGGER nodes_ad AFTER DELETE ON nodes BEGIN
    INSERT INTO nodes_fts(nodes_fts, rowid, id, title, keywords) VALUES ('delete', old.rowid, old.id, old.title, old.keywords);
END;
INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild');
"""
# Helpers
def _row_to_node(row: sqlite3.Row) -> Node:
    keys = row.keys()
    return Node(
        id=NodeId(row["id"]),
        node_type=NodeType(row["node_type"]),
        name=row["name"],
        title=row["title"],
        file_path=row["file_path"],
        line_start=row["line_start"],
        line_end=row["line_end"],
        signature=row["signature"],
        keywords=row["keywords"] if "keywords" in keys else "",
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
        self._migrate()

    def _migrate(self) -> None:
        """
        Apply schema migrations idempotently using a schema_version key in meta.

        Version history:
          1 — baseline (no keywords)
          2 — added keywords column + FTS rebuild

        The FTS rebuild (version 2) is expensive on large repos — it scans all
        nodes and rewrites the B-tree index. We gate it behind a version check
        so it only runs once, not on every DB open.
        """
        # Read current schema version (missing key → version 0)
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'schema_version'"
        ).fetchone()
        version = int(row["value"]) if row else 0

        if version >= 2:
            return  # already up to date — fast path, no work done

        # v1 → v2: add keywords column + rebuild FTS to include it
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(nodes)").fetchall()}
        if "keywords" not in cols:
            try:
                self._conn.execute(
                    "ALTER TABLE nodes ADD COLUMN keywords TEXT NOT NULL DEFAULT ''"
                )
                self._conn.commit()
            except sqlite3.OperationalError:
                pass  # already exists — concurrent open

        fts_cols: set[str] = set()
        try:
            fts_cols = {r[2] for r in self._conn.execute("PRAGMA table_info(nodes_fts)").fetchall()}
        except sqlite3.OperationalError:
            pass

        if "keywords" not in fts_cols:
            self._conn.executescript(_MIGRATION_FTS_KEYWORDS)

        # Record new version — skip migration on all future opens
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES ('schema_version', '2') "
            "ON CONFLICT(key) DO UPDATE SET value = '2'"
        )
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
            INSERT INTO nodes (id, node_type, name, title, file_path, line_start, line_end, signature, keywords)
            VALUES (:id, :node_type, :name, :title, :file_path, :line_start, :line_end, :signature, :keywords)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, signature=excluded.signature,
                line_end=excluded.line_end, keywords=excluded.keywords
        """
        with self._tx():
            self._conn.execute(sql, {
                "id": node.id.hex, "node_type": node.node_type.value,
                "name": node.name, "title": node.title,
                "file_path": node.file_path, "line_start": node.line_start,
                "line_end": node.line_end, "signature": node.signature,
                "keywords": node.keywords,
            })

    def upsert_edge(self, edge: Edge) -> None:
        with self._tx():
            self._conn.execute(
                "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type) VALUES (?,?,?)",
                (edge.source_id.hex, edge.target_id.hex, edge.edge_type),
            )

    def bulk_upsert_nodes(self, nodes: list[Node]) -> None:
        """Insert/update all nodes in a single transaction — O(1) commits."""
        if not nodes:
            return
        sql = """
            INSERT INTO nodes (id, node_type, name, title, file_path, line_start, line_end, signature, keywords)
            VALUES (:id, :node_type, :name, :title, :file_path, :line_start, :line_end, :signature, :keywords)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, signature=excluded.signature,
                line_end=excluded.line_end, keywords=excluded.keywords
        """
        with self._tx():
            self._conn.executemany(sql, [
                {
                    "id": n.id.hex, "node_type": n.node_type.value,
                    "name": n.name, "title": n.title,
                    "file_path": n.file_path, "line_start": n.line_start,
                    "line_end": n.line_end, "signature": n.signature,
                    "keywords": n.keywords,
                }
                for n in nodes
            ])

    def bulk_update_keywords(self, items: list[tuple[NodeId, str]]) -> None:
        """
        Update the keywords column for a batch of nodes and refresh FTS.
        items: list of (node_id, keywords_string) pairs.
        keywords_string is space-separated tokens, e.g. "mymoneybazaar.com prefr.in".
        """
        if not items:
            return
        with self._tx():
            self._conn.executemany(
                "UPDATE nodes SET keywords = ? WHERE id = ?",
                [(kw, nid.hex) for nid, kw in items],
            )
            # Rebuild FTS so the new keywords are immediately searchable
            self._conn.execute("INSERT INTO nodes_fts(nodes_fts) VALUES ('rebuild')")

    def bulk_upsert_edges(self, edges: list[Edge]) -> None:
        """Insert all edges in a single transaction — O(1) commits."""
        if not edges:
            return
        with self._tx():
            self._conn.executemany(
                "INSERT OR IGNORE INTO edges (source_id, target_id, edge_type) VALUES (?,?,?)",
                [(e.source_id.hex, e.target_id.hex, e.edge_type) for e in edges],
            )

    def bulk_put_file_hashes(self, hashes: dict[str, str]) -> None:
        """Upsert all file hashes in a single transaction."""
        if not hashes:
            return
        now = time.time()
        with self._tx():
            self._conn.executemany(
                """INSERT INTO file_hashes (file_path, content_hash, indexed_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(file_path) DO UPDATE SET
                       content_hash=excluded.content_hash,
                       indexed_at=excluded.indexed_at""",
                [(fp, ch, now) for fp, ch in hashes.items()],
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

    def bulk_put_embeddings(self, items: list[tuple[NodeId, list[float]]]) -> None:
        """Upsert all embeddings in a single transaction — O(1) commits."""
        if not items:
            return
        now = time.time()
        with self._tx():
            self._conn.executemany(
                """INSERT INTO node_embeddings (node_id, vector, dims, embedded_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(node_id) DO UPDATE SET
                       vector=excluded.vector, dims=excluded.dims,
                       embedded_at=excluded.embedded_at""",
                [(nid.hex, _pack_vector(vec), len(vec), now) for nid, vec in items],
            )

    def all_successors(self) -> dict:
        """Return {source_id → [Edge]} for the entire graph in one query."""
        rows = self._conn.execute(
            "SELECT source_id, target_id, edge_type FROM edges"
        ).fetchall()
        result: dict = {}
        for r in rows:
            result.setdefault(r["source_id"], []).append(_row_to_edge(r))
        return result

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
        """FTS5 search over node titles and keywords, with optional scope filter."""
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
