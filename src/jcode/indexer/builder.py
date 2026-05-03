"""Indexer / Builder — orchestrates a full or incremental index run."""

import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jcode.domain.models import Edge, IndexSnapshot, Node, NodeId, NodeType
from jcode.domain.ports import GraphWriterPort, LanguageParserPort, ObjectStorePort


class Indexer:
    """Coordinates the full index pipeline. Depends only on ports (DIP)."""

    def __init__(
        self,
        parser: LanguageParserPort,
        object_store: ObjectStorePort,
        graph_writer: GraphWriterPort,
    ) -> None:
        self._parser = parser
        self._store = object_store
        self._graph = graph_writer
    # Public entry point
    def index(self, repo_root: str, *, full_reindex: bool = False,
              workers: int = 4) -> IndexSnapshot:
        """
        Index *repo_root*.

        full_reindex=True  — wipe everything and rebuild from scratch.
                             Equivalent to clearing then running incremental
                             with an empty manifest (all files look "new").
        full_reindex=False — incremental: only parse files whose content hash
                             has changed since the last run.
        """
        if full_reindex:
            self._graph.clear()
        # Both paths funnel through _incremental.
        # After clear(), the manifest is empty so every file is "new" —
        # that gives us a correct full rebuild while still writing the manifest.
        return self._incremental(repo_root, workers=workers)
    # Incremental core
    def _incremental(self, repo_root: str, *, workers: int = 4) -> IndexSnapshot:
        import click

        # 1. Scan current files → sha256(content)
        current: dict[str, str] = {}   # rel_path → hash
        abs_map:  dict[str, str] = {}   # rel_path → abs_path

        print("  scanning files …", end="\r", flush=True)
        for abs_path in self._discover_files(repo_root):
            try:
                raw = Path(abs_path).read_bytes()
            except OSError:
                continue
            rel = os.path.relpath(abs_path, repo_root)
            current[rel] = hashlib.sha256(raw).hexdigest()
            abs_map[rel]  = abs_path
        # clear the \r line
        print(f"  scanned {len(current)} files              ")

        # 2. Load stored manifest
        stored: dict[str, str] = self._graph.get_file_hashes()

        # 3. Categorise
        new_files     = {p for p in current if p not in stored}
        changed_files = {p for p in current if p in stored and current[p] != stored[p]}
        deleted_files = {p for p in stored  if p not in current}

        # 4. Remove stale data for changed/deleted files
        for rel in changed_files | deleted_files:
            self._graph.delete_nodes_for_file(rel)
            self._graph.delete_file_hash(rel)

        # 5. Short-circuit if nothing needs parsing
        to_parse = new_files | changed_files
        if not to_parse and not deleted_files:
            snap = self._graph.get_snapshot()
            if snap:
                click.echo("  nothing changed — index is up to date")
                return snap

        # 6. Load existing nodes for cross-file call resolution
        existing_nodes: dict[NodeId, Node] = {n.id: n for n in self._graph.all_nodes()}

        # 7. Parse new / changed files
        abs_to_parse = {abs_map[r] for r in to_parse if r in abs_map}
        snapshot = self._parse_and_persist(repo_root, abs_to_parse, existing_nodes,
                                           workers=workers)

        # 8. Update manifest (single transaction)
        self._graph.bulk_put_file_hashes(
            {rel: current[rel] for rel in to_parse if rel in current}
        )

        if to_parse or deleted_files:
            added   = len(new_files)
            changed = len(changed_files)
            deleted = len(deleted_files)
            if added or changed or deleted:
                click.echo(f"  incremental: +{added} new, ~{changed} changed, -{deleted} deleted")

        return snapshot
    # Parsing + persistence (shared by every run)
    def _parse_and_persist(
        self,
        repo_root: str,
        files_to_parse: set[str],
        existing_nodes: dict[NodeId, Node],
        *,
        workers: int = 4,
    ) -> IndexSnapshot:
        import click

        new_nodes: dict[NodeId, Node] = {}
        provisional_nodes: dict[NodeId, Node] = {}
        all_edges: list[Edge] = []

        files_list = sorted(files_to_parse)
        total = len(files_list)

        def _parse_one(abs_path: str) -> tuple[list[Node], list[Edge]]:
            try:
                source = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                return [], []
            return self._parser.parse_file(abs_path, source, repo_root)

        label = f"  Parsing {total} file{'s' if total != 1 else ''}"
        effective_workers = min(workers, total) if total else 1

        with click.progressbar(
            length=total,
            label=label,
            show_eta=True,
            show_percent=True,
            bar_template="%(label)s  %(bar)s  %(info)s",
            width=40,
        ) as bar:
            if effective_workers > 1:
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    futures = {executor.submit(_parse_one, f): f for f in files_list}
                    for future in as_completed(futures):
                        nodes, edges = future.result()
                        for node in nodes:
                            if node.file_path == "<unresolved>":
                                provisional_nodes[node.id] = node
                            else:
                                new_nodes[node.id] = node
                        all_edges.extend(edges)
                        bar.update(1)
            else:
                for abs_path in files_list:
                    nodes, edges = _parse_one(abs_path)
                    for node in nodes:
                        if node.file_path == "<unresolved>":
                            provisional_nodes[node.id] = node
                        else:
                            new_nodes[node.id] = node
                    all_edges.extend(edges)
                    bar.update(1)

        # Resolve calls against ALL known nodes (existing + newly parsed)
        t0 = time.time()
        click.echo(f"  resolving {len(all_edges):,} edges …", nl=False)
        all_known = {**existing_nodes, **new_nodes}
        resolved_edges, persisted_provisionals = self._resolve_calls(
            all_known, provisional_nodes, all_edges
        )
        click.echo(f"  done ({time.time() - t0:.1f}s)")

        # Build valid edge set
        all_known_extended = {**all_known, **persisted_provisionals}
        valid_edges = [
            e for e in resolved_edges
            if e.source_id in all_known_extended and e.target_id in all_known_extended
        ]
        all_real_nodes = list(new_nodes.values())
        n_nodes = len(all_real_nodes) + len(persisted_provisionals)
        n_edges = len(valid_edges)

        # Object store blobs (content-addressable, skips if already exists)
        t_blobs = time.time()
        with click.progressbar(
            all_real_nodes,
            label=f"  Blobs  {n_nodes:,} nodes",
            show_eta=True,
            show_percent=True,
            bar_template="%(label)s  %(bar)s  %(info)s",
            width=40,
        ) as bar:
            for node in bar:
                self._store.put(node)
        click.echo(f"  blobs done ({time.time() - t_blobs:.1f}s)")

        # Single-transaction bulk DB writes — O(1) commits regardless of node count
        click.echo(f"  DB    {n_nodes:,} nodes + {n_edges:,} edges …", nl=False)
        t_db = time.time()
        self._graph.bulk_upsert_nodes(all_real_nodes)
        self._graph.bulk_upsert_nodes(list(persisted_provisionals.values()))
        self._graph.bulk_upsert_edges(valid_edges)
        click.echo(f"  done ({time.time() - t_db:.1f}s)")

        snapshot = self._make_snapshot(repo_root)
        self._graph.save_snapshot(snapshot)

        # Embed new nodes — silently skipped if sentence-transformers not installed
        try:
            from jcode.indexer.embedder import embed_graph
            click.echo("  embedding nodes …", nl=False)
            t2 = time.time()
            embedded = embed_graph(self._graph)
            if embedded:
                click.echo(f"  {embedded:,} nodes embedded ({time.time() - t2:.1f}s)")
            else:
                click.echo(" skipped")
        except Exception:
            click.echo(" skipped")

        return snapshot
    # Helpers
    def _discover_files(self, repo_root: str):
        exts = self._parser.file_extensions
        skip = {"__pycache__", "node_modules", ".venv", "venv", "dist", "build", ".jcode"}
        for dirpath, dirnames, filenames in os.walk(repo_root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in skip]
            for fname in filenames:
                if Path(fname).suffix in exts:
                    yield os.path.join(dirpath, fname)

    def _resolve_calls(
        self,
        all_nodes: dict[NodeId, Node],
        provisional_nodes: dict[NodeId, Node],
        all_edges: list[Edge],
    ) -> tuple[list[Edge], dict[NodeId, Node]]:
        """
        Replace provisional targets with real node ids where possible.

        For every edge whose target is provisional, we try to match it against
        the name index. If a real node is found the edge is rewritten. If not
        (e.g. a cross-module view referenced from urls.py), the provisional
        stub node is returned in `persisted_provisionals` so callers can
        persist it and keep the edge intact.

        Returns (resolved_edges, persisted_provisionals).
        """
        name_index: dict[str, list[NodeId]] = {}
        for node in all_nodes.values():
            if node.node_type in (NodeType.FUNCTION, NodeType.METHOD,
                                  NodeType.CLASS, NodeType.VARIABLE):
                name_index.setdefault(node.name, []).append(node.id)

        resolved: list[Edge] = []
        persisted_provisionals: dict[NodeId, Node] = {}

        for edge in all_edges:
            if edge.target_id in all_nodes:
                # Target already known — keep as-is
                resolved.append(edge)
                continue
            provisional = provisional_nodes.get(edge.target_id)
            if provisional is None:
                # Non-provisional unknown target — drop
                continue
            real_ids = name_index.get(provisional.name, [])
            if real_ids:
                # Resolved — rewrite edge(s) to point at real nodes
                for real_id in real_ids:
                    resolved.append(Edge(
                        source_id=edge.source_id,
                        target_id=real_id,
                        edge_type=edge.edge_type,
                    ))
            else:
                # Unresolved (e.g. cross-module symbol) — keep edge + persist stub
                resolved.append(edge)
                persisted_provisionals[provisional.id] = provisional

        return resolved, persisted_provisionals

    def _make_snapshot(self, repo_root: str) -> IndexSnapshot:
        """Build a snapshot from current graph state."""
        all_nodes = self._graph.all_nodes()
        sorted_ids = sorted(n.id.hex for n in all_nodes)
        snap_hash = hashlib.sha256("|".join(sorted_ids).encode()).hexdigest()

        file_hashes = self._graph.get_file_hashes()
        file_count = len(file_hashes) if file_hashes else len(
            [n for n in all_nodes if n.node_type == NodeType.MODULE]
        )

        return IndexSnapshot(
            snapshot_hash=snap_hash,
            indexed_at=time.time(),
            file_count=file_count,
            node_count=len(all_nodes),
            edge_count=self._graph.count_edges(),
            root_path=repo_root,
        )
