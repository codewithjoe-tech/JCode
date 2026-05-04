"""
MCP server — exposes jcode as five tools any AI agent can call.

Tools
-----
jcode_search        semantic + keyword search → candidate entry points + inline snippets
jcode_context       DFS forward with optional scope → code context
jcode_blast_radius  BFS reverse (always global) → impact analysis
jcode_feature_map   folder-level overview of the codebase
jcode_index         trigger a (re-)index of one or more directories
"""

import os
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from jcode.domain.models import BlastRadiusResult, Node, NodeId, TraversalResult
from jcode.graph.traversal import GraphTraversal, search_semantic, _fp_root
from jcode.indexer.builder import Indexer
from jcode.storage.graph_db import GraphDB
from jcode.storage.object_store import ObjectStore

mcp = FastMCP("jcode")

_JCODE_DIR_ENV = "JCODE_DIR"

# Module-level cache: keyed by resolved path string so multiple repos can
# coexist in the same process, but the common case (one repo) pays the
# SQLite connection + DDL cost exactly once per server lifetime.
_graph_cache: dict[str, GraphDB] = {}

def _open_graph(jcode_dir: str | None) -> GraphDB:
    path = Path(jcode_dir or os.environ.get(_JCODE_DIR_ENV, ".jcode")).resolve()
    key  = str(path)
    if key not in _graph_cache:
        if not path.exists():
            raise FileNotFoundError(
                f"jcode store not found at '{path}'. Run `jcode index <repo>` first."
            )
        _graph_cache[key] = GraphDB(path)
    return _graph_cache[key]

# Snippet helper — reads source lines for a node inline, avoiding a separate Read call.
_SNIPPET_LINES = 12   # max lines returned per node

def _read_snippet(
    node: Node,
    repo_root: str,
    file_cache: dict[str, list[str]],
) -> str:
    """
    Return up to _SNIPPET_LINES of source starting at node.line_start.
    Uses file_cache to avoid re-reading the same file for multiple nodes.
    Returns empty string silently on any IO error.
    """
    try:
        # Normalise separator so Windows-stored paths resolve on any OS
        rel = node.file_path.replace("\\", "/")
        if rel not in file_cache:
            file_cache[rel] = (
                Path(repo_root, rel)
                .read_text(encoding="utf-8", errors="replace")
                .splitlines()
            )
        lines = file_cache[rel]
        start = max(0, node.line_start - 1)
        end   = min(len(lines), start + _SNIPPET_LINES)
        return "\n".join(lines[start:end])
    except OSError:
        return ""

def _node_dict(n: Node) -> dict[str, Any]:
    return {
        "id": n.id.hex,
        "type": n.node_type.value,
        "name": n.name,
        "title": n.title,
        "file": n.file_path,
        "lines": f"{n.line_start}-{n.line_end}",
        "signature": n.signature,
    }

def _node_dict_with_snippet(
    n: Node,
    repo_root: str | None,
    file_cache: dict[str, list[str]],
) -> dict[str, Any]:
    """Like _node_dict but includes an inline source snippet when repo_root is known."""
    d = _node_dict(n)
    if repo_root:
        snippet = _read_snippet(n, repo_root, file_cache)
        if snippet:
            d["snippet"] = snippet
    return d

def _traversal_dict(r: TraversalResult, repo_root: str | None = None) -> dict[str, Any]:
    file_cache: dict[str, list[str]] = {}
    out: dict[str, Any] = {
        "entry": _node_dict_with_snippet(r.entry_node, repo_root, file_cache),
        "depth_reached": r.depth_reached,
        "nodes": [_node_dict_with_snippet(n, repo_root, file_cache) for n in r.nodes],
        "edges": [
            {"from": e.source_id.hex[:12], "to": e.target_id.hex[:12],
             "type": e.edge_type}
            for e in r.edges
        ],
    }
    if r.external_deps:
        out["external_deps"] = [_node_dict(n) for n in r.external_deps]
        out["external_deps_note"] = (
            "These nodes are outside the requested scope. "
            "Call jcode_context or jcode_blast_radius on them if needed."
        )
    return out

def _blast_dict(r: BlastRadiusResult) -> dict[str, Any]:
    return {
        "changed_node": _node_dict(r.changed_node),
        "confidence": round(r.confidence, 3),
        "risk_reasons": r.risk_reasons,
        "affected_count": len(r.affected_nodes),
        "affected_nodes": [_node_dict(n) for n in r.affected_nodes],
    }
# Tools
@mcp.tool()
def jcode_search(
    query: str,
    limit: int = 10,
    scope: str | None = None,
    jcode_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search the feature graph for nodes matching a concept or symbol name.

    Returns results with inline code snippets so you can read the code
    without a separate file Read call in most cases.

    Uses hybrid search: FTS keyword match first (exact tokens, URL domains,
    symbol names), then semantic vector search to fill remaining slots.

    scope:  Optional folder prefix to restrict search, e.g. "partners/mmb".
            If scoped search returns nothing, automatically expands to full graph.

    Args:
        query:     Natural-language or symbol-name, e.g. "date parsing comments".
        limit:     Max results (default 10).
        scope:     Folder prefix to restrict search (optional).
        jcode_dir: Path to the .jcode store directory.
    """
    graph = _open_graph(jcode_dir)
    snap = graph.get_snapshot()
    repo_root = snap.root_path if snap else None

    results = search_semantic(graph, query, limit=limit, scope=scope)
    file_cache: dict[str, list[str]] = {}
    out = []
    for node, score in results:
        d = _node_dict_with_snippet(node, repo_root, file_cache)
        d["score"] = round(score, 3)
        out.append(d)
    return out

@mcp.tool()
def jcode_context(
    node_id: str,
    max_depth: int = 3,
    scope: str | None = None,
    jcode_dir: str | None = None,
) -> dict[str, Any]:
    """
    Return code context rooted at node_id via forward DFS.

    Includes inline snippets for the entry node and all traversed nodes,
    so you rarely need a follow-up Read call.

    scope:  Optional folder prefix — DFS stops at the boundary and returns
            external dependencies as a separate list. The LLM should inspect
            external_deps and decide whether to follow them.

    Args:
        node_id:   Node id hex string (from jcode_search results).
        max_depth: Hops to follow (default 3).
        scope:     Folder prefix to limit traversal (optional).
        jcode_dir: Path to the .jcode store directory.
    """
    graph = _open_graph(jcode_dir)
    snap = graph.get_snapshot()
    repo_root = snap.root_path if snap else None
    traversal = GraphTraversal(graph)
    result = traversal.context(NodeId(node_id), max_depth=max_depth, scope=scope)
    return _traversal_dict(result, repo_root=repo_root)

@mcp.tool()
def jcode_blast_radius(
    node_id: str,
    max_depth: int = 5,
    jcode_dir: str | None = None,
) -> dict[str, Any]:
    """
    Analyse blast radius of changing node_id via reverse BFS.

    Always global — scope is never applied here because impact does not
    respect folder boundaries. A low confidence score means the agent
    should warn the user before applying the change.

    Args:
        node_id:   Node id hex string.
        max_depth: Reverse traversal depth (default 5).
        jcode_dir: Path to the .jcode store directory.
    """
    graph = _open_graph(jcode_dir)
    result = GraphTraversal(graph).blast_radius(NodeId(node_id), max_depth=max_depth)
    return _blast_dict(result)

@mcp.tool()
def jcode_feature_map(
    jcode_dir: str | None = None,
) -> dict[str, Any]:
    """
    Return a high-level map of the codebase organised by folder/feature.

    Use this first when the user's request doesn't name a specific function —
    it gives you the lay of the land so you can choose the right scope
    before calling jcode_search or jcode_context.

    Returns: folders with their top-level classes and functions (capped at 20
    per type — use jcode_search with scope= to drill in).
    """
    graph = _open_graph(jcode_dir)
    from jcode.domain.models import NodeType
    nodes = graph.all_nodes()

    _SKIP = {NodeType.MODULE, NodeType.IMPORT, NodeType.VARIABLE}
    _MAX_PER_TYPE = 20

    feature_map: dict[str, dict] = {}
    for node in nodes:
        if node.node_type in _SKIP:
            continue
        folder = _fp_root(node.file_path)
        if folder not in feature_map:
            feature_map[folder] = {"classes": set(), "functions": set(), "methods": set()}
        if node.node_type == NodeType.CLASS:
            feature_map[folder]["classes"].add(node.name)
        elif node.node_type == NodeType.FUNCTION:
            feature_map[folder]["functions"].add(node.name)
        elif node.node_type == NodeType.METHOD:
            feature_map[folder]["methods"].add(node.name)

    # Sort, cap, and convert sets to lists
    summary: dict[str, dict] = {}
    for folder, buckets in sorted(feature_map.items()):
        summary[folder] = {}
        for key, names in buckets.items():
            sorted_names = sorted(names)
            summary[folder][f"{key}_count"] = len(sorted_names)
            summary[folder][key] = sorted_names[:_MAX_PER_TYPE]
            if len(sorted_names) > _MAX_PER_TYPE:
                summary[folder][key].append(f"… +{len(sorted_names) - _MAX_PER_TYPE} more")

    return {
        "folders": summary,
        "hint": (
            "Each folder is a feature area. Lists are capped at 20 — use jcode_search "
            "with scope=<folder> to find specific symbols within a folder."
        ),
    }

@mcp.tool()
def jcode_index(
    repo_path: str,
    extra_paths: list[str] | None = None,
    full_reindex: bool = False,
    jcode_dir: str | None = None,
) -> dict[str, Any]:
    """
    Index or re-index one or more directories into a shared .jcode store.

    For monorepos with multiple sub-projects, pass the additional roots via
    extra_paths. All paths are indexed into the same graph so jcode_search
    and jcode_context work across the whole workspace.

    Example — index a monorepo with two sub-projects:
        jcode_index(
            repo_path="/workspace/pluto-mono/pluto",
            extra_paths=["/workspace/pluto-mono/file-import-pluto"],
            jcode_dir="/workspace/pluto-mono/.jcode",
        )

    Args:
        repo_path:    Primary repo root (also determines default .jcode location).
        extra_paths:  Additional repo roots to index into the same store.
        full_reindex: Drop and rebuild entire graph (default False).
        jcode_dir:    Override .jcode store location.
    """
    from jcode.indexer.plugins import build_parser

    jcode_path = Path(jcode_dir or os.path.join(repo_path, ".jcode")).resolve()
    jcode_path.mkdir(parents=True, exist_ok=True)

    # Evict cache so next tool call picks up fresh index
    _graph_cache.pop(str(jcode_path), None)

    store = ObjectStore(jcode_path)
    graph = GraphDB(jcode_path)

    all_paths = [repo_path] + (extra_paths or [])
    snapshots = []

    for i, path in enumerate(all_paths):
        parser  = build_parser(path)
        indexer = Indexer(parser, store, graph)
        # Only wipe on the first path — subsequent paths are always incremental
        snap = indexer.index(path, full_reindex=(full_reindex and i == 0))
        snapshots.append(snap)

    last = snapshots[-1]
    return {
        "snapshot_hash": last.snapshot_hash[:16],
        "paths_indexed": len(all_paths),
        "files": last.file_count,
        "nodes": last.node_count,
        "edges": last.edge_count,
        "indexed_at": last.indexed_at,
    }
