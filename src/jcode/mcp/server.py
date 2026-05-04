"""
MCP server — exposes jcode as five tools any AI agent can call.

Tools
-----
jcode_search        semantic + keyword search → candidate entry points
jcode_context       DFS forward with optional scope → code context
jcode_blast_radius  BFS reverse (always global) → impact analysis
jcode_feature_map   folder-level overview of the codebase
jcode_index         trigger a (re-)index of a directory
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

def _traversal_dict(r: TraversalResult) -> dict[str, Any]:
    out: dict[str, Any] = {
        "entry": _node_dict(r.entry_node),
        "depth_reached": r.depth_reached,
        "nodes": [_node_dict(n) for n in r.nodes],
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

    Uses semantic vector search when available (sentence-transformers installed),
    falls back to FTS5 keyword search otherwise.

    scope:  Optional folder prefix to restrict search, e.g. "comments/".
            If scoped search returns nothing, automatically expands to full graph.
            Pass scope only when the user's request clearly names a feature area.

    Args:
        query:     Natural-language or symbol-name, e.g. "date parsing comments".
        limit:     Max results (default 10).
        scope:     Folder prefix to restrict search (optional).
        jcode_dir: Path to the .jcode store directory.
    """
    graph = _open_graph(jcode_dir)
    results = search_semantic(graph, query, limit=limit, scope=scope)
    out = []
    for node, score in results:
        d = _node_dict(node)
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

    scope:  Optional folder prefix — DFS stops at the boundary and returns
            external dependencies as a separate list. The LLM should inspect
            external_deps and decide whether to follow them with another
            jcode_context call or jcode_blast_radius.

    Args:
        node_id:   Node id hex string (from jcode_search results).
        max_depth: Hops to follow (default 3).
        scope:     Folder prefix to limit traversal (optional).
        jcode_dir: Path to the .jcode store directory.
    """
    graph = _open_graph(jcode_dir)
    traversal = GraphTraversal(graph)
    result = traversal.context(NodeId(node_id), max_depth=max_depth, scope=scope)
    return _traversal_dict(result)

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

    Returns: folders with their top-level classes and functions.
    """
    graph = _open_graph(jcode_dir)
    from jcode.domain.models import NodeType
    nodes = graph.all_nodes()

    _SKIP = {NodeType.MODULE, NodeType.IMPORT, NodeType.VARIABLE}
    # Cap per-folder lists so the response stays a usable overview rather than
    # a full symbol dump. Claude should use jcode_search with scope= to drill in.
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
    full_reindex: bool = False,
    jcode_dir: str | None = None,
) -> dict[str, Any]:
    """
    Index or re-index a repository.

    Args:
        repo_path:    Absolute path to the repository root.
        full_reindex: Drop and rebuild entire graph (default False).
        jcode_dir:    Override .jcode store location.
    """
    from jcode.indexer.plugins import build_parser
    jcode_path = Path(jcode_dir or os.path.join(repo_path, ".jcode")).resolve()
    jcode_path.mkdir(parents=True, exist_ok=True)

    # Evict from cache so the next tool call picks up the fresh index
    _graph_cache.pop(str(jcode_path), None)

    store    = ObjectStore(jcode_path)
    graph    = GraphDB(jcode_path)
    parser   = build_parser(repo_path)
    indexer  = Indexer(parser, store, graph)
    snapshot = indexer.index(repo_path, full_reindex=full_reindex)

    return {
        "snapshot_hash": snapshot.snapshot_hash[:16],
        "files": snapshot.file_count,
        "nodes": snapshot.node_count,
        "edges": snapshot.edge_count,
        "indexed_at": snapshot.indexed_at,
    }
