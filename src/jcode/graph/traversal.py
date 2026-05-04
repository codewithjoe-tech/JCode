"""
Graph traversal — BFS / DFS, blast-radius, and scoped context.

context() now accepts an optional scope (folder prefix).
  - DFS stops at scope boundary
  - External dependencies are flagged separately for the LLM
  - LLM decides whether to follow them or not

blast_radius() is always global — impact doesn't respect folder boundaries.
"""

from collections import deque

from jcode.domain.models import (
    BlastRadiusResult,
    Node,
    NodeId,
    TraversalResult,
)
from jcode.domain.ports import GraphReaderPort, TraversalPort


class GraphTraversal(TraversalPort):

    def __init__(self, reader: GraphReaderPort) -> None:
        self._reader = reader

    def context(
        self,
        entry_node_id: NodeId,
        max_depth: int = 3,
        scope: str | None = None,
    ) -> TraversalResult:
        """
        DFS forward from *entry_node_id*.
        scope: folder prefix e.g. "comments/" — DFS stops at boundary and
               flags external dependencies instead of following them blindly.
        """
        entry = self._reader.get_node(entry_node_id)
        if entry is None:
            raise KeyError(f"Node not found: {entry_node_id}")

        result = TraversalResult(entry_node=entry)
        visited: set[str] = set()
        self._dfs(entry_node_id, 0, max_depth, visited, result, scope)
        return result

    def blast_radius(
        self,
        changed_node_id: NodeId,
        max_depth: int = 5,
    ) -> BlastRadiusResult:
        """BFS reverse — always global, scope never applied here."""
        changed = self._reader.get_node(changed_node_id)
        if changed is None:
            raise KeyError(f"Node not found: {changed_node_id}")

        result = BlastRadiusResult(changed_node=changed)
        visited: set[str] = set()
        depth_map: dict[str, int] = {}
        queue: deque[tuple[NodeId, int]] = deque([(changed_node_id, 0)])
        visited.add(changed_node_id.hex)

        while queue:
            current_id, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for edge in self._reader.predecessors(current_id):
                if edge.source_id.hex in visited:
                    continue
                visited.add(edge.source_id.hex)
                depth_map[edge.source_id.hex] = depth + 1
                caller = self._reader.get_node(edge.source_id)
                if caller:
                    result.affected_nodes.append(caller)
                    result.affected_edges.append(edge)
                    queue.append((edge.source_id, depth + 1))

        result.confidence, result.risk_reasons = self._score(
            changed, result.affected_nodes, depth_map
        )
        return result
    # Internal DFS with scope awareness
    def _dfs(
        self,
        node_id: NodeId,
        depth: int,
        max_depth: int,
        visited: set[str],
        result: TraversalResult,
        scope: str | None,
    ) -> None:
        if node_id.hex in visited or depth > max_depth:
            return
        visited.add(node_id.hex)
        result.depth_reached = max(result.depth_reached, depth)

        node = self._reader.get_node(node_id)
        if node and node != result.entry_node:
            result.nodes.append(node)

        for edge in self._reader.successors(node_id):
            result.edges.append(edge)
            target = self._reader.get_node(edge.target_id)
            if target is None:
                continue

            # Scope boundary check
            if scope and not target.file_path.startswith(scope.rstrip("/") + "/"):
                # External dependency — flag it, don't follow
                if target not in result.external_deps:
                    result.external_deps.append(target)
                continue

            self._dfs(edge.target_id, depth + 1, max_depth, visited, result, scope)
    # Confidence scoring
    def _score(
        self,
        changed: Node,
        affected: list[Node],
        depth_map: dict[str, int],
    ) -> tuple[float, list[str]]:
        score = 1.0
        reasons: list[str] = []

        if len(affected) > 5:
            score -= 0.20
            reasons.append(f"High fan-in: {len(affected)} callers/dependents will be affected.")

        changed_module = changed.file_path.split("/")[0]
        cross = [n for n in affected if n.file_path.split("/")[0] != changed_module]
        if cross:
            score -= 0.10
            names = ", ".join(n.title for n in cross[:3])
            extra = f" (+{len(cross)-3} more)" if len(cross) > 3 else ""
            reasons.append(f"Cross-module impact: {names}{extra}.")

        deep = [nid for nid, d in depth_map.items() if d >= 3]
        if deep:
            score -= 0.05
            reasons.append(f"{len(deep)} node(s) are indirect dependencies (3+ hops).")

        if not self._has_test_coverage(changed):
            score -= 0.15
            reasons.append(
                f"No test file found for '{changed.file_path}' — change is unverified."
            )

        return max(0.0, min(1.0, score)), reasons

    def _has_test_coverage(self, node: Node) -> bool:
        return "test" in node.file_path.lower()
# Search helpers
def search_entry_points(
    reader: GraphReaderPort,
    query: str,
    limit: int = 10,
    scope: str | None = None,
) -> list[Node]:
    """
    Search nodes by keyword. scope limits to a folder prefix.
    Falls back to full search if scoped search returns nothing.
    """
    if scope:
        results = reader.search_nodes(query, limit=limit, scope=scope)
        if results:
            return results
        # Nothing found in scope — expand to full graph
    return reader.search_nodes(query, limit=limit)

def search_semantic(
    reader,
    query: str,
    limit: int = 10,
    scope: str | None = None,
) -> list[tuple[Node, float]]:
    """
    Hybrid search: FTS keyword match first, then semantic vector search.

    FTS runs first so exact tokens (e.g. a partner domain name like
    "mymoneybazaar") are always found even when the semantic score is low.
    Semantic results fill the remaining slots, deduplicated against FTS hits.
    Falls back to FTS-only if embeddings aren't available.
    """
    from jcode.indexer.embedder import get_embedder

    # --- FTS pass (always runs) ---
    fts_nodes = search_entry_points(reader, query, limit=limit, scope=scope)
    fts_ids = {n.id.hex for n in fts_nodes}
    results: list[tuple[Node, float]] = [(n, 1.0) for n in fts_nodes]

    # --- Semantic pass (fills remaining slots) ---
    embedder = get_embedder()
    if embedder.is_available() and reader.embeddings_count() > 0:
        query_vec = embedder.embed(query)
        semantic = reader.search_semantic(query_vec, limit=limit, scope=scope)
        for node, score in semantic:
            if node.id.hex not in fts_ids:
                results.append((node, score))

    # Return up to `limit` results, FTS hits first
    return results[:limit]
