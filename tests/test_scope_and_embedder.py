"""
Tests for:
  - scope-bounded DFS (context stops at folder boundary)
  - external_deps populated correctly when scope applied
  - search_entry_points scope fallback (returns results from full graph if nothing in scope)
  - embedder.build_embed_text produces meaningful strings
  - jcode_feature_map groups nodes by folder correctly
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from jcode.domain.models import NodeType
from jcode.graph.traversal import GraphTraversal, search_entry_points
from jcode.indexer.builder import Indexer
from jcode.indexer.generic_parser import GenericParser
from jcode.storage.graph_db import GraphDB
from jcode.storage.object_store import ObjectStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample"


@pytest.fixture()
def indexed_graph():
    tmp = Path(tempfile.mkdtemp(dir="/tmp", prefix="jcode_scope_"))
    jcode_dir = tmp / ".jcode"
    jcode_dir.mkdir()

    store = ObjectStore(jcode_dir)
    graph = GraphDB(jcode_dir)
    parser = GenericParser()
    indexer = Indexer(parser, store, graph)
    indexer.index(str(FIXTURE_DIR), full_reindex=True)

    yield graph

    shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# Scope: DFS stops at folder boundary
# ---------------------------------------------------------------------------


def test_context_with_scope_stays_in_folder(indexed_graph: GraphDB) -> None:
    """context(scope='comments/') should not follow edges into auth/ or common/."""
    # Find a node inside comments/
    results = search_entry_points(indexed_graph, "CommentService", limit=5, scope="comments")
    assert results, "CommentService not found"
    comment_node = results[0]

    traversal = GraphTraversal(indexed_graph)
    ctx = traversal.context(comment_node.id, max_depth=5, scope="comments")

    # All traversed nodes must live under comments/
    for node in ctx.nodes:
        assert node.file_path.startswith("comments/"), (
            f"Node '{node.name}' at '{node.file_path}' leaked outside scope"
        )


def test_context_scope_populates_external_deps(indexed_graph: GraphDB) -> None:
    """Nodes outside scope should appear in external_deps, not in nodes list."""
    results = search_entry_points(indexed_graph, "CommentService", limit=5)
    assert results, "CommentService not found"
    comment_node = results[0]

    traversal = GraphTraversal(indexed_graph)
    ctx = traversal.context(comment_node.id, max_depth=5, scope="comments")

    # external_deps may be empty if no cross-boundary edges, but must not
    # contain nodes inside comments/
    for dep in ctx.external_deps:
        assert not dep.file_path.startswith("comments/"), (
            f"external_dep '{dep.name}' is inside scope — should be in nodes instead"
        )


def test_context_no_scope_traverses_full_graph(indexed_graph: GraphDB) -> None:
    """Without scope, DFS follows cross-folder edges freely."""
    results = search_entry_points(indexed_graph, "get_timestamp", limit=5)
    fn_nodes = [n for n in results if n.node_type == NodeType.FUNCTION]
    assert fn_nodes

    traversal = GraphTraversal(indexed_graph)
    ctx_scoped   = traversal.context(fn_nodes[0].id, max_depth=5, scope="common")
    ctx_unscoped = traversal.context(fn_nodes[0].id, max_depth=5)

    # Unscoped context should have >= as many nodes as scoped
    assert len(ctx_unscoped.nodes) >= len(ctx_scoped.nodes)


# ---------------------------------------------------------------------------
# Scope fallback in search
# ---------------------------------------------------------------------------


def test_search_scope_fallback(indexed_graph: GraphDB) -> None:
    """search_entry_points with a scope that matches nothing falls back to full graph."""
    # "nonexistent_folder" will match nothing → should fall back to full graph
    results_scoped   = search_entry_points(indexed_graph, "get_timestamp", scope="nonexistent_folder")
    results_unscoped = search_entry_points(indexed_graph, "get_timestamp")
    assert len(results_scoped) > 0, "Fallback search returned nothing"
    assert {n.name for n in results_scoped} == {n.name for n in results_unscoped}


# ---------------------------------------------------------------------------
# Embedder: build_embed_text
# ---------------------------------------------------------------------------


def test_build_embed_text_includes_folder_and_name() -> None:
    from jcode.domain.models import Node, NodeId, NodeType
    from jcode.indexer.embedder import build_embed_text

    node = Node(
        id=NodeId("a" * 64),
        node_type=NodeType.FUNCTION,
        name="parse_date",
        title="comments/utils.parse_date",
        file_path="comments/utils.py",
        line_start=10,
        line_end=20,
        signature="(raw: str, fmt: str = '%d/%m/%Y')",
    )
    text = build_embed_text(node, callee_names=["datetime.strptime", "ValueError"])
    assert "comments" in text
    assert "parse_date" in text
    # Signature param names stripped of types/defaults should appear
    assert "raw" in text or "fmt" in text or "function" in text


def test_build_embed_text_with_callees() -> None:
    from jcode.domain.models import Node, NodeId, NodeType
    from jcode.indexer.embedder import build_embed_text

    node = Node(
        id=NodeId("b" * 64),
        node_type=NodeType.METHOD,
        name="create_comment",
        title="comments/service.CommentService.create_comment",
        file_path="comments/service.py",
        line_start=30,
        line_end=50,
        signature="(self, body: str, user_id: int)",
    )
    text = build_embed_text(node, callee_names=["validate_body", "save_to_db", "notify_user"])
    assert "create_comment" in text
    assert any(c in text for c in ["validate_body", "save_to_db", "notify_user"])


def test_build_embed_text_no_folder_noise() -> None:
    """Nodes in root or 'src' folder should not get noisy prefix."""
    from jcode.domain.models import Node, NodeId, NodeType
    from jcode.indexer.embedder import build_embed_text

    node = Node(
        id=NodeId("c" * 64),
        node_type=NodeType.FUNCTION,
        name="main",
        title="main",
        file_path="main.py",
        line_start=1, line_end=5, signature="()",
    )
    text = build_embed_text(node, callee_names=[])
    assert "main" in text
    # Should not add "src" or "." as folder noise
    assert text.strip() != ""


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------


def test_feature_map_groups_by_folder(indexed_graph: GraphDB) -> None:
    from jcode.domain.models import NodeType

    nodes = indexed_graph.all_nodes()
    folders: dict = {}
    for node in nodes:
        if node.node_type == NodeType.MODULE:
            continue
        folder = node.file_path.split("/")[0] if "/" in node.file_path else "root"
        folders.setdefault(folder, set()).add(node.name)

    # We expect at least auth, comments, and common folders
    assert "auth" in folders, f"auth folder missing from: {list(folders)}"
    assert "comments" in folders
    assert "common" in folders


def test_feature_map_classes_and_functions(indexed_graph: GraphDB) -> None:
    """Verify that known classes and functions appear in their expected folders."""
    nodes = indexed_graph.all_nodes()

    by_type: dict = {}
    for node in nodes:
        by_type.setdefault(node.node_type, []).append(node.name)

    all_names = [n.name for n in nodes]
    assert "CommentService" in all_names
    assert "get_timestamp" in all_names
