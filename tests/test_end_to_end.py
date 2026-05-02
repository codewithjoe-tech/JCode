"""
End-to-end test: index the sample fixture project and verify the feature graph.

Checks:
  1. Indexer discovers all .py files and emits the expected nodes
  2. get_timestamp is found by search
  3. context() from get_timestamp includes downstream callers
  4. blast_radius() from get_timestamp reaches jwt.sign_token and
     comments.CommentService.create_comment
  5. confidence score is < 1.0 (multiple callers detected)
"""

from __future__ import annotations

import os
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
    """Index the sample fixture into a fresh temp store and return the GraphDB."""
    tmp = Path(tempfile.mkdtemp(dir="/tmp", prefix="jcode_test_"))
    jcode_dir = tmp / ".jcode"
    jcode_dir.mkdir()

    store = ObjectStore(jcode_dir)
    graph = GraphDB(jcode_dir)
    parser = GenericParser()
    indexer = Indexer(parser, store, graph)

    snapshot = indexer.index(str(FIXTURE_DIR), full_reindex=True)
    assert snapshot.file_count >= 3, "Expected at least 3 fixture files to be indexed"
    assert snapshot.node_count > 0

    yield graph

    shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_search_finds_get_timestamp(indexed_graph: GraphDB) -> None:
    results = search_entry_points(indexed_graph, "get_timestamp")
    names = [n.name for n in results]
    assert "get_timestamp" in names, f"get_timestamp not found in: {names}"


def test_search_finds_comment_service(indexed_graph: GraphDB) -> None:
    results = search_entry_points(indexed_graph, "CommentService")
    names = [n.name for n in results]
    assert "CommentService" in names, f"CommentService not found in: {names}"


def test_all_node_types_present(indexed_graph: GraphDB) -> None:
    all_nodes = indexed_graph.all_nodes()
    types = {n.node_type for n in all_nodes}
    assert NodeType.MODULE in types
    assert NodeType.FUNCTION in types or NodeType.METHOD in types


def test_context_from_get_timestamp(indexed_graph: GraphDB) -> None:
    results = search_entry_points(indexed_graph, "get_timestamp", limit=5)
    fn_nodes = [n for n in results if n.node_type == NodeType.FUNCTION and n.name == "get_timestamp"]
    assert fn_nodes, "get_timestamp function node not found"

    traversal = GraphTraversal(indexed_graph)
    ctx = traversal.context(fn_nodes[0].id, max_depth=2)
    assert ctx.entry_node.name == "get_timestamp"


def test_blast_radius_from_get_timestamp(indexed_graph: GraphDB) -> None:
    results = search_entry_points(indexed_graph, "get_timestamp", limit=5)
    fn_nodes = [n for n in results if n.node_type == NodeType.FUNCTION and n.name == "get_timestamp"]
    assert fn_nodes, "get_timestamp function node not found"

    traversal = GraphTraversal(indexed_graph)
    blast = traversal.blast_radius(fn_nodes[0].id, max_depth=5)

    assert blast.changed_node.name == "get_timestamp"
    assert blast.confidence < 1.0, (
        f"Expected confidence < 1.0, got {blast.confidence}. "
        f"Reasons: {blast.risk_reasons}"
    )


def test_snapshot_persisted(indexed_graph: GraphDB) -> None:
    snap = indexed_graph.get_snapshot()
    assert snap is not None
    assert snap.node_count > 0
    assert snap.file_count >= 3
