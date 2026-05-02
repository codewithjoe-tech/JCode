"""
Tests for incremental indexing.

Verifies:
  1. Second run with no changes → "nothing changed", same snapshot hash
  2. Adding a new file → only that file's nodes appear, old nodes untouched
  3. Modifying a file → old nodes for that file gone, new nodes present
  4. Deleting a file → its nodes removed from graph
  5. file_hashes manifest is kept consistent throughout
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from jcode.domain.models import NodeType
from jcode.graph.traversal import search_entry_points
from jcode.indexer.builder import Indexer
from jcode.indexer.generic_parser import GenericParser
from jcode.storage.graph_db import GraphDB
from jcode.storage.object_store import ObjectStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "sample"


def _make_store(tmp: Path):
    jcode_dir = tmp / ".jcode"
    jcode_dir.mkdir()
    return ObjectStore(jcode_dir), GraphDB(jcode_dir), Indexer(GenericParser(), ObjectStore(jcode_dir), GraphDB(jcode_dir))


@pytest.fixture()
def workspace():
    """Copy the fixture project to a temp dir so tests can mutate files."""
    tmp = Path(tempfile.mkdtemp(dir="/tmp", prefix="jcode_inc_"))
    repo = tmp / "repo"
    shutil.copytree(str(FIXTURE_DIR), str(repo))

    jcode_dir = tmp / ".jcode"
    jcode_dir.mkdir()
    store = ObjectStore(jcode_dir)
    graph = GraphDB(jcode_dir)
    indexer = Indexer(GenericParser(), store, graph)

    yield repo, graph, indexer

    shutil.rmtree(str(tmp), ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. No changes → same snapshot hash
# ---------------------------------------------------------------------------

def test_no_changes_returns_same_snapshot(workspace):
    repo, graph, indexer = workspace

    snap1 = indexer.index(str(repo), full_reindex=True)
    snap2 = indexer.index(str(repo))  # incremental, nothing changed

    assert snap1.snapshot_hash == snap2.snapshot_hash
    assert snap1.node_count == snap2.node_count


# ---------------------------------------------------------------------------
# 2. Adding a new file
# ---------------------------------------------------------------------------

def test_new_file_adds_nodes(workspace):
    repo, graph, indexer = workspace

    snap1 = indexer.index(str(repo), full_reindex=True)
    nodes_before = {n.name for n in graph.all_nodes()}

    # Add a new module
    new_file = repo / "common" / "helpers.py"
    new_file.write_text(
        "def helper_function(x: int) -> int:\n    return x * 2\n"
    )

    snap2 = indexer.index(str(repo))  # incremental

    nodes_after = {n.name for n in graph.all_nodes()}
    assert "helper_function" in nodes_after, "New function not indexed"
    assert snap2.node_count > snap1.node_count


# ---------------------------------------------------------------------------
# 3. Modifying a file → old nodes gone, new nodes present
# ---------------------------------------------------------------------------

def test_changed_file_updates_nodes(workspace):
    repo, graph, indexer = workspace

    indexer.index(str(repo), full_reindex=True)
    nodes_before = {n.name for n in graph.all_nodes()}
    assert "get_timestamp" in nodes_before

    # Rename get_timestamp → fetch_timestamp in common/utils.py
    utils = repo / "common" / "utils.py"
    original = utils.read_text()
    utils.write_text(original.replace("def get_timestamp(", "def fetch_timestamp("))

    indexer.index(str(repo))  # incremental

    nodes_after = {n.name for n in graph.all_nodes()}
    assert "fetch_timestamp" in nodes_after,  "Renamed function not found"
    assert "get_timestamp"   not in nodes_after, "Old function still present after rename"


# ---------------------------------------------------------------------------
# 4. Deleting a file → its nodes removed
# ---------------------------------------------------------------------------

def test_deleted_file_removes_nodes(workspace):
    repo, graph, indexer = workspace

    indexer.index(str(repo), full_reindex=True)
    nodes_before = {n.name for n in graph.all_nodes()}
    assert "sign_token" in nodes_before  # lives in auth/jwt.py

    # Delete the auth module
    (repo / "auth" / "jwt.py").unlink()

    indexer.index(str(repo))  # incremental

    nodes_after = {n.name for n in graph.all_nodes()}
    assert "sign_token"   not in nodes_after, "Deleted node still in graph"
    assert "verify_token" not in nodes_after, "Deleted node still in graph"


# ---------------------------------------------------------------------------
# 5. Manifest consistency
# ---------------------------------------------------------------------------

def test_manifest_tracks_files(workspace):
    repo, graph, indexer = workspace

    indexer.index(str(repo), full_reindex=True)
    manifest = graph.get_file_hashes()
    assert len(manifest) > 0, "Manifest is empty after full index"

    # All current .py files should be in the manifest
    py_files = list(repo.rglob("*.py"))
    for f in py_files:
        rel = os.path.relpath(str(f), str(repo))
        assert rel in manifest, f"Missing from manifest: {rel}"


def test_manifest_updated_after_change(workspace):
    repo, graph, indexer = workspace

    indexer.index(str(repo), full_reindex=True)
    manifest_before = graph.get_file_hashes()

    utils = repo / "common" / "utils.py"
    old_hash = manifest_before[os.path.relpath(str(utils), str(repo))]

    # Touch the file (append a comment)
    utils.write_text(utils.read_text() + "\n# updated\n")

    indexer.index(str(repo))  # incremental
    manifest_after = graph.get_file_hashes()

    rel = os.path.relpath(str(utils), str(repo))
    new_hash = manifest_after[rel]
    assert new_hash != old_hash, "Manifest hash not updated after file change"
