"""
Embedder — builds text from nodes and embeds with sentence-transformers.

Falls back gracefully if sentence-transformers is not installed:
  - search_semantic() won't be called (MCP server checks has_embeddings())
  - FTS5 keyword search is used instead

Text built from (in order, use what's available):
  1. source snippet — first 8 lines of the node body (richest signal)
  2. signature      — parameter names carry strong semantic signal
  3. callee names   — what a function calls describes what it does
  4. name + type    — always present
  5. file_path      — folder name tells you the feature area
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jcode.domain.models import Node
    from jcode.storage.graph_db import GraphDB

_MODEL_NAME = "all-MiniLM-L6-v2"       # sentence-transformers fallback
_FAST_MODEL = "BAAI/bge-small-en-v1.5" # fastembed ONNX model (~130MB, ~150ms load)
# Source snippet helper
def _read_source_snippet(
    node: "Node",
    repo_root: str,
    file_cache: "dict[str, list[str]] | None" = None,
) -> list[str]:
    """Read the full node body from source. Uses file_cache to avoid re-reading."""
    try:
        rel = node.file_path
        if file_cache is not None and rel in file_cache:
            lines = file_cache[rel]
        else:
            lines = (Path(repo_root) / rel).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if file_cache is not None:
                file_cache[rel] = lines
        start = max(0, node.line_start - 1)
        end   = min(len(lines), node.line_end)   # full body, not just 8 lines
        return lines[start:end]
    except OSError:
        return []
# Embedding text builder
def build_embed_text(
    node: "Node",
    callee_names: list[str],
    source_lines: list[str] | None = None,
) -> str:
    """
    Build a rich text string for embedding from available node metadata.
    callee_names:  names of functions this node calls (from graph edges).
    source_lines:  raw source lines from the node's definition (optional).
    """
    parts = []

    # Feature area from folder name (e.g. "comments" from "comments/routes.py")
    folder = Path(node.file_path).parts[0] if node.file_path else ""
    if folder and folder not in (".", "src"):
        parts.append(folder)

    # Node type + name
    parts.append(f"{node.node_type.value} {node.name}")

    # Signature — parameter names are rich semantic signal
    if node.signature:
        sig = re.sub(r":[^,)=]+", "", node.signature)   # remove type hints
        sig = re.sub(r"=\s*[^,)]+", "", sig)             # remove defaults
        sig = re.sub(r"[(),]", " ", sig).strip()
        if sig:
            parts.append(sig)

    # Callee names — what it calls tells you what it does
    if callee_names:
        filtered = [n for n in callee_names
                    if len(n) > 3 and n not in ("self", "cls", "super",
                                                  "print", "len", "str", "int")]
        if filtered:
            parts.append(" ".join(filtered[:8]))

    # Source snippet — most important for classes/configs with no callees
    if source_lines:
        snippet = " ".join(line.strip() for line in source_lines if line.strip())
        # Strip common Python noise to keep tokens meaningful
        snippet = re.sub(r"#.*", "", snippet)          # remove comments
        snippet = re.sub(r'""".*?"""', "", snippet)    # remove docstrings
        snippet = re.sub(r"'''.*?'''", "", snippet)
        snippet = snippet.strip()
        if snippet:
            parts.append(snippet[:300])  # cap to avoid overloading the embedding

    return "  ".join(parts)
# Embedder — tries fastembed (ONNX, fast) then falls back to sentence-transformers
class Embedder:
    """
    Embedding backend with automatic fast/fallback selection.

    Priority:
      1. fastembed  — ONNX Runtime, ~150ms load, no PyTorch required
      2. sentence-transformers — PyTorch, ~2s load, wider model selection

    Install the fast backend:  pip install fastembed
    Install the slow fallback: pip install sentence-transformers
    """

    def __init__(self) -> None:
        self._backend: str | None = None   # "fastembed" | "sentence_transformers" | None
        self._model = None

    def is_available(self) -> bool:
        try:
            import fastembed  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import sentence_transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _load(self) -> None:
        if self._model is not None:
            return

        # Try fastembed first (ONNX — fast)
        try:
            import logging
            logging.getLogger("fastembed").setLevel(logging.ERROR)
            from fastembed import TextEmbedding
            self._model = TextEmbedding(model_name=_FAST_MODEL, show_progress_bar=False)
            self._backend = "fastembed"
            return
        except ImportError:
            pass

        # Fall back to sentence-transformers (PyTorch — slower)
        import logging
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(_MODEL_NAME)
        self._backend = "sentence_transformers"

    def embed(self, text: str) -> list[float]:
        self._load()
        if self._backend == "fastembed":
            return next(self._model.embed([text])).tolist()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load()
        if self._backend == "fastembed":
            return [v.tolist() for v in self._model.embed(texts)]
        import numpy as np
        vecs = self._model.encode(
            texts, normalize_embeddings=True,
            batch_size=64, show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]

# Singleton — one model load per process
_embedder: Embedder | None = None

def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
# Graph embedding — called after full index build
def embed_graph(
    graph: "GraphDB",
    embedder: Embedder | None = None,
    repo_root: str | None = None,
) -> int:
    """
    Embed every node in *graph* and store vectors in node_embeddings table.
    Returns number of nodes embedded.
    Skips nodes that already have an up-to-date embedding.

    repo_root: path to the repo being indexed. When provided, source lines are
               read and included in the embedding text for richer semantic signal
               (especially useful for config classes, constants, and settings).
    """
    from jcode.domain.models import EdgeType

    if embedder is None:
        embedder = get_embedder()

    if not embedder.is_available():
        return 0

    # Resolve repo_root from graph snapshot if not supplied
    if repo_root is None:
        snap = graph.get_snapshot()
        if snap:
            repo_root = snap.root_path

    CALL_TYPES = {EdgeType.CALLS, "depends"}

    nodes = graph.all_nodes()
    already = graph.embeddings_count()

    # Only embed if we have new/changed nodes
    if already == len(nodes):
        return 0

    # Build texts — share a file cache so each file is read exactly once
    file_cache: dict[str, list[str]] = {}
    texts = []
    for node in nodes:
        callees = [
            graph.get_node(e.target_id).name
            for e in graph.successors(node.id)
            if e.edge_type in CALL_TYPES
            and graph.get_node(e.target_id) is not None
        ]
        source_lines = (
            _read_source_snippet(node, repo_root, file_cache) if repo_root else None
        )
        texts.append(build_embed_text(node, callees, source_lines))

    # Batch embed
    vectors = embedder.embed_batch(texts)

    # Store
    for node, vector in zip(nodes, vectors, strict=False):
        graph.put_embedding(node.id, vector)

    return len(nodes)
