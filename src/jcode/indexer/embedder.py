"""
Embedder — builds text from nodes and embeds with sentence-transformers.

Falls back gracefully if sentence-transformers is not installed:
  - search_semantic() won't be called (MCP server checks has_embeddings())
  - FTS5 keyword search is used instead

Text built from (priority order, use what's available):
  1. docstring   — first string literal in function/class body
  2. comments    — # lines immediately above the definition
  3. signature   — parameter names carry strong semantic signal
  4. name        — always present
  5. file_path   — folder name tells you the feature area
  6. callee names — what a function calls describes what it does
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jcode.domain.models import Node
    from jcode.storage.graph_db import GraphDB

_MODEL_NAME = "all-MiniLM-L6-v2"   # 80MB, 384-dim, fast on CPU
# Embedding text builder
def build_embed_text(node: "Node", callee_names: list[str]) -> str:
    """
    Build a rich text string for embedding from available node metadata.
    callee_names: names of functions this node calls (from graph edges).
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
        # Extract just parameter names, strip types and defaults
        sig = re.sub(r":[^,)=]+", "", node.signature)   # remove type hints
        sig = re.sub(r"=\s*[^,)]+", "", sig)             # remove defaults
        sig = re.sub(r"[(),]", " ", sig).strip()
        if sig:
            parts.append(sig)

    # Callee names — what it calls tells you what it does
    if callee_names:
        # Filter out builtins and very generic names
        filtered = [n for n in callee_names
                    if len(n) > 3 and n not in ("self", "cls", "super",
                                                  "print", "len", "str", "int")]
        if filtered:
            parts.append(" ".join(filtered[:8]))  # cap at 8 callees

    return "  ".join(parts)
# Sentence-transformers wrapper
class Embedder:
    """Thin wrapper around sentence-transformers. Lazy-loads the model."""

    def __init__(self, model_name: str = _MODEL_NAME):
        self._model_name = model_name
        self._model = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import sentence_transformers  # noqa: F401
                self._available = True
            except ImportError:
                self._available = False
        return self._available

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, text: str) -> list[float]:
        self._load()
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True,
                                   batch_size=64, show_progress_bar=False)
        return [v.tolist() for v in vecs]

# Singleton — shared across the process
_embedder = Embedder()

def get_embedder() -> Embedder:
    return _embedder
# Graph embedding — called after full index build
def embed_graph(graph: "GraphDB", embedder: Embedder | None = None) -> int:
    """
    Embed every node in *graph* and store vectors in node_embeddings table.
    Returns number of nodes embedded.
    Skips nodes that already have an up-to-date embedding.
    """
    from jcode.domain.models import EdgeType

    if embedder is None:
        embedder = get_embedder()

    if not embedder.is_available():
        return 0

    CALL_TYPES = {EdgeType.CALLS, "depends"}

    nodes = graph.all_nodes()
    already = graph.embeddings_count()

    # Only embed if we have new/changed nodes
    if already == len(nodes):
        return 0

    # Build texts
    texts = []
    for node in nodes:
        callees = [
            graph.get_node(e.target_id).name
            for e in graph.successors(node.id)
            if e.edge_type in CALL_TYPES
            and graph.get_node(e.target_id) is not None
        ]
        texts.append(build_embed_text(node, callees))

    # Batch embed
    vectors = embedder.embed_batch(texts)

    # Store
    for node, vector in zip(nodes, vectors, strict=False):
        graph.put_embedding(node.id, vector)

    return len(nodes)
