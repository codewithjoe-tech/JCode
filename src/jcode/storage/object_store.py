"""
Git-style content-addressable object store.

Layout on disk (mirrors .git/objects/):

    .jcode/
      objects/
        ab/
          cdef1234...   ← msgpack-serialised Node, named by SHA-256(content)

Writes are atomic: we serialise to a temp file then rename into place,
so a crash mid-write never leaves a corrupt object.
"""


import hashlib
import os
import tempfile
from pathlib import Path

from jcode.domain.models import Node, NodeId, NodeType
from jcode.domain.ports import ObjectStorePort


# Serialisation helpers  (keep format concerns inside this module)
def _node_to_dict(node: Node) -> dict:
    return {
        "id": node.id.hex,
        "node_type": node.node_type.value,
        "name": node.name,
        "title": node.title,
        "file_path": node.file_path,
        "line_start": node.line_start,
        "line_end": node.line_end,
        "signature": node.signature,
    }

def _dict_to_node(d: dict) -> Node:
    return Node(
        id=NodeId(d["id"]),
        node_type=NodeType(d["node_type"]),
        name=d["name"],
        title=d["title"],
        file_path=d["file_path"],
        line_start=d["line_start"],
        line_end=d["line_end"],
        signature=d.get("signature", ""),
    )

def _content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def node_id_for(node: Node) -> NodeId:
    """
    Deterministic id: hash the canonical fields that define identity.
    Changing *title* or *signature* does NOT change identity (those are mutable
    metadata).  Changing *name*, *file_path*, or *line_start* does.
    """
    key = f"{node.node_type.value}:{node.file_path}:{node.name}:{node.line_start}"
    return NodeId(hashlib.sha256(key.encode()).hexdigest())
# ObjectStore implementation
class ObjectStore(ObjectStorePort):
    """
    Single responsibility: persist and retrieve Node objects by content hash.
    Callers should use `node_id_for` to compute the id before calling `put`.
    """

    def __init__(self, jcode_dir: Path) -> None:
        self._objects_dir = jcode_dir / "objects"
        self._objects_dir.mkdir(parents=True, exist_ok=True)
    # Internal path helpers
    def _object_path(self, node_id: NodeId) -> Path:
        bucket = self._objects_dir / node_id.prefix
        bucket.mkdir(exist_ok=True)
        return bucket / node_id.suffix
    # ObjectStorePort implementation
    def put(self, node: Node) -> NodeId:
        node_id = node_id_for(node)
        path = self._object_path(node_id)
        if path.exists():
            return node_id  # idempotent — same content already stored

        import msgpack
        payload = msgpack.packb(_node_to_dict(node), use_bin_type=True)

        # Atomic write: temp → rename
        tmp_fd, tmp_path = tempfile.mkstemp(dir=self._objects_dir)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                f.write(payload)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

        return node_id

    def get(self, node_id: NodeId) -> Node:
        path = self._object_path(node_id)
        if not path.exists():
            raise KeyError(f"Object not found: {node_id}")
        import msgpack
        raw = path.read_bytes()
        return _dict_to_node(msgpack.unpackb(raw, raw=False))

    def exists(self, node_id: NodeId) -> bool:
        return self._