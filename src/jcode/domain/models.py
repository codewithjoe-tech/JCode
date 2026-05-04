"""
Domain models - pure dataclasses, no external dependencies.
Nothing in this module imports from outside the stdlib.
"""
from dataclasses import dataclass, field
from enum import Enum


class NodeType(str, Enum):
    MODULE   = "module"
    CLASS    = "class"
    FUNCTION = "function"
    METHOD   = "method"
    IMPORT   = "import"
    VARIABLE = "variable"   # class attributes + module-level constants/globals

class EdgeType(str, Enum):
    DEFINES    = "defines"      # module -> function/class it defines
    CALLS      = "calls"        # function -> function it calls
    IMPORTS    = "imports"      # module -> module it imports
    CONTAINS   = "contains"     # class -> method it contains
    INHERITS   = "inherits"     # class -> parent class
    REFERENCES = "references"   # function -> variable it reads (attribute access)

@dataclass(frozen=True, slots=True)
class NodeId:
    """SHA-256 hex string — content-addressable like a git blob hash."""
    hex: str

    def __str__(self) -> str:
        return self.hex

    @property
    def prefix(self) -> str:
        return self.hex[:2]

    @property
    def suffix(self) -> str:
        return self.hex[2:]

@dataclass(frozen=True, slots=True)
class Node:
    id: NodeId
    node_type: NodeType
    name: str
    title: str
    file_path: str
    line_start: int
    line_end: int
    signature: str = ""
    keywords: str = ""   # space-separated tokens extracted post-index (e.g. URL domains)

@dataclass(frozen=True, slots=True)
class Edge:
    source_id: NodeId
    target_id: NodeId
    edge_type: str

    def __post_init__(self):
        if hasattr(self.edge_type, "value"):
            object.__setattr__(self, "edge_type", self.edge_type.value)

@dataclass(slots=True)
class IndexSnapshot:
    snapshot_hash: str
    indexed_at: float
    file_count: int
    node_count: int
    edge_count: int
    root_path: str

@dataclass(slots=True)
class TraversalResult:
    entry_node: Node
    nodes: list = field(default_factory=list)
    edges: list = field(default_factory=list)
    depth_reached: int = 0
    external_deps: list = field(default_factory=list)  # nodes outside scope boundary

@dataclass(slots=True)
class BlastRadiusResult:
    changed_node: Node
    affected_nodes: list = field(default_factory=list)
    affected_edges: list = field(default_factory=list)
    confidence: float = 1.0
    risk_reasons: list = field(default_factory=list)
