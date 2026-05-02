"""
FastAPI edge plugin.

Detects two patterns:

1. Depends(fn) / Security(fn) in function parameter defaults
   → emits a DEPENDS edge from the route handler to fn

2. @router.get / @router.post / @app.get etc. decorator
   → no extra edge needed (the function IS the handler; blast radius
     from decode_token will reach it via the DEPENDS chain)

Because `Depends(get_current_user)` is a call node inside the
*parameters* of a route function, _collect_calls already visits it.
This plugin intercepts the "Depends" callee name and instead of a
generic CALLS edge emits a typed DEPENDS edge pointing at the
*argument* (the actual dependency function), not at Depends itself.
"""
from jcode.domain.models import Edge, Node, NodeId, NodeType
from jcode.storage.object_store import node_id_for

EDGE_DEPENDS = "depends"

# Names treated as dependency wrappers — argument is the real dependency
_WRAPPER_NAMES = frozenset({"Depends", "Security", "Annotated"})

def _node_text(ts_node, source: bytes) -> str:
    return source[ts_node.start_byte:ts_node.end_byte].decode("utf-8", errors="replace")

def _build_provisional(name: str) -> Node:
    ph = Node(
        id=NodeId("0" * 64), node_type=NodeType.FUNCTION,
        name=name, title=name, file_path="<unresolved>",
        line_start=0, line_end=0,
    )
    return Node(
        id=node_id_for(ph), node_type=NodeType.FUNCTION,
        name=name, title=name, file_path="<unresolved>",
        line_start=0, line_end=0,
    )

class FastAPIPlugin:
    """Implements EdgePlugin protocol for FastAPI dependency injection."""

    @property
    def handled_names(self) -> frozenset:
        return _WRAPPER_NAMES

    def handle_call(self, call_node, source: bytes, caller: Node):
        """
        For Depends(some_fn) emit:
          DEPENDS edge: caller -> some_fn  (the injected dependency)

        We look at the first positional argument of the call.
        Handles both `Depends(fn)` and `Depends(dependency=fn)`.
        """
        arg_list = next(
            (c for c in call_node.children if c.type == "argument_list"), None
        )
        if arg_list is None:
            return [], []

        for arg in arg_list.children:
            # positional identifier:  Depends(get_current_user)
            if arg.type == "identifier":
                dep_name = _node_text(arg, source)
                prov = _build_provisional(dep_name)
                return [prov], [Edge(
                    source_id=caller.id,
                    target_id=prov.id,
                    edge_type=EDGE_DEPENDS,
                )]
            # keyword argument:  Depends(dependency=get_current_user)
            if arg.type == "keyword_argument":
                for child in arg.children:
                    if child.type == "identifier":
                        dep_name = _node_text(child, source)
                        # skip the keyword name itself ("dependency")
                        # take the value (last identifier)
                        pass
                # re-walk for value
                children = list(arg.children)
                # keyword_argument: identifier '=' expression
                if len(children) >= 3 and children[2].type == "identifier":
                    dep_name = _node_text(children[2], source)
                    prov = _build_provisional(dep_name)
                    return [prov], [Edge(
                        source_id=caller.id,
                        target_id=prov.id,
                        edge_type=EDGE_DEPENDS,
                    )]

        return [], []

def create() -> FastAPIPlugin:
    return FastAPIPlugin()
