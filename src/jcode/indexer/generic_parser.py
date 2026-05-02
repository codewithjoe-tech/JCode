"""
GenericParser — single tree-sitter parser for all supported languages.

Language knowledge lives in .scm query files (one per language).
Adding a new language = install grammar + drop a .scm file + add a LangConfig.
No Python parser code changes ever needed.

Three-layer architecture:
  .scm query files   ← what syntax constructs to capture (per language, ~10 lines each)
  GenericParser      ← single walker that interprets captures for all languages
  Plugin system      ← framework semantic edges (FastAPI Depends, SQLAlchemy, etc.)
"""
import importlib
import os
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Parser, Query, QueryCursor

from jcode.domain.models import Edge, EdgeType, Node, NodeId, NodeType
from jcode.storage.object_store import node_id_for

QUERIES_DIR = Path(__file__).parent / "queries"

@dataclass(frozen=True)
class LangConfig:
    name: str
    module_name: str          # pip package name, e.g. "tree_sitter_python"
    query_file: str           # filename inside queries/, e.g. "python.scm"
    extensions: frozenset     # file extensions handled, e.g. frozenset({".py"})
    func_types: frozenset     # AST node types that represent functions/methods
    class_types: frozenset    # AST node types that represent classes/structs
    root_types: frozenset     # AST node types that mark the top-level scope boundary

_CONFIGS: list[LangConfig] = [
    LangConfig(
        name="python", module_name="tree_sitter_python", query_file="python.scm",
        extensions=frozenset({".py"}),
        func_types=frozenset({"function_definition"}),
        class_types=frozenset({"class_definition"}),
        root_types=frozenset({"module"}),
    ),
    LangConfig(
        name="typescript", module_name="tree_sitter_typescript", query_file="typescript.scm",
        extensions=frozenset({".ts", ".tsx"}),
        func_types=frozenset({"function_declaration", "method_definition", "function",
                               "generator_function_declaration", "arrow_function"}),
        class_types=frozenset({"class_declaration", "interface_declaration"}),
        root_types=frozenset({"program"}),
    ),
    LangConfig(
        name="javascript", module_name="tree_sitter_javascript", query_file="javascript.scm",
        extensions=frozenset({".js", ".jsx", ".mjs"}),
        func_types=frozenset({"function_declaration", "method_definition", "function",
                               "generator_function_declaration", "arrow_function"}),
        class_types=frozenset({"class_declaration"}),
        root_types=frozenset({"program"}),
    ),
    LangConfig(
        name="go", module_name="tree_sitter_go", query_file="go.scm",
        extensions=frozenset({".go"}),
        func_types=frozenset({"function_declaration", "method_declaration"}),
        class_types=frozenset({"type_declaration"}),
        root_types=frozenset({"source_file"}),
    ),
    LangConfig(
        name="rust", module_name="tree_sitter_rust", query_file="rust.scm",
        extensions=frozenset({".rs"}),
        func_types=frozenset({"function_item", "closure_expression"}),
        class_types=frozenset({"impl_item", "struct_item", "trait_item", "enum_item"}),
        root_types=frozenset({"source_file"}),
    ),
]

_EXT_TO_CONFIG: dict[str, LangConfig] = {
    ext: cfg for cfg in _CONFIGS for ext in cfg.extensions
}
# Node helpers
def _rel_path(file_path: str, repo_root: str) -> str:
    try:
        return os.path.relpath(file_path, repo_root)
    except ValueError:
        return file_path

def _make_title(parts: list) -> str:
    return ".".join(p for p in parts if p)

def _module_name(rel: str) -> str:
    p = rel.replace(os.sep, ".").replace("/", ".")
    for ext in (".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".go", ".rs"):
        if p.endswith(ext):
            return p[: -len(ext)]
    return p

def _build_node(node_type, name, title, file_path, line_start, line_end, signature=""):
    placeholder = Node(
        id=NodeId("0" * 64), node_type=node_type, name=name, title=title,
        file_path=file_path, line_start=line_start, line_end=line_end, signature=signature,
    )
    return Node(
        id=node_id_for(placeholder), node_type=node_type, name=name, title=title,
        file_path=file_path, line_start=line_start, line_end=line_end, signature=signature,
    )

def _node_text(ts_node, source: bytes) -> str:
    return source[ts_node.start_byte:ts_node.end_byte].decode("utf-8", errors="replace")

def _get_node_name(ts_node, source: bytes) -> str:
    """Extract the identifier name from a definition node via field 'name' or first identifier."""
    name_child = ts_node.child_by_field_name("name")
    if name_child is not None:
        return _node_text(name_child, source)
    for child in ts_node.children:
        if child.type in ("identifier", "field_identifier", "property_identifier"):
            return _node_text(child, source)
    return ""

def _get_span_node(ts_node):
    """Use the decorated_definition parent for line spans when the node is decorated."""
    parent = ts_node.parent
    if parent is not None and parent.type == "decorated_definition":
        return parent
    return ts_node

def _get_scope_parts(ts_node, module_name: str, source: bytes, cfg: LangConfig) -> list[str]:
    """Walk up the parent chain to build the dotted scope prefix (excludes own name)."""
    parts = []
    node = ts_node.parent
    while node is not None and node.type not in cfg.root_types:
        if node.type in cfg.func_types or node.type in cfg.class_types:
            name = _get_node_name(node, source)
            if name:
                parts.append(name)
        node = node.parent
    parts.reverse()
    return [module_name] + parts

def _is_method(ts_node, cfg: LangConfig) -> bool:
    """True if the function lives directly inside a class body (not a nested function)."""
    node = ts_node.parent
    while node is not None and node.type not in cfg.root_types:
        if node.type in cfg.class_types:
            return True
        if node.type in cfg.func_types:
            return False  # Hit a function boundary before a class
        node = node.parent
    return False

def _find_caller(ts_node, cfg: LangConfig):
    """Return the nearest enclosing function/method definition TSNode, or None."""
    node = ts_node.parent
    while node is not None and node.type not in cfg.root_types:
        if node.type in cfg.func_types:
            return node
        node = node.parent
    return None

def _get_call_callee(call_node, source: bytes) -> str:
    """Extract the final callee name from a call/call_expression node."""
    if not call_node.children:
        return ""
    func_expr = call_node.children[0]
    t = func_expr.type
    if t in ("identifier", "field_identifier", "property_identifier"):
        return _node_text(func_expr, source)
    if t == "attribute":
        # Python: obj.method() — take last identifier child
        last = ""
        for child in func_expr.children:
            if child.type in ("identifier", "property_identifier"):
                last = _node_text(child, source)
        return last
    if t == "member_expression":
        # TypeScript/JS: obj.method()
        prop = func_expr.child_by_field_name("property")
        return _node_text(prop, source) if prop else ""
    if t == "selector_expression":
        # Go: pkg.Func()
        field = func_expr.child_by_field_name("field")
        return _node_text(field, source) if field else ""
    return ""
# GenericParser
class GenericParser:
    """
    Single tree-sitter parser for all supported languages.
    Language knowledge lives in .scm query files — no Python code changes to add a language.
    """

    def __init__(self, plugins: list | None = None):
        self._plugins: list = plugins or []
        self._plugin_map: dict[str, object] = {}
        for p in self._plugins:
            for name in p.handled_names:
                self._plugin_map[name] = p
        # Lazy cache: ext -> (LangConfig, Language, Query, Parser) | None
        self._cache: dict[str, tuple | None] = {}

    @property
    def language(self) -> str:
        return "generic"

    @property
    def file_extensions(self) -> frozenset:
        available = frozenset()
        for cfg in _CONFIGS:
            ext = next(iter(cfg.extensions))
            if self._load_lang(ext) is not None:
                available |= cfg.extensions
        return available

    def _load_lang(self, ext: str) -> tuple | None:
        if ext in self._cache:
            return self._cache[ext]
        cfg = _EXT_TO_CONFIG.get(ext)
        if cfg is None:
            self._cache[ext] = None
            return None
        try:
            mod = importlib.import_module(cfg.module_name)
            lang = Language(mod.language())
            query_text = (QUERIES_DIR / cfg.query_file).read_text()
            query = Query(lang, query_text)
            parser = Parser(lang)
            result: tuple | None = (cfg, lang, query, parser)
        except Exception:
            result = None
        self._cache[ext] = result
        return result

    def parse_file(self, file_path: str, source: str, repo_root: str):
        ext = Path(file_path).suffix.lower()
        loaded = self._load_lang(ext)
        if loaded is None:
            return [], []

        cfg, lang, query, parser = loaded
        src = source.encode("utf-8")
        tree = parser.parse(src)
        rel = _rel_path(file_path, repo_root)
        mod_name = _module_name(rel)

        nodes: list = []
        edges: list = []

        mod_node = _build_node(NodeType.MODULE, mod_name, mod_name, rel,
                                1, tree.root_node.end_point[0] + 1)
        nodes.append(mod_node)

        # Run all query patterns
        cursor = QueryCursor(query)
        all_matches = list(cursor.matches(tree.root_node))

        # fn_node_map: ts_node.id -> jcode Node (built in pass 1, consumed in pass 2)
        fn_node_map: dict[int, Node] = {}
        # Pass 1: definitions (functions, classes, imports, inheritance)
        for _, capture_dict in all_matches:
            # Functions / Methods
            for ts_fn in capture_dict.get("function", []):
                span = _get_span_node(ts_fn)
                name = _get_node_name(ts_fn, src)
                if not name:
                    continue
                scope = _get_scope_parts(ts_fn, mod_name, src, cfg)
                title = _make_title(scope + [name])
                is_m = _is_method(ts_fn, cfg)
                sig_node = ts_fn.child_by_field_name("parameters")
                sig = _node_text(sig_node, src) if sig_node else ""
                ntype = NodeType.METHOD if is_m else NodeType.FUNCTION
                node = _build_node(ntype, name, title, rel,
                                   span.start_point[0] + 1, span.end_point[0] + 1,
                                   signature=sig)
                nodes.append(node)
                fn_node_map[ts_fn.id] = node
                etype = EdgeType.CONTAINS if is_m else EdgeType.DEFINES
                parent_jnode = self._find_parent_jnode(ts_fn, fn_node_map, mod_node)
                edges.append(Edge(source_id=parent_jnode.id, target_id=node.id, edge_type=etype))

            # Classes
            for ts_cls in capture_dict.get("class", []):
                span = _get_span_node(ts_cls)
                name = _get_node_name(ts_cls, src)
                if not name:
                    continue
                scope = _get_scope_parts(ts_cls, mod_name, src, cfg)
                title = _make_title(scope + [name])
                node = _build_node(NodeType.CLASS, name, title, rel,
                                   span.start_point[0] + 1, span.end_point[0] + 1)
                nodes.append(node)
                fn_node_map[ts_cls.id] = node
                edges.append(Edge(source_id=mod_node.id, target_id=node.id,
                                  edge_type=EdgeType.DEFINES))

            # Imports
            for ts_imp in capture_dict.get("import", []):
                text = _node_text(ts_imp, src).strip()
                imp = _build_node(NodeType.IMPORT, text, text, rel,
                                  ts_imp.start_point[0] + 1, ts_imp.end_point[0] + 1)
                nodes.append(imp)
                edges.append(Edge(source_id=mod_node.id, target_id=imp.id,
                                  edge_type=EdgeType.IMPORTS))

            # Inheritance bases
            for ts_base in capture_dict.get("base", []):
                base_name = _node_text(ts_base, src)
                if not base_name:
                    continue
                # Walk up from the base identifier to find the class_definition
                cls_ts = ts_base.parent  # argument_list
                if cls_ts is not None:
                    cls_ts = cls_ts.parent  # class_definition
                cls_jnode = fn_node_map.get(cls_ts.id) if cls_ts is not None else None
                if cls_jnode:
                    prov = _build_node(NodeType.CLASS, base_name, base_name, rel, 0, 0)
                    nodes.append(prov)
                    edges.append(Edge(source_id=cls_jnode.id, target_id=prov.id,
                                      edge_type=EdgeType.INHERITS))
        # Pass 2: calls (fn_node_map is now fully populated)
        for _, capture_dict in all_matches:
            for ts_call in capture_dict.get("call", []):
                callee_name = _get_call_callee(ts_call, src)
                if not callee_name:
                    continue
                caller_ts = _find_caller(ts_call, cfg)
                caller_jnode = fn_node_map.get(caller_ts.id) if caller_ts is not None else None
                # For plugin-handled calls, fall back to module node when there
                # is no enclosing function (e.g. urlpatterns = [...] at module level,
                # class-body field definitions like ForeignKey, decorators).
                plugin = self._plugin_map.get(callee_name)
                if plugin:
                    effective_caller = caller_jnode or mod_node
                    p_nodes, p_edges = plugin.handle_call(ts_call, src, effective_caller)
                    nodes.extend(p_nodes)
                    edges.extend(p_edges)
                else:
                    if caller_jnode is None:
                        continue
                    prov = _build_node(NodeType.FUNCTION, callee_name, callee_name,
                                       "<unresolved>", 0, 0)
                    nodes.append(prov)
                    edges.append(Edge(source_id=caller_jnode.id, target_id=prov.id,
                                      edge_type=EdgeType.CALLS))

        return nodes, edges

    def _find_parent_jnode(self, ts_node, fn_node_map: dict, mod_node: Node) -> Node:
        """Walk up the parent chain to find the nearest enclosing jcode Node."""
        node = ts_node.parent
        while node is not None:
            jnode = fn_node_map.get(node.id)
            if jnode is not None:
                return jnode
            node = node.parent
        return mod_node
