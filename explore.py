"""
jcode explorer — run this against any Python project.

Usage:
  python explore.py <repo_path>
  python explore.py <repo_path> <function_name>          # one-shot
  python explore.py <repo_path> --index-dir C:/tmp/idx   # custom index location
"""
import sys, argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("repo", help="Path to your project root")
    p.add_argument("name", nargs="?", help="Function/class name for one-shot lookup")
    p.add_argument("--index-dir", default=None,
                   help="Where to store the .jcode index (default: <repo>/.jcode)")
    p.add_argument("--reindex", action="store_true", help="Force a full re-index")
    return p.parse_args()


def setup(repo_path: str, index_dir: str | None, reindex: bool):
    from jcode.indexer.plugins import build_parser
    from jcode.indexer.builder import Indexer
    from jcode.storage.object_store import ObjectStore
    from jcode.storage.graph_db import GraphDB

    jcode_dir = Path(index_dir) if index_dir else Path(repo_path) / ".jcode"
    jcode_dir.mkdir(parents=True, exist_ok=True)

    store = ObjectStore(jcode_dir)
    graph = GraphDB(jcode_dir)
    snap  = graph.get_snapshot()

    if snap is None or reindex:
        print(f"Indexing {repo_path} ...")
        parser  = build_parser(repo_path)
        print(f"  plugins loaded: {[type(p).__name__ for p in parser._plugins] or 'none'}")
        indexer = Indexer(parser, store, graph)
        snap    = indexer.index(repo_path, full_reindex=True)
        print(f"  done — {snap.file_count} files, {snap.node_count} nodes, {snap.edge_count} edges\n")
    else:
        print(f"Using existing index — {snap.node_count} nodes, {snap.edge_count} edges")
        print(f"  (run with --reindex to rebuild)\n")

    return graph


SKIP_EDGE_TYPES = None  # set after import

def _skip():
    from jcode.domain.models import EdgeType
    global SKIP_EDGE_TYPES
    SKIP_EDGE_TYPES = {EdgeType.IMPORTS, EdgeType.DEFINES, EdgeType.CONTAINS}


def show_node(graph, node):
    print(f"\n[{node.node_type.value}]  {node.title}")
    print(f"  file : {node.file_path}:{node.line_start}")
    if node.signature:
        print(f"  sig  : {node.signature}")


def show_edges(graph, node):
    out = [(e, graph.get_node(e.target_id)) for e in graph.successors(node.id)
           if e.edge_type not in SKIP_EDGE_TYPES]
    inc = [(e, graph.get_node(e.source_id)) for e in graph.predecessors(node.id)
           if e.edge_type not in SKIP_EDGE_TYPES]

    if out:
        print(f"\n  CALLS / DEPENDS ON  ({len(out)}):")
        for e, n in out:
            if n:
                print(f"    --[{e.edge_type}]--> {n.title}  ({n.file_path}:{n.line_start})")
    else:
        print("\n  CALLS / DEPENDS ON: none")

    if inc:
        print(f"\n  CALLED BY / DEPENDED ON BY  ({len(inc)}):")
        for e, n in inc:
            if n:
                print(f"    <--[{e.edge_type}]-- {n.title}  ({n.file_path}:{n.line_start})")
    else:
        print("\n  CALLED BY: none")


def show_blast(graph, node):
    from jcode.graph.traversal import GraphTraversal
    b = GraphTraversal(graph).blast_radius(node.id, max_depth=5)
    print(f"\n  BLAST RADIUS — confidence {b.confidence:.0%}")
    for r in b.risk_reasons:
        print(f"    ⚠  {r}")
    if b.affected_nodes:
        print(f"\n  Affected nodes ({len(b.affected_nodes)}):")
        for n in sorted(b.affected_nodes, key=lambda x: (x.node_type.value, x.title)):
            print(f"    [{n.node_type.value:8}] {n.title}")
    else:
        print("  No affected nodes found.")


def pick_node(graph, name, results):
    """Pick the best match — exact name match preferred."""
    exact = [n for n in results if n.name == name]
    return exact[0] if exact else results[0]


def repl(graph):
    from jcode.graph.traversal import search_entry_points
    _skip()
    print("Commands:")
    print("  search <name>   — find nodes by name")
    print("  edges  <name>   — show what it calls and what calls it")
    print("  blast  <name>   — full blast radius + confidence score")
    print("  quit            — exit\n")

    while True:
        try:
            line = input("jcode> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line in ("quit", "exit", "q"):
            break

        parts = line.split(None, 1)
        cmd = parts[0]
        arg = parts[1].strip() if len(parts) > 1 else ""

        if not arg:
            print("  provide a name, e.g:  edges get_current_user")
            continue

        results = search_entry_points(graph, arg, limit=10)
        if not results:
            print(f"  nothing found for '{arg}' — try a partial name")
            continue

        if cmd == "search":
            for n in results:
                print(f"  [{n.node_type.value:8}] {n.title:<55} {n.file_path}:{n.line_start}")

        elif cmd == "edges":
            node = pick_node(graph, arg, results)
            show_node(graph, node)
            show_edges(graph, node)

        elif cmd == "blast":
            node = pick_node(graph, arg, results)
            show_node(graph, node)
            show_blast(graph, node)

        else:
            print("  unknown command — use search / edges / blast / quit")


def one_shot(graph, name):
    from jcode.graph.traversal import search_entry_points
    _skip()
    results = search_entry_points(graph, name, limit=10)
    if not results:
        print(f"Nothing found for '{name}'")
        return
    node = pick_node(graph, name, results)
    show_node(graph, node)
    show_edges(graph, node)
    show_blast(graph, node)


if __name__ == "__main__":
    # Make sure jcode src is on the path if running from the jcode folder
    here = Path(__file__).parent
    src  = here / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))

    args  = parse_args()
    graph = setup(args.repo, args.index_dir, args.reindex)

    if args.name:
        one_shot(graph, args.name)
    else:
        repl(graph)
