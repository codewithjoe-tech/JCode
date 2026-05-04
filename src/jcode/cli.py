"""
jcode CLI — git-style command interface.

Commands
--------
jcode init       <repo>   Initialise a .jcode store + write CLAUDE.md
jcode index      <repo>   Index / re-index a repo
jcode status     <repo>   Show snapshot info
jcode search     <query>  Search node titles
jcode setup-mcp  <repo>   Print the claude mcp add command to run
jcode serve               Start the MCP server (stdio transport)
jcode plugins             List available plugins
jcode add        <name>   Install a plugin from the jcode registry
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import click

from jcode.graph.traversal import search_entry_points, search_semantic
from jcode.indexer.builder import Indexer
from jcode.indexer.generic_parser import GenericParser
from jcode.storage.graph_db import GraphDB
from jcode.storage.object_store import ObjectStore

# CLAUDE.md template — tells the agent to use jcode before grepping
_CLAUDE_MD = """\
# Code Intelligence — jcode

This repository is indexed with **jcode** (feature-graph code intelligence).
The `.jcode/` directory contains a content-addressable node store and an
SQLite graph of every function, class, method, and module, with typed edges
(CALLS, IMPORTS, CONTAINS, INHERITS) plus plugin-defined types like "depends".

## Workflow — follow this order every time

1. **Orient** — call `jcode_feature_map()` first.
   It shows the folder/feature layout in one call. Use it to pick the right
   `scope` before searching.

2. **Search** — call `jcode_search(query, scope=<folder>)`.
   This uses semantic vector search (or FTS5 fallback). Never run grep to
   find a function — use this instead.

3. **Get context** — call `jcode_context(node_id, scope=<folder>)`.
   DFS-forward from the entry point. Nodes outside the scope are flagged as
   `external_deps` — follow them only if needed.

4. **Check blast radius** — call `jcode_blast_radius(node_id)` **before**
   editing any function. A confidence score < 0.8 means callers outside the
   current scope will break — warn the user first.

5. **Read targeted files only** — after jcode tells you the exact file and
   line range, read only that file. Do not read files to understand structure.

## Rules

- NEVER use grep/ripgrep to locate a function, class, or feature. Use `jcode_search`.
- NEVER read a file just to understand its structure. Use `jcode_context`.
- ALWAYS call `jcode_blast_radius` before editing a function.
- Pass `scope=<folder>` when the user's request clearly names a feature area
  (e.g. "in the comments module", "fix the auth flow").
- If scoped search returns nothing, jcode automatically falls back to the
  full graph — you do not need to retry manually.

## When grep IS the right tool

Use grep (or ripgrep `rg`) directly — without going through jcode — when you
are looking for an **exact string literal** that does not correspond to a
code symbol. jcode indexes identifiers and call graphs; it does not index
arbitrary string values inside code.

Good grep targets (jcode will NOT find these reliably):
- A URL or base URL string:  `rg "mymoneybazaar.com"`
- A hard-coded API key name: `rg "X-Api-Key"`
- A Django URL pattern:      `rg "path.*login"`
- A specific error message:  `rg "Invalid OTP"`
- A config value or secret:  `rg "REDIS_HOST"`

Use jcode for everything else — structure, behaviour, and relationships.

## Tool reference

| Tool | When to call |
|------|-------------|
| `jcode_feature_map()` | Start of every task — orient yourself |
| `jcode_search(query, scope?)` | Find the entry-point node |
| `jcode_context(node_id, scope?)` | Understand what a node calls and is called by |
| `jcode_blast_radius(node_id)` | Before any edit — see who breaks |
| `jcode_index(repo_path)` | After large refactors — refresh the graph |
"""

_BUILTIN_PLUGINS: dict[str, str] = {}
# All plugins are separate packages — install via: jcode add <name>


# Helpers
def _resolve_jcode_dir(repo: str | None, jcode_dir: str | None) -> Path:
    base = Path(repo) if repo else Path.cwd()
    return Path(jcode_dir) if jcode_dir else base / ".jcode"

def _open_stores(jcode_dir: Path) -> tuple[ObjectStore, GraphDB]:
    if not jcode_dir.exists():
        raise click.ClickException(
            f"No .jcode store at '{jcode_dir}'. Run `jcode init <repo>` first."
        )
    return ObjectStore(jcode_dir), GraphDB(jcode_dir)

def _write_claude_md(repo: Path) -> None:
    """Write jcode instructions into CLAUDE.md.
    Creates the file if it doesn't exist; appends if it does (so existing
    project instructions are preserved and jcode is still discovered).
    """
    claude_md = repo / "CLAUDE.md"
    if claude_md.exists():
        contents = claude_md.read_text(encoding="utf-8")
        if "jcode" in contents:
            click.echo(f"  CLAUDE.md already contains jcode instructions — skipped")
            return
        # Append jcode section to existing file
        separator = "\n" if contents and not contents.endswith("\n") else ""
        claude_md.write_text(contents + separator + "\n" + _CLAUDE_MD, encoding="utf-8")
        click.echo(f"  appended jcode instructions → {claude_md}")
    else:
        claude_md.write_text(_CLAUDE_MD, encoding="utf-8")
        click.echo(f"  wrote CLAUDE.md → {claude_md}")

def _update_gitignore(repo: Path) -> None:
    """Add .jcode to .gitignore. Creates the file if it doesn't exist."""
    gitignore = repo / ".gitignore"
    entry = ".jcode"

    if gitignore.exists():
        contents = gitignore.read_text(encoding="utf-8")
        lines = contents.splitlines()
        if any(line.strip() == entry for line in lines):
            click.echo(f"  .gitignore already contains {entry} — skipped")
            return
        # Append with a trailing newline
        separator = "\n" if contents and not contents.endswith("\n") else ""
        gitignore.write_text(contents + separator + entry + "\n", encoding="utf-8")
        click.echo(f"  added {entry} to .gitignore")
    else:
        gitignore.write_text(entry + "\n", encoding="utf-8")
        click.echo(f"  created .gitignore with {entry}")


# CLI group
@click.group()
@click.version_option(package_name="jcode")
def main() -> None:
    """jcode — feature-graph code intelligence for AI agents."""


# jcode init
@main.command()
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--jcode-dir", default=None, help="Override .jcode location.")
@click.option("--no-claude-md", is_flag=True, default=False,
              help="Skip writing CLAUDE.md.")
def init(repo: str, jcode_dir: str | None, no_claude_md: bool) -> None:
    """Initialise a jcode store and write CLAUDE.md (like git init)."""
    path = _resolve_jcode_dir(repo, jcode_dir)
    if path.exists():
        click.echo(f"Store already exists at {path}")
    else:
        path.mkdir(parents=True)
        (path / "objects").mkdir()
        click.echo(f"Initialised jcode store at {path}")

    repo_path = Path(repo).resolve()
    _update_gitignore(repo_path)
    if not no_claude_md:
        _write_claude_md(repo_path)


# jcode index
@main.command()
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--jcode-dir", default=None, help="Override .jcode location.")
@click.option("--full", is_flag=True, default=False,
              help="Drop and rebuild the entire graph.")
@click.option("--workers", "-j", default=4, show_default=True,
              help="Parallel workers for file parsing. Use 1 to disable parallelism.")
def index(repo: str, jcode_dir: str | None, full: bool, workers: int) -> None:
    """Index a repository into the feature graph (incremental by default)."""
    jcode_path = _resolve_jcode_dir(repo, jcode_dir)
    jcode_path.mkdir(parents=True, exist_ok=True)

    store   = ObjectStore(jcode_path)
    graph   = GraphDB(jcode_path)
    parser  = GenericParser()
    indexer = Indexer(parser, store, graph)

    click.echo(f"Indexing {repo} …")
    snapshot = indexer.index(repo, full_reindex=full, workers=workers)

    click.echo(
        f"Done — {snapshot.file_count} files  "
        f"{snapshot.node_count} nodes  "
        f"{snapshot.edge_count} edges  "
        f"[{snapshot.snapshot_hash[:12]}]"
    )


# jcode status
@main.command()
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--jcode-dir", default=None, help="Override .jcode location.")
def status(repo: str, jcode_dir: str | None) -> None:
    """Show the current index snapshot (like git log -1)."""
    jcode_path = _resolve_jcode_dir(repo, jcode_dir)
    _, graph = _open_stores(jcode_path)

    snap = graph.get_snapshot()
    if snap is None:
        click.echo("No index snapshot found. Run `jcode index` first.")
        return

    dt = datetime.fromtimestamp(snap.indexed_at).strftime("%Y-%m-%d %H:%M:%S")
    click.echo(f"snapshot  {snap.snapshot_hash[:16]}")
    click.echo(f"indexed   {dt}")
    click.echo(f"root      {snap.root_path}")
    click.echo(f"files     {snap.file_count}")
    click.echo(f"nodes     {snap.node_count}")
    click.echo(f"edges     {snap.edge_count}")


# jcode search
@main.command()
@click.argument("query")
@click.option("--repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--jcode-dir", default=None, help="Override .jcode location.")
@click.option("--limit", default=10, show_default=True, help="Max results.")
def search(query: str, repo: str, jcode_dir: str | None, limit: int) -> None:
    """Search node titles in the feature graph."""
    jcode_path = _resolve_jcode_dir(repo, jcode_dir)
    _, graph = _open_stores(jcode_path)

    results = search_semantic(graph, query, limit=limit)
    if not results:
        click.echo("No results.")
        return

    for node, score in results:
        click.echo(
            f"  [{node.node_type.value:8}]  {node.title:45}  "
            f"{node.file_path}:{node.line_start}  score={score:.2f}  id={node.id.hex[:12]}"
        )


# jcode setup-mcp
@main.command("setup-mcp")
@click.argument("repo", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--jcode-dir", default=None, help="Override .jcode location.")
def setup_mcp(repo: str, jcode_dir: str | None) -> None:
    """Print the claude mcp add command for this repository.

    Run the printed command once to register jcode with Claude Code.
    After that, Claude will use jcode automatically instead of grepping.
    """
    repo_path  = Path(repo).resolve()
    jcode_path = Path(jcode_dir).resolve() if jcode_dir else repo_path / ".jcode"

    if not jcode_path.exists():
        click.echo(
            f"Warning: no .jcode store found at {jcode_path}. "
            "Run `jcode index <repo>` first.",
            err=True,
        )

    click.echo("\nRun this command once to register jcode with Claude Code:\n")
    click.echo(
        f'  claude mcp add jcode jcode serve -e "JCODE_DIR={jcode_path}"\n'
    )
    click.echo("Then open Claude Code in this repo and ask it anything — it will")
    click.echo("call jcode_feature_map() before touching any files.\n")
    md_status = "yes" if (repo_path / "CLAUDE.md").exists() else "no — run: jcode init"
    click.echo(f"CLAUDE.md presence: {md_status}")


# jcode serve
@main.command()
def serve() -> None:
    """Start the jcode MCP server (stdio transport, for use with Claude Code)."""
    from jcode.mcp.server import mcp
    mcp.run()


# jcode plugins
@main.command()
def plugins() -> None:
    """List available plugins (built-in and registry)."""
    from importlib.metadata import entry_points
    from jcode.registry import fetch_registry, RegistryError

    click.echo("Fetching registry …")

    try:
        registry = fetch_registry()
    except RegistryError as e:
        click.echo(f"  Error: {e}", err=True)
        return

    installed = {ep.name for ep in entry_points(group="jcode.plugins")}
    registry_plugins = registry.get("plugins", {})

    if not registry_plugins:
        click.echo("Registry plugins: (none yet — be the first to contribute!)")
        return

    click.echo("Registry plugins:")
    for name, info in registry_plugins.items():
        status = "installed" if name in installed else "not installed"
        click.echo(f"  {name:<20} {info['description']:<50} [{status}]")
        click.echo(f"    pip: {info['pip']}  repo: {info.get('repo', 'n/a')}")


# jcode add
@main.command()
@click.argument("plugin")
def add(plugin: str) -> None:
    """Install a plugin from the jcode registry.

    Example: jcode add sqlalchemy

    Plugins not in the registry must be installed manually with pip install.
    Any package that declares a 'jcode.plugins' entry point will be
    auto-detected once installed.
    """
    from jcode.registry import fetch_registry, RegistryError

    if plugin in _BUILTIN_PLUGINS:
        click.echo(
            f"'{plugin}' is a built-in plugin — already included with jcode, "
            "nothing to install."
        )
        return

    try:
        registry = fetch_registry()
    except RegistryError as e:
        raise click.ClickException(str(e))

    registry_plugins = registry.get("plugins", {})

    if plugin not in registry_plugins:
        names = ", ".join(registry_plugins.keys()) or "none yet"
        raise click.ClickException(
            f"'{plugin}' is not in the jcode plugin registry.\n"
            f"  Available: {names}\n"
            f"  To install a plugin not in the registry:\n"
            f"    pip install <package-name>\n"
            f"  The package must declare a 'jcode.plugins' entry point.\n"
            f"  To see the full registry: jcode plugins"
        )

    git_url = registry_plugins[plugin].get("git")
    if not git_url:
        raise click.ClickException(
            f"No install source found for '{plugin}' in the registry."
        )

    click.echo(f"Installing {plugin} …")

    # Install into the same Python environment that is running jcode
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", f"git+{git_url}"],
        check=False,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"Installation failed. Try manually:\n"
            f"  pip install git+{git_url}"
        )

    click.echo(f"Done. '{plugin}' will auto-load for repos that use {plugin}.")
