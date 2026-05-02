# jcode

> Code intelligence for AI agents — without burning your token budget.

---

If you've ever watched Claude or GPT spend 40 tool calls grepping through your codebase just to answer "where does authentication happen?", you'll get why this exists.

jcode builds a feature graph of your repo — functions, classes, imports, call relationships — stores it locally as a SQLite database, and exposes it as an MCP server. Instead of reading twenty files, your AI agent calls one tool and gets exactly the context it needs.

It's fast, it's local, and it works with any repo that has Python in it (TypeScript, Go, Rust, and JS support is also in there).

---

## What it actually does

When you run `jcode index`, it walks your repo with Tree-sitter, extracts every function, class, method, and import, and builds a directed graph of how they relate to each other. That graph lives in a `.jcode/` folder at your repo root.

From there, your AI agent has five tools:

| Tool | What it does |
|------|-------------|
| `jcode_feature_map` | High-level folder/feature overview — always call this first |
| `jcode_search` | Semantic + keyword search to find entry points |
| `jcode_context` | Forward DFS from a node — "show me everything this function touches" |
| `jcode_blast_radius` | Reverse BFS — "if I change this, what breaks?" |
| `jcode_index` | Trigger a (re-)index from inside a conversation |

The blast radius tool is the one people find most surprising. Change a model field and instantly know every view, signal handler, and serializer that depends on it.

---

## Quick start

**Step 1 — Install from source**

Clone the repo and install it locally. Not on PyPI yet.

```bash
git clone https://github.com/<your-username>/jcode.git
cd jcode
pip install -e .
```

Or if you already have the folder:

```bash
cd C:\path\to\jcode
pip install -e .
```

Verify it worked:

```bash
jcode --help
```

**Step 2 — Index your project**

```bash
jcode index C:\path\to\your\project
```

This creates a `.jcode/` folder inside your project with the graph database.

**Step 3 — Write CLAUDE.md**

```bash
jcode init C:\path\to\your\project
```

This writes a `CLAUDE.md` at your project root so Claude Code knows to use jcode.

**Step 4 — Connect to Claude Code**

```bash
jcode setup-mcp C:\path\to\your\project
```

This prints the exact command you need to run, something like:

```
claude mcp add jcode -e JCODE_DIR=C:\path\to\your\project\.jcode -- jcode serve
```

Copy that output and run it. You only need to do this once per project.

**Step 5 — Open Claude Code in your project**

```bash
cd C:\path\to\your\project
claude
```

Claude will read `CLAUDE.md` on startup and use jcode instead of grepping through files.

---

## Incremental indexing

Re-indexing only processes files that changed since the last run. On a large repo it's fast enough to wire up as a file-watcher or a pre-commit hook.

```bash
jcode index /path/to/repo            # incremental (default)
jcode index /path/to/repo --full-reindex   # wipe and rebuild
```

---

## Plugin system

jcode has a plugin system for framework-specific edges that generic call analysis can't see. Auto-detection is based on what's in `requirements.txt` or `pyproject.toml` — no config needed.

**FastAPI plugin** — detects `Depends(fn)` and `Security(fn)` in route handlers and emits typed `depends` edges. So blast radius from `get_current_user` correctly reaches every protected route.

**Django plugin** — three patterns:
- `path('login/', views.login_view)` → `route` edge to the view
- `ForeignKey(User, ...)` → `references` edge between models
- `@receiver(post_save, sender=User)` → `signal` edge from handler to model

Writing your own plugin is about 50 lines — implement `handled_names` and `handle_call`, drop it in `jcode/indexer/plugins/`, register it in `_REGISTRY`.

---

## Semantic search

If you install `sentence-transformers`, search upgrades from keyword matching to semantic vector search. Useful for queries like "date parsing" when the actual function is called `_normalise_timestamp`.

```bash
pip install jcode[embed]
```

Embeddings are computed once on first index and stored in the graph. Incremental re-indexing only embeds new or changed nodes.

---

## Language support

Python is the main target and gets the most love. The others work but are less battle-tested.

```bash
pip install jcode[typescript]
pip install jcode[javascript]
pip install jcode[go]
pip install jcode[rust]
pip install jcode[all-langs]   # all of the above
```

---

## How the graph is stored

Everything goes in `.jcode/` at your repo root. Two things live there:

- `graph.db` — SQLite database with nodes, edges, FTS5 index, embeddings, and a file-hash manifest for incremental indexing
- `objects/` — content-addressable store for raw node data (git-style, keyed by SHA-256)

Add `.jcode/` to your `.gitignore`. It's generated data — no point committing it.

---

## CLI reference

```
jcode index <repo>          Index or re-index a repository
jcode init  <repo>          Write CLAUDE.md to the repo root
jcode status <repo>         Show index stats
jcode setup-mcp <repo>      Print the claude mcp add command
jcode serve                 Start the MCP server (stdio)
```

---

## Project structure

```
src/jcode/
├── domain/         models, ports (no external deps)
├── storage/        SQLite graph DB + content-addressable object store
├── indexer/        Tree-sitter parser, builder, plugins, embedder
├── graph/          DFS context + BFS blast radius traversal
└── mcp/            FastMCP server (the five tools)
```

---

## Limitations / known things

- Python parsing is mature. Other languages are usable but won't catch every edge.
- Cross-file call resolution is name-based — same-name functions in different modules are treated as the same target. This is almost always fine in practice.
- The semantic search is as good as your sentence-transformer model. The default (`all-MiniLM-L6-v2`) is fast and decent.
- `.jcode/` can get large on very big repos. Run `jcode index --full-reindex` occasionally if it feels stale.

---

## License

Personal and non-commercial use only. See [LICENSE](LICENSE).

---

Built because reading source code line by line is a bad use of an AI agent's attention.

---

## Author

**Joel Thomas**

[🌐 codewithjoe.in](https://codewithjoe.in) · [LinkedIn](https://www.linkedin.com/in/codewithjoe) · [Instagram](https://www.instagram.com/codewithjoe16/)
