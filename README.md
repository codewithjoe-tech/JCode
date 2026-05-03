# jcode

> Code intelligence for AI agents — without burning your token budget.

---

If you've ever watched Claude or GPT spend 40 tool calls grepping through your codebase just to answer "where does authentication happen?", you'll get why this exists.

jcode builds a feature graph of your repo — functions, classes, imports, call relationships — stores it locally as a SQLite database, and exposes it as an MCP server. Instead of reading twenty files, your AI agent calls one tool and gets exactly the context it needs.

It's fast, it's local, and it works with Python, TypeScript, JavaScript, Go, and Rust repos right out of the box.

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

The blast radius tool is the one people find most surprising. Change a function signature, a model field, or a shared utility and instantly see every caller, handler, and dependent that needs to know about it — across the entire codebase, regardless of language.

---

## Language support

jcode uses [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) under the hood, so it understands the actual structure of your code rather than just searching text.

| Language | Status |
|----------|--------|
| Python | ✅ Full support — functions, classes, methods, imports, call edges |
| TypeScript | ✅ Supported |
| JavaScript | ✅ Supported |
| Go | ✅ Supported |
| Rust | ✅ Supported |

Mixed-language repos work fine — index once, query across everything.

Install the parsers you need:

```bash
pip install jcode[typescript]
pip install jcode[javascript]
pip install jcode[go]
pip install jcode[rust]
pip install jcode[all-langs]   # everything at once
```

---

## Quick start

**Step 1 — Install**

```bash
pip install jcode
```

Or with uv:

```bash
uv add jcode
```

Verify it worked:

```bash
jcode --version
jcode --help
```

**Step 2 — Index your project**

```bash
jcode index /path/to/your/project
```

This creates a `.jcode/` folder inside your project with the graph database. Works on any supported language — point it at a Python backend, a TypeScript frontend, a Go service, whatever.

**Step 3 — Write CLAUDE.md**

```bash
jcode init /path/to/your/project
```

This writes a `CLAUDE.md` at your project root so Claude Code knows to use jcode instead of grepping.

**Step 4 — Connect to Claude Code**

```bash
jcode setup-mcp /path/to/your/project
```

This prints the exact command you need to run, something like:

```
claude mcp add jcode jcode serve -e "JCODE_DIR=/path/to/your/project/.jcode"
```

Copy that and run it. One-time setup per project.

**Step 5 — Open Claude Code in your project**

```bash
cd /path/to/your/project
claude
```

Claude will read `CLAUDE.md` on startup and use jcode instead of reading files one by one.

---

## Incremental indexing

Re-indexing only processes files that changed since the last run — content hashed, not timestamp compared. Fast enough to run as a file-watcher or a pre-commit hook.

```bash
jcode index /path/to/repo                    # incremental (default)
jcode index /path/to/repo --full-reindex     # wipe and rebuild
```

---

## Framework plugins

For some frameworks, generic call analysis isn't enough — you need to understand what the framework is doing with your code. jcode has a plugin system for exactly this. Auto-detection is based on your `requirements.txt` or `pyproject.toml`, no config needed.

**FastAPI** — detects `Depends(fn)` and `Security(fn)` in route handlers and emits typed `depends` edges. Blast radius from `get_current_user` correctly reaches every protected route.

**Django** — three patterns covered:
- `path('login/', views.login_view)` → `route` edge to the view function
- `ForeignKey(User, ...)` → `references` edge between models
- `@receiver(post_save, sender=User)` → `signal` edge from handler to model

More plugins are coming. Writing one is about 50 lines — implement `handled_names` and `handle_call`, register it in `_REGISTRY`.

---

## Semantic search

Install `sentence-transformers` and search upgrades from keyword matching to semantic vector search. Useful when you know what something *does* but not what it's *called*.

```bash
pip install jcode[embed]
```

Embeddings are computed once on first index and stored in the graph. Incremental runs only embed new or changed nodes.

---

## How the graph is stored

Everything goes in `.jcode/` at your repo root:

- `graph.db` — SQLite database with nodes, edges, FTS5 full-text index, vector embeddings, and a file-hash manifest for incremental indexing
- `objects/` — content-addressable store for raw node data (git-style blobs, keyed by SHA-256)

Add `.jcode/` to your `.gitignore` — it's generated data.

---

## CLI reference

```
jcode index <repo>          Index or re-index a repository
jcode init  <repo>          Write CLAUDE.md to the repo root
jcode status <repo>         Show index stats
jcode setup-mcp <repo>      Print the claude mcp add command
jcode serve                 Start the MCP server (stdio transport)
```

---

## Project structure

```
src/jcode/
├── domain/         core models and port interfaces (zero external deps)
├── storage/        SQLite graph DB + content-addressable object store
├── indexer/        Tree-sitter parsers, incremental builder, plugins, embedder
├── graph/          DFS context traversal + BFS blast radius
└── mcp/            FastMCP server exposing the five tools
```

---

## Known limitations

- Cross-file call resolution is name-based — if two modules have a function with the same name, they're treated as the same target. Rare in practice but worth knowing.
- The semantic search quality depends on the embedding model. The default is `BAAI/bge-small-en-v1.5` via fastembed (ONNX, fast, no PyTorch required). Falls back to `all-MiniLM-L6-v2` via sentence-transformers if fastembed is not installed.
- `.jcode/` grows with your repo. `--full-reindex` cleans it up if it feels stale.

---

## License

MIT — free to use, modify, distribute, and build commercial products on top of. See [LICENSE](LICENSE).

---

Built because reading source code line by line is a bad use of an AI agent's attention.

---

## Author

**Joel Thomas**

[🌐 codewithjoe.in](https://codewithjoe.in) · [LinkedIn](https://www.linkedin.com/in/codewithjoe) · [Instagram](https://www.instagram.com/codewithjoe16/) · [𝕏](https://x.com/codewithjoee)
