"""
Plugin loader and auto-detector.

Plugins are discovered via the ``jcode.plugins`` entry-point group.
Install plugins with ``jcode add <name>`` — they register themselves as
entry points so jcode core never needs to change.

Auto-detection: scans the repo root for requirements.txt / pyproject.toml
(Python) and package.json (Node.js) and loads only the plugins whose
framework name appears in the target repo's declared dependencies.

Manual override: pass plugin instances directly to GenericParser(plugins=[...])
"""
import json
from importlib.metadata import entry_points
from pathlib import Path

import tomllib


def _scan_dependencies(repo_root: str) -> set[str]:
    """Return lowercase package/module names declared in the repo's dependency files."""
    names: set[str] = set()
    root = Path(repo_root)

    # Python — requirements.txt
    req = root / "requirements.txt"
    if req.exists():
        for line in req.read_text(errors="replace").splitlines():
            pkg = line.strip().lower().split("==")[0].split(">=")[0].split("[")[0]
            if pkg and not pkg.startswith("#"):
                names.add(pkg)

    # Python — pyproject.toml
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
            deps: list[str] = []
            deps += data.get("project", {}).get("dependencies", [])
            deps += list(data.get("tool", {}).get("poetry", {}).get("dependencies", {}).keys())
            for dep in deps:
                pkg = dep.strip().lower()
                for sep in (">=", "==", "[", ";"):
                    pkg = pkg.split(sep)[0]
                pkg = pkg.strip()
                if pkg:
                    names.add(pkg)
        except (tomllib.TOMLDecodeError, OSError):
            pass

    # Python — installed venv packages
    for venv in [root / ".venv", root / "venv"]:
        site = venv / "Lib" / "site-packages"
        if not site.exists():
            site = venv / "lib"
        if site.exists():
            for entry in site.iterdir():
                names.add(entry.name.lower().split("-")[0])

    # Node.js — package.json
    package_json = root / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                for pkg in data.get(section, {}):
                    names.add(pkg.strip().lower())
        except (json.JSONDecodeError, OSError):
            pass

    return names


def load_plugins_for_repo(repo_root: str) -> list:
    """
    Auto-detect and instantiate plugins appropriate for *repo_root*.

    Discovers all plugins registered under the ``jcode.plugins`` entry-point
    group (built-in and third-party), then filters to those whose framework
    name appears in the target repo's declared dependencies.

    Entry point name convention: the name must match the package name of the
    framework it targets (e.g. ``fastapi``, ``django``, ``express``, ``mongoose``).
    Third-party plugins follow the same convention in their own pyproject.toml.
    """
    deps = _scan_dependencies(repo_root)
    plugins = []
    for ep in entry_points(group="jcode.plugins"):
        if ep.name in deps:
            try:
                create_fn = ep.load()
                plugins.append(create_fn())
            except Exception:
                pass
    return plugins


def build_parser(repo_root: str):
    """Return a GenericParser auto-configured with plugins for *repo_root*."""
    from jcode.indexer.generic_parser import GenericParser
    return GenericParser(plugins=load_plugins_for_repo(repo_root))
