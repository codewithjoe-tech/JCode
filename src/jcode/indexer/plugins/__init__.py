"""
Plugin registry and auto-detector.

Auto-detection: scans the repo root for requirements.txt / pyproject.toml
and loads matching plugins without user configuration.

Manual override: pass plugin instances directly to GenericParser(plugins=[...])
"""
import importlib
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

_REGISTRY: dict[str, str] = {
    "fastapi":    "jcode.indexer.plugins.fastapi_plugin",
    "django":     "jcode.indexer.plugins.django_plugin",
    "sqlalchemy": "jcode.indexer.plugins.sqlalchemy_plugin",
}

def _scan_dependencies(repo_root: str) -> set[str]:
    """Return lowercase package names declared in requirements.txt / pyproject.toml."""
    names: set[str] = set()
    root = Path(repo_root)

    req = root / "requirements.txt"
    if req.exists():
        for line in req.read_text(errors="replace").splitlines():
            pkg = line.strip().lower().split("==")[0].split(">=")[0].split("[")[0]
            if pkg and not pkg.startswith("#"):
                names.add(pkg)

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

    for venv in [root / ".venv", root / "venv"]:
        site = venv / "Lib" / "site-packages"
        if not site.exists():
            site = venv / "lib"
        if site.exists():
            for entry in site.iterdir():
                names.add(entry.name.lower().split("-")[0])

    return names

def load_plugins_for_repo(repo_root: str) -> list:
    """Auto-detect and instantiate plugins appropriate for *repo_root*."""
    deps = _scan_dependencies(repo_root)
    plugins = []
    for framework, module_path in _REGISTRY.items():
        if framework in deps:
            try:
                mod = importlib.import_module(module_path)
                plugins.append(mod.create())
            except Exception:
                pass
    return plugins

def build_parser(repo_root: str):
    """Return a GenericParser auto-configured with plugins for *repo_root*."""
    from jcode.indexer.generic_parser import GenericParser
    return GenericParser(plugins=load_plugins_for_repo(repo_root))
