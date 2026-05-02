"""
Git utilities for jcode.

All subprocess calls to git are isolated here so the rest of the codebase
stays subprocess-free.  Every function returns None / empty results if the
repo is not a git repo or git is not installed — callers fall back gracefully.
"""
import os
import subprocess
from pathlib import Path


def is_git_repo(repo_root: str) -> bool:
    """Return True if *repo_root* is inside a git repository."""
    return (Path(repo_root) / ".git").exists()


def git_head(repo_root: str) -> str | None:
    """Return the current HEAD commit SHA, or None on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None


def git_is_clean(repo_root: str) -> bool:
    """
    Return True if the working tree and index are clean (no uncommitted changes).

    Uses `git status --porcelain` — any output means dirty.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip() == ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return False


def git_changed_files(
    repo_root: str,
    since_commit: str,
    until_commit: str = "HEAD",
) -> tuple[set[str], set[str]]:
    """
    Return (changed, deleted) relative paths between *since_commit* and *until_commit*.

    *changed* — added, modified, renamed (new path), copied (new path).
    *deleted* — deleted files.

    Paths are relative to *repo_root*.
    Returns (set(), set()) on any failure.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", since_commit, until_commit],
            cwd=repo_root, capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            return set(), set()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return set(), set()

    changed: set[str] = set()
    deleted: set[str] = set()
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        status = parts[0]
        # Rename/copy: R100\told_path\tnew_path  — take new path
        path = parts[-1].strip()
        if not path:
            continue
        if status.startswith("D"):
            deleted.add(path)
        else:
            changed.add(path)

    return changed, deleted


def git_working_tree_changes(repo_root: str) -> set[str]:
    """
    Return paths modified in the working tree or index (uncommitted changes).

    Includes untracked files so edits made after the last commit are caught.
    Paths are relative to *repo_root*.
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return set()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return set()

    paths: set[str] = set()
    for line in result.stdout.splitlines():
        if len(line) < 4:
            continue
        # Format: XY path  or  XY old -> new  (renames)
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ")[-1]
        paths.add(path)
    return paths
