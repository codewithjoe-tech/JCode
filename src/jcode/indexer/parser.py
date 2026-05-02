"""Backward-compatibility shim — GenericParser is the canonical implementation."""
from jcode.indexer.generic_parser import GenericParser as PythonParser  # noqa: F401

__all__ = ["PythonParser"]
