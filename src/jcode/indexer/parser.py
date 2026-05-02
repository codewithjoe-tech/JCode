# Backward-compatibility alias — not used internally.
# GenericParser is the canonical implementation; import it directly.
from jcode.indexer.generic_parser import GenericParser as PythonParser  # noqa: F401

__all__ = ["PythonParser"]
