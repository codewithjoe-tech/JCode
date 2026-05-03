import tree_sitter_python
from tree_sitter import Language, Parser, Query, QueryCursor
from pathlib import Path

try:
    lang = Language(tree_sitter_python.language())
    print("Language ok")

    import jcode
    scm_path = Path(jcode.__file__).parent / "indexer" / "queries" / "python.scm"
    print("scm exists:", scm_path.exists())

    q = Query(lang, scm_path.read_text())
    print("Query ok")

    p = Parser(lang)
    print("Parser ok")

    from jcode.indexer.generic_parser import GenericParser
    gp = GenericParser()
    print("file_extensions:", gp.file_extensions)

    import importlib.metadata
    print("jcode version:", importlib.metadata.version("jcode"))

except Exception as e:
    import traceback
    traceback.print_exc()
