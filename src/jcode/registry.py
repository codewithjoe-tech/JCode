"""
Registry client for jcode plugins.

The registry lives in its own repo:
  https://github.com/codewithjoe-tech/jcode-registry

Lists community plugins installable via ``jcode add <name>``.
Any package that declares a ``jcode.plugins`` entry point works automatically —
it does not need to be in the registry, but listing it enables ``jcode add``.
"""
import json
import urllib.request
import urllib.error

REGISTRY_URL = (
    "https://raw.githubusercontent.com/codewithjoe-tech/jcode-registry/main/registry.json"
)


class RegistryError(Exception):
    pass


def fetch_registry(url: str = REGISTRY_URL) -> dict:
    """
    Fetch the plugin registry from the jcode-registry GitHub repo.

    Raises RegistryError if the network is unreachable or the response
    is not valid JSON.
    """
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise RegistryError(
            f"Could not reach the plugin registry ({e.reason}).\n"
            "  Check your internet connection, or browse it directly:\n"
            "  https://github.com/codewithjoe-tech/jcode-registry"
        ) from e
    except (json.JSONDecodeError, OSError) as e:
        raise RegistryError(f"Registry response was not valid JSON: {e}") from e
