"""Dataset manifest and path helpers."""

import json
from pathlib import Path
from typing import Any, Dict, Union


def load_manifest(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON dataset manifest with explicit UTF-8 handling."""
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def resolve_data_path(
    project_root: Union[str, Path],
    value: Union[str, Path],
) -> str:
    """Resolve a manifest path against the repository's ``data`` directory.

    Existing absolute paths are preserved. Relative paths are first resolved
    from the repository root and then from its ``data`` directory.
    """
    root = Path(project_root).expanduser().resolve()
    direct = Path(value).expanduser()

    if direct.is_absolute():
        return str(direct)

    candidates = (root / direct, root / "data" / direct)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])
