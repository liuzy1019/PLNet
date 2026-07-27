"""Composable YAML configuration loading."""

from pathlib import Path
from typing import Any, Dict, Tuple, Union

import yaml


def load_config(
    path: Union[str, Path],
    _stack: Tuple[Path, ...] = (),
) -> Dict[str, Any]:
    """Load a flat config, recursively merging files listed in ``_base_``."""
    config_path = Path(path).expanduser().resolve()
    if config_path in _stack:
        chain = " -> ".join(str(item) for item in (*_stack, config_path))
        raise ValueError(f"Circular configuration inheritance: {chain}")

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")

    bases = config.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    if not isinstance(bases, list) or not all(isinstance(item, str) for item in bases):
        raise ValueError(f"_base_ must be a string or list of strings: {config_path}")

    nested = [key for key, value in config.items() if isinstance(value, dict)]
    if nested:
        keys = ", ".join(sorted(nested))
        raise ValueError(f"Configuration must be flat; nested keys found: {keys}")

    merged: Dict[str, Any] = {}
    for base in bases:
        merged.update(load_config(config_path.parent / base, (*_stack, config_path)))
    merged.update(config)
    return merged
