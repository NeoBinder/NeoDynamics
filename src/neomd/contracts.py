"""Locate versioned contract schemas in a source tree or an installed wheel.

Engine contracts ship as package data under share/neodynamics/contracts (see
[tool.setuptools.data-files]). Consumers must not hard-code
parents[2]/contracts or the share prefix: this module resolves both layouts
from the installed code location.

Stability: the three functions below are part of the documented integration
surface and change only with a major version. Contract URNs, schema ids and
directory names are unchanged by repository extraction.
"""

from __future__ import annotations

import sysconfig
from pathlib import Path
from typing import Iterator

__all__ = ["contracts_root", "schema_dir", "schema_paths"]

_CONTRACT_SUFFIXES = (".json", ".yaml", ".yml")


def _source_root() -> Path:
    # src/neomd/contracts.py -> repository root
    return Path(__file__).resolve().parents[2] / "contracts"


def _installed_root() -> Path:
    return Path(sysconfig.get_path("data")) / "share" / "neodynamics" / "contracts"


def contracts_root() -> Path:
    """Return the directory holding the versioned contract folders.

    Prefers the source-tree contracts/ directory; otherwise uses the data
    files recorded by the installed distribution. Raises FileNotFoundError
    with both candidate paths when neither exists.
    """
    source = _source_root()
    if source.is_dir():
        return source
    installed = _installed_root()
    if installed.is_dir():
        return installed
    raise FileNotFoundError(
        "no contracts directory found; checked "
        f"{source} (source tree) and {installed} (installed data files)"
    )


def schema_dir(relative: str) -> Path:
    """Return an existing contract directory such as "prepared-bundle/v2"."""
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"contract directory must be relative: {relative!r}")
    path = contracts_root() / candidate
    if not path.is_dir():
        raise FileNotFoundError(f"contract directory not found: {path}")
    return path


def schema_paths(*relative_dirs: str) -> Iterator[Path]:
    """Yield the bounded set of contract documents in each directory, sorted.

    Only files directly inside the directory are returned; fixtures and other
    nested data are not part of the wire-contract surface.
    """
    for relative in relative_dirs:
        directory = schema_dir(relative)
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix in _CONTRACT_SUFFIXES:
                yield path
