"""The engine only reaches the preparation distribution through lazy facades.

The preparation subsystem moved to ``neomd_preparation``; the engine keeps
stable facades at the historical module paths. This source scan pins where that
distribution may be named and requires every such import to be function-level
(inside a function or a module ``__getattr__``), so ``import neomd`` never
needs the preparation distribution installed.
"""

import ast
from pathlib import Path

ENGINE_SRC = Path(__file__).resolve().parents[2] / "src" / "neomd"

# The stable facades: the only modules allowed to name the moved distribution.
ALLOWED_FILES = {
    "__init__.py",
    "system.py",
    "integration.py",
    "prepared_pipeline.py",
    "topology_preparation.py",
}


def _preparation_imports(tree):
    """Every import of ``neomd_preparation[.*]`` in *tree*, with its node."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "neomd_preparation":
                    yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] == "neomd_preparation":
                yield node, module


def _engine_modules():
    for path in sorted(ENGINE_SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path.relative_to(ENGINE_SRC).as_posix(), path


def test_preparation_distribution_is_only_named_by_the_facades():
    offenders = []
    for relative, path in _engine_modules():
        if relative in ALLOWED_FILES:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node, module in _preparation_imports(tree):
            offenders.append(f"{relative}:{node.lineno}: import {module}")
    assert not offenders, (
        "neomd_preparation may only appear in the lazy facades "
        f"{sorted(ALLOWED_FILES)}:\n" + "\n".join(offenders)
    )


def test_facade_imports_are_function_level():
    offenders = []
    for relative, path in _engine_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        module_level = {id(node) for node in tree.body}
        for node, module in _preparation_imports(tree):
            if id(node) in module_level:
                offenders.append(f"{relative}:{node.lineno}: module-level import {module}")
    assert not offenders, (
        "importing the engine must not require neomd_preparation; move these "
        "imports inside the function or module __getattr__:\n" + "\n".join(offenders)
    )
