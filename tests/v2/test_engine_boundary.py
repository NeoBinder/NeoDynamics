"""The engine tree never reaches into the extracted resident service."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ENGINE_SRC = REPO_ROOT / "src" / "neomd"
TESTS = REPO_ROOT / "tests"

# The import package, console script, test tree and source trees that belong to
# the NeomdServer distribution. None of them may appear in the engine sources,
# the engine tests, the published docs or the README.
SERVICE_MARKERS = (
    "neomd_server",
    "neomd-server",
    "tests/server",
    "src/neomd_server",
    "deploy/",
    "docker/",
)

# Two engine documents describe the extraction itself. ADR 0010 must name the
# extracted package and the paths it used to occupy to record the decision, and
# the integration-surface page must name the import package an integrator pins.
# Every other document describes the SDK and links to those two instead.
BOUNDARY_DOCUMENTS = {
    "docs/adr/0010-server-repository-extraction.md": "records the extraction decision",
    "docs/integration-surface.md": "documents the integrator-facing surface",
}

# Tests may mention the service package in a negative assertion (the engine
# surface test proves it is absent), so the test-tree scan looks for real
# imports rather than the bare name.
SERVICE_IMPORT = re.compile(
    r"^\s*(?:from\s+neomd_server\b|import\s+neomd_server\b)"
    r"|importlib\.import_module\(\s*[\"']neomd_server",
    re.MULTILINE,
)

# This guard names the forbidden markers, so it is the one engine test file
# exempt from the scan.
GUARD_FILE = Path(__file__).resolve()


def _offending_files(root: Path, pattern: str, markers: tuple[str, ...], skip=()) -> list[str]:
    skipped = {Path(path).resolve() for path in skip}
    offenders = []
    for path in sorted(root.rglob(pattern)):
        resolved = path.resolve()
        if resolved in skipped or "__pycache__" in resolved.parts:
            continue
        text = resolved.read_text(encoding="utf-8", errors="replace")
        if any(marker in text for marker in markers):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    return offenders


def test_engine_sources_do_not_import_the_service():
    assert _offending_files(ENGINE_SRC, "*.py", ("neomd_server",)) == []


def test_engine_tests_do_not_import_the_service():
    skipped = {GUARD_FILE}
    offenders = []
    for path in sorted(TESTS.rglob("*.py")):
        resolved = path.resolve()
        if resolved in skipped or "__pycache__" in resolved.parts:
            continue
        if SERVICE_IMPORT.search(resolved.read_text(encoding="utf-8", errors="replace")):
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == []


def test_docs_and_readme_do_not_reference_service_source_paths():
    offenders = _offending_files(
        REPO_ROOT / "docs",
        "*.md",
        SERVICE_MARKERS,
        skip=tuple(REPO_ROOT / name for name in BOUNDARY_DOCUMENTS),
    )
    readme = REPO_ROOT / "README.md"
    if any(marker in readme.read_text(encoding="utf-8") for marker in SERVICE_MARKERS):
        offenders.append("README.md")
    assert offenders == []


def test_boundary_documents_still_exist():
    # The exemption above is only meaningful while these two files keep naming
    # the boundary; a deleted or renamed document must not silently widen it.
    for relative in BOUNDARY_DOCUMENTS:
        assert (REPO_ROOT / relative).is_file(), relative
