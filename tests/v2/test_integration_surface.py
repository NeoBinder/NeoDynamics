"""The documented service-facing integration surface stays importable and bound.

The resident service is extracted into its own distribution; it may only use
state=public engine modules plus `neomd.integration`. These tests keep that
surface from silently disappearing during engine refactors. The
preparation-backed names resolve lazily from the separate preparation
distribution, so this module pins the engine-owned ones directly and keeps the
lazy spellings in ``__all__``.
"""

import os
import subprocess
import sys
from pathlib import Path

from neomd import __version__, integration, probes
from neomd.kernel import _bootstrap

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_public_names_are_bound_to_their_owner_modules():
    assert integration.ensure_adapters is _bootstrap.ensure_adapters
    assert integration.STATE_HEADER == probes.STATE_HEADER
    assert integration.engine_version() == __version__


def test_preparation_backed_names_stay_in_the_public_surface():
    # the spellings remain supported; they resolve lazily from neomd_preparation
    for name in (
        "BoundedRunner",
        "GAFF_VERSION",
        "ambertools_receipt",
        "topology_from_document",
    ):
        assert name in integration.__all__


def test_surface_has_no_private_names_and_no_service_import():
    assert [name for name in integration.__all__ if name.startswith("_")] == []
    source = Path(integration.__file__).read_text(encoding="utf-8")
    assert "neomd_server" not in source


def test_importing_neomd_does_not_require_preparation_distribution():
    """``import neomd`` must not import the preparation distribution."""
    code = (
        "import neomd, sys; "
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'neomd_preparation'); "
        "assert not loaded, loaded"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
