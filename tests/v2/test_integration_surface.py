"""The documented service-facing integration surface stays importable and bound.

The resident service is extracted into its own distribution; it may only use
state=public engine modules plus `neomd.integration`. These tests keep that
surface from silently disappearing during engine refactors.
"""

from pathlib import Path

from neomd import __version__, component_gaff, component_system_qc, integration, probes
from neomd.kernel import _bootstrap


def test_public_names_are_bound_to_their_owner_modules():
    assert integration.ensure_adapters is _bootstrap.ensure_adapters
    assert integration.BoundedRunner is component_gaff.BoundedRunner
    assert integration.GAFF_VERSION == component_gaff.GAFF_VERSION
    assert integration.ambertools_receipt is component_gaff.ambertools_receipt
    assert integration.topology_from_document is component_system_qc.topology_from_document
    assert integration.STATE_HEADER == probes.STATE_HEADER
    assert integration.engine_version() == __version__


def test_surface_has_no_private_names_and_no_service_import():
    assert [name for name in integration.__all__ if name.startswith("_")] == []
    source = Path(integration.__file__).read_text(encoding="utf-8")
    assert "neomd_server" not in source


def test_legacy_private_spellings_remain_aliases():
    assert component_gaff._BoundedRunner is component_gaff.BoundedRunner
    assert component_gaff._GAFF == component_gaff.GAFF_VERSION
    assert component_gaff._ambertools_receipt is component_gaff.ambertools_receipt
    assert (
        component_system_qc._topology_from_document
        is component_system_qc.topology_from_document
    )
    assert probes._STATE_HEADER == probes.STATE_HEADER
