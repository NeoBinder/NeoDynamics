"""Method control revisions travel atomically with the kernel checkpoint."""

from dataclasses import replace

import numpy as np
import pytest

from neomd.kernel.checkpoint import MAGIC, unwrap_checkpoint, wrap_checkpoint
from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import CheckpointMetadataOps, KernelSpec, provides


def control(revision):
    return {"gamd_control": {"revision": revision, "sigma0": .2 + revision}}


def test_fake_same_step_controls_and_legacy_clear_are_snapshot_specific():
    kernel = FakeKernel(KernelSpec(kind="fake", seed=79))
    assert provides(kernel, CheckpointMetadataOps)
    raw = kernel.snapshot()
    assert not raw.startswith(MAGIC)
    kernel.step(2)
    kernel.set_checkpoint_metadata(control(0))
    before = kernel.snapshot()
    kernel.set_checkpoint_metadata(control(1))
    after = kernel.snapshot()
    kernel.restore(before)
    assert kernel.current_step == 2
    assert kernel.checkpoint_metadata() == control(0)
    kernel.restore(after)
    assert kernel.current_step == 2
    assert kernel.checkpoint_metadata() == control(1)
    kernel.restore(raw)
    assert kernel.current_step == 0
    assert kernel.checkpoint_metadata() == {}
    assert kernel.snapshot() == raw


def test_metadata_is_deep_copied_in_both_directions():
    kernel = FakeKernel(KernelSpec(kind="fake"))
    original = {"nested": [{"value": 1}]}
    kernel.set_checkpoint_metadata(original)
    original["nested"][0]["value"] = 2
    fetched = kernel.checkpoint_metadata()
    fetched["nested"][0]["value"] = 3
    assert kernel.checkpoint_metadata() == {"nested": [{"value": 1}]}


@pytest.mark.parametrize("invalid", [[], {"x": float("nan")}, {"x": float("inf")},
                                     {"x": {1: "key"}}, {"x": object()}, {"x": (1, 2)}])
def test_invalid_metadata_does_not_replace_previous_state(invalid):
    kernel = FakeKernel(KernelSpec(kind="fake"))
    kernel.set_checkpoint_metadata(control(0))
    with pytest.raises(ValueError):
        kernel.set_checkpoint_metadata(invalid)
    assert kernel.checkpoint_metadata() == control(0)


@pytest.mark.parametrize("data", [MAGIC, MAGIC + (100).to_bytes(8, "big") + b"{}native",
                                  MAGIC + (2).to_bytes(8, "big") + b"[]native",
                                  MAGIC + (9).to_bytes(8, "big") + b'{"x":NaN}native',
                                  b"NEOMD-METADATA-2\x00"])
def test_malformed_metadata_is_rejected(data):
    with pytest.raises(ValueError):
        unwrap_checkpoint(data)


def test_raw_wrapper_compatibility_and_replay_has_no_capability():
    from neomd.kernel.replay import ReplayKernel

    assert wrap_checkpoint(b"native", {}) == b"native"
    assert unwrap_checkpoint(b"native") == (b"native", {})
    replay = ReplayKernel(KernelSpec(kind="replay"), tape={"energies": [0.]})
    assert not provides(replay, CheckpointMetadataOps)


@pytest.mark.parametrize("npt", [False, True])
def test_openmm_metadata_direct_and_lazy_resume_preserve_native_state(tmp_path, npt):
    pytest.importorskip("openmm")
    from neomd.kernel.openmm import OpenMMKernel
    from neomd.kernel.port import BoostChannelIR
    from tests.v2.test_gamd_partition import fixture

    _, _, spec = fixture(tmp_path, periodic=npt)
    if npt:
        # Explicit opt-in: only adapter-owned proposal state can be resumed
        # bit-exactly mid-leg (ADR-0009); native NPT state is not checkpointed.
        spec = replace(spec, barostat={"frequency": 3, "seed": 419, "managed": True})

    def build(settings):
        kernel = OpenMMKernel(settings)
        # A boost channel's P comes from the compiled partition's CV energy
        # model (ADR-0009), so the whole-system partition must be compiled
        # before the channel is installed — the public GaMD order.
        report = kernel.compile_boost_partitions([{
            "label": "total",
            "selectors": [{"atom_indices": "all",
                           "force_types": ("bond", "angle", "dihedral", "nonbonded")}],
        }])
        groups = tuple(report["channels"][0]["groups"])
        kernel.install_boost([BoostChannelIR("total", groups, 100., .002)])
        return kernel

    kernel = build(spec)
    assert provides(kernel, CheckpointMetadataOps)
    kernel.step(7)  # Preserve a nonzero scheduling phase for the managed case.
    raw = kernel.snapshot()
    kernel.set_checkpoint_metadata(control(0))
    saved = kernel.snapshot()
    assert unwrap_checkpoint(saved) == (raw, control(0))
    path = tmp_path / "metadata.ckpt"
    path.write_bytes(saved)
    kernel.set_checkpoint_metadata(control(1))
    after = kernel.snapshot()  # Same simulation step, different committed revision.
    kernel.restore(saved)
    assert kernel.checkpoint_metadata() == control(0)
    kernel.restore(after)
    assert kernel.checkpoint_metadata() == control(1)
    kernel.restore(raw)
    assert kernel.checkpoint_metadata() == {}
    kernel.restore(saved)
    kernel.step(8)

    resumed = build(replace(spec, resume={"checkpoint": str(path)}))
    assert resumed._simulation is None
    assert resumed.checkpoint_metadata() == control(0)  # Getter must load lazy resume.
    assert resumed.current_step == 7
    resumed.step(8)
    np.testing.assert_array_equal(resumed.positions(), kernel.positions())
    np.testing.assert_array_equal(resumed.box_vectors(), kernel.box_vectors())
    assert resumed.boost_potentials() == kernel.boost_potentials()
    assert resumed.snapshot() == kernel.snapshot()
