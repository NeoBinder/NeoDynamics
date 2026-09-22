"""GaMD frame timestamps use the effective integrator timestep."""

import pytest

from neomd.kernel.fake import FakeKernel
from neomd.kernel.port import BoostChannelIR, KernelSpec
from neomd.probes import GamdProbe, KernelView
from neomd.sinks import MemorySink


@pytest.mark.parametrize("integrator,explicit_dt,expected", [
    ({}, None, 0.008),
    ({"dt": 0.002}, None, 0.004),
    ({}, 0.003, 0.006),
])
def test_effective_timestamp_with_default_explicit_and_supplied_timestep(integrator, explicit_dt, expected):
    kernel = FakeKernel(KernelSpec(kind="fake", integrator=integrator))
    kernel.install_boost([BoostChannelIR(label="total", groups=())])
    kernel.step(2)
    sink = MemorySink()
    GamdProbe(sink, 1, ["total"], dt_ps=explicit_dt).observe(KernelView(kernel, 2))
    header, row = sink.read_bytes("gamd.tsv").decode().splitlines()
    assert float(row.split("\t")[header.split("\t").index("time_ps")]) == pytest.approx(expected)


@pytest.mark.parametrize("dt", [0, -1, float("nan"), float("inf")])
def test_invalid_explicit_timestep_is_rejected_before_writing(dt):
    sink = MemorySink()
    with pytest.raises(ValueError, match="dt_ps"):
        GamdProbe(sink, 1, ["total"], dt_ps=dt)
    assert not sink.names()
