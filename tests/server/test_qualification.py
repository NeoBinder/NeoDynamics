"""Actual disposable-process qualification through the public probe interface."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from neomd_server.qualification import METHODS, probe_methods


def test_all_builtin_methods_are_qualified_by_real_cpu_in_a_separate_process(tmp_path):
    import sys

    pytest.importorskip("rfc8785")
    script = """
import json, sys
from neomd_server.qualification import probe_methods
result = probe_methods(platform='cpu', device_index='0', cpu_threads=1, work_directory=sys.argv[1])
print(json.dumps(result))
"""
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], text=True,
                             capture_output=True, timeout=120,
                             env={**os.environ, "PYTHONNOUSERSITE": "1"})
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert set(result) == set(METHODS)
    contract = json.loads((Path(__file__).parents[2] / "contracts/imd/v1/protocol.schema.json").read_text())
    schema = contract["$defs"]["RuntimeCapabilities"]["properties"]["methods"]["items"]
    for method, entry in result.items():
        assert entry["availability"] == "available", (method, entry, process.stderr)
        assert entry["qualification"] == "cpu" and entry["reason"] is None
        assert {"pause", "resume", "checkpoint", "set_forces", "set_restraint"} <= set(entry["controls"]["commands"])
        Draft202012Validator(schema).validate({"method": method, **entry})
    assert result["gamd"]["controls"]["parameters"]["sigma0"]["phase"] == "calibration"
    assert result["opes"]["controls"]["parameters"]["deposition_enabled"]["unit"] == "boolean"
    assert "set_parameter" in result["smd"]["controls"]["commands"]
    assert result["smd"]["controls"]["parameters"] == {}
    assert result["rbfe"]["controls"]["parameters"] == {}
    assert list(tmp_path.iterdir()) == []


def test_missing_engine_dependency_never_reports_qualified_methods(tmp_path):
    import sys

    script = """
import json, sys
sys.modules['openmm'] = None
from neomd_server.qualification import probe_methods
print(json.dumps(probe_methods(platform='cpu', device_index='0', cpu_threads=1, work_directory=sys.argv[1])))
"""
    process = subprocess.run([sys.executable, "-c", script, str(tmp_path)], text=True,
                             capture_output=True, timeout=30,
                             env={**os.environ, "PYTHONNOUSERSITE": "1"})
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout)
    assert set(result) == set(METHODS)
    assert all(value == {"availability": "unavailable", "qualification": "unqualified",
                         "reason": "dependency_missing", "controls": None} for value in result.values())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("options", [
    {"platform": "fake"}, {"platform": "CPU"}, {"device_index": "0,1"},
    {"device_index": "secret/path"}, {"cpu_threads": 0}, {"cpu_threads": True},
    {"work_directory": "relative"},
])
def test_probe_requires_an_explicit_real_platform_and_host_bounds(tmp_path, options):
    args = {"platform": "cpu", "device_index": "0", "cpu_threads": 1, "work_directory": str(tmp_path), **options}
    with pytest.raises(ValueError):
        probe_methods(**args)
    assert list(tmp_path.iterdir()) == []
