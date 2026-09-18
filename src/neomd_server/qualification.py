"""Actual tiny-system runtime qualification, run only in a spawned probe process."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import struct
import tempfile
from dataclasses import replace
from importlib import resources
from pathlib import Path

METHODS = ("min", "eq", "md", "prod", "metadynamics", "smd", "opes", "gamd", "rbfe")
_CV_ID = "00000000-0000-4000-8000-000000000101"
_STEERING_ID = "00000000-0000-4000-8000-000000000102"
_MANIFEST_SHA256 = "c25b295179d0bee7157e42d92945c3a4917b08883eadbfaf44caa7d736061623"
_logger = logging.getLogger("neomd_server.qualification")


def _descriptor(manifest, method):
    atoms = [row["atom_id"] for row in manifest["atoms"]]
    result = {"schema_version": 1, "prepared_revision_id": manifest["prepared_revision_id"],
              "method": method, "steps": 8,
              "physics": {"temperature_K": 298, "timestep_ps": .0005, "friction_per_ps": 1, "seed": 42},
              "reporting": {"report_interval_steps": 2}}
    if method == "min":
        del result["steps"]
        result["minimization"] = {"tolerance_kj_mol_nm": 10, "max_iterations": 2}
    if method in {"metadynamics", "opes"}:
        result["colvars"] = [{"id": _CV_ID, "type": "distance", "parameters": {
            "grp1_idx": [atoms[0]], "grp2_idx": [atoms[1]], "min_cv_nm": 0,
            "max_cv_nm": 2, "biasWidth_nm": .1, "bins": 10}}]
        result[method] = ({"cv_ids": [_CV_ID], "bias_factor": 5, "height_kj_mol": .1, "frequency_steps": 2}
                          if method == "metadynamics" else
                          {"cv_ids": [_CV_ID], "pace_steps": 2, "barrier_kj_mol": 20})
    if method == "smd":
        result["steering"] = [{"id": _STEERING_ID, "type": "distance", "parameters": {
            "grp1": [atoms[0]], "grp2": [atoms[1]], "restr_k": 10, "max_nm": [.4, .6]}}]
    if method == "gamd":
        result["gamd"] = {"mode": "total", "sigma0_kj_mol": 6, "calibration_steps": 4,
                          "calibration_interval_steps": 2, "frequency_steps": 2}
    if method == "rbfe":
        # The serialized fixture actually owns this lambda global and force.
        result["rbfe"] = {"windows": [{"lambda_sterics": 0}, {"lambda_sterics": 1}]}
    return result


def _check_frame(session):
    frame = session.frame()
    if (frame["particle_count"] != 6 or frame["coordinate_unit"] != "nm"
            or frame["particle_indices"] != list(range(6))
            or len(frame["positions_nm"]) != 6
            or any(len(row) != 3 or any(not math.isfinite(x) for x in row)
                   for row in frame["positions_nm"])):
        raise RuntimeError("qualification produced an invalid frame")
    return frame


def _exercise_controls(session, commands, parameters):
    state = session.status()
    commands.update(state["controls"])
    parameters.update(state["parameters"])
    if "pause" in state["controls"]:
        session.command({"type": "pause"})
        paused = session.status()
        commands.update(paused["controls"])
        frame = _check_frame(session)
        if session.advance(1)["step"] != paused["step"] or _check_frame(session) != frame:
            raise RuntimeError("qualification pause advanced the engine")
        session.command({"type": "resume"})
    if "checkpoint" in state["controls"]:
        session.command({"type": "checkpoint"})
    if "set_forces" in state["controls"]:
        session.command({"type": "set_forces", "indices": [0], "vectors": [[.1, 0, 0]],
                         "unit": "kJ/mol/nm", "ttl_steps": 2, "ttl_ms": 100})
        session.command({"type": "clear_forces"})
        session.command({"type": "set_restraint", "name": "qualification", "indices": [0],
                         "targets": [_check_frame(session)["positions_nm"][0]], "k": .1,
                         "position_unit": "nm", "k_unit": "kJ/mol/nm^2"})
        session.command({"type": "clear_restraints", "name": "qualification"})
    for name, descriptor in state["parameters"].items():
        value = True if descriptor["unit"] == "boolean" else .45 if descriptor["unit"] == "nm" else .1
        session.command({"type": "set_parameter", "name": name, "value": value,
                         "unit": descriptor["unit"], "replace_schedule": descriptor.get("replaces_schedule", False)})


def _probe_method(method, manifest, source, output, platform, device_index, positions):
    from neomd.kernel._bootstrap import ensure_adapters
    from neomd.kernel.port import KernelFactory

    from .driver import open_session
    from .protocols import ScientificLimits, compile_protocol

    compiled = compile_protocol(_descriptor(manifest, method), prepared_manifest=manifest,
                                prepared_directory=source, limits=ScientificLimits(100, 2, 1024),
                                output_directory=output / "output", protocol_directory=output / "protocol")
    ensure_adapters()

    def kernel_factory(spec):
        return KernelFactory.create(replace(spec, device_index=device_index,
                                            initial_positions_nm=positions,
                                            topology_particle_indices=tuple(manifest["topology_particle_indices"])))

    session = open_session(compiled.plan, kernel_factory=kernel_factory, platform=platform)
    commands, parameters, phases = set(), {}, set()
    try:
        for _ in range(32):
            state = session.status()
            _check_frame(session)
            if state["state"] == "completed":
                if method == "gamd" and not {"calibration", "production"} <= phases:
                    raise RuntimeError("qualification skipped a GaMD phase")
                if method == "rbfe" and ("window_boundary" not in phases or state["window_index"] != 1):
                    raise RuntimeError("qualification skipped an RBFE window")
                # Steering names are scoped to this synthetic protocol. Live
                # sessions publish their own names; never advertise fixture IDs.
                if method == "smd":
                    parameters = {}
                return {"availability": "available", "qualification": platform,
                        "reason": None, "controls": {"commands": sorted(commands), "parameters": parameters}}
            if state["state"] in {"stopped", "failed", "closed"}:
                raise RuntimeError("qualification did not complete")
            if state["phase"] not in phases:
                phases.add(state["phase"])
                _exercise_controls(session, commands, parameters)
            session.advance(2)
        raise RuntimeError("qualification exceeded its finite step budget")
    finally:
        session.close()


def _unavailable(reason):
    return {"availability": "unavailable", "qualification": "unqualified", "reason": reason, "controls": None}


def probe_methods(*, platform: str, device_index: str, cpu_threads: int, work_directory: str) -> dict:
    """Qualify actual configured hardware with all built-in methods on six atoms.

    The host runs this in a disposable spawned process and owns its wall-clock
    deadline/termination. This synchronous function never chooses deployment
    admission policy, reports fake results or makes throughput claims. Returned
    controls aggregate observed fixture phases; a live session's current snapshot
    remains authoritative, including its own named CVs and steering definitions.
    """
    if platform not in {"cpu", "cuda"}:
        raise ValueError("qualification requires cpu or cuda")
    if (not isinstance(device_index, str) or not device_index.isascii()
            or not device_index.isdecimal() or len(device_index) > 10):
        raise ValueError("qualification requires one explicit device index")
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or not 1 <= cpu_threads <= 1024:
        raise ValueError("qualification requires a bounded CPU thread count")
    work = Path(work_directory)
    if not work.is_absolute() or not work.is_dir() or work.is_symlink():
        raise ValueError("qualification requires an owned absolute work directory")
    # Safe only because this API is explicitly a disposable child-process entry.
    os.environ["OPENMM_CPU_THREADS"] = str(cpu_threads)
    result = {method: _unavailable("qualification_missing") for method in METHODS}
    try:
        from neomd.system import verify_prepared_system

        resource = resources.files("neomd_server").joinpath("fixtures/qualification")
        with resources.as_file(resource) as source, tempfile.TemporaryDirectory(prefix="qualification-", dir=work) as scratch:
            if hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest() != _MANIFEST_SHA256:
                raise ValueError("packaged qualification fixture changed")
            manifest = verify_prepared_system(source / "manifest.json")
            raw = (source / "coordinates.f64le").read_bytes()
            if manifest["particle_count"] != 6 or len(raw) != 6 * 3 * 8:
                raise ValueError("invalid packaged qualification fixture")
            coordinates = struct.unpack("<18d", raw)
            positions = tuple(tuple(coordinates[i:i + 3]) for i in range(0, 18, 3))
            for method in METHODS:
                try:
                    result[method] = _probe_method(method, manifest, source, Path(scratch) / method,
                                                   platform, device_index, positions)
                except ImportError:
                    result[method] = _unavailable("dependency_missing")
                except Exception as error:
                    _logger.warning("Method qualification failed for %s: %s", method, type(error).__name__)
    except ImportError:
        result = {method: _unavailable("dependency_missing") for method in METHODS}
    except Exception as error:
        _logger.warning("Qualification fixture unavailable: %s", type(error).__name__)
    return result
