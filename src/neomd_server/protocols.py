"""Closed scientific protocol compilation; see docs/scientific-protocol.md."""

from __future__ import annotations

import copy
import difflib
import json
import math
import sysconfig
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping
from uuid import UUID
from xml.etree import ElementTree

import jsonschema
import numpy as np

import neomd.colvars  # noqa: F401  (trusted built-in vocabulary)
import neomd.restraints  # noqa: F401
from neomd.plan import Plan
from neomd.prepared import file_digest

from .contract import ContractError, canonical_bytes, content_hash

__all__ = ["CompiledProtocol", "CoordinateReference", "ScientificLimits", "compile_protocol",
           "validate_protocol", "particle_identity_hash"]


@dataclass(frozen=True)
class CoordinateReference:
    """Host-authorized immutable coordinates; path never comes from protocol JSON.

    Metadata follows coordinate-reference.schema.json. Bytes are frame-major,
    then System particle index, then xyz; little-endian float64 nanometres.
    The host must keep the file immutable throughout compilation/execution.
    """

    metadata: Mapping
    path: Path


@dataclass(frozen=True)
class ScientificLimits:
    """Required host resource guards, independent of scientific wire semantics."""

    max_bias_grid_points: int
    max_reference_frames: int
    max_reference_bytes: int

    def __post_init__(self):
        for value in (self.max_bias_grid_points, self.max_reference_frames, self.max_reference_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("scientific resource limits must be positive integers")


@dataclass(frozen=True)
class CompiledProtocol:
    plan: Plan
    cv_names: Mapping[str, str]
    restraint_names: Mapping[str, str]
    steering_names: Mapping[str, str]
    content_sha256: str
    resource_requirements: Mapping[str, int]


@lru_cache(maxsize=2)
def _validator(name):
    root = Path(__file__).resolve().parents[2] / "contracts/scientific-protocol/v1"
    if not root.exists():
        root = Path(sysconfig.get_path("data")) / "share/neodynamics/contracts/scientific-protocol/v1"
    return jsonschema.Draft202012Validator(
        json.loads((root / f"{name}.schema.json").read_text()),
        format_checker=jsonschema.FormatChecker())


def _shape_problems(value, name):
    problems = []
    for error in _validator(name).iter_errors(value):
        path = "/" + "/".join(map(str, error.absolute_path))
        # Do not include submitted values, paths, expressions or URLs in errors.
        message = {"additionalProperties": "unexpected fields; use the versioned scientific vocabulary",
                   "required": "missing required fields: " + ", ".join(error.validator_value)
                   if isinstance(error.validator_value, list) else "missing required field"}.get(
                       error.validator, f"does not satisfy {error.validator or 'closed schema'}")
        if error.validator == "additionalProperties" and isinstance(error.instance, dict):
            known = error.schema.get("properties", {})
            for key in error.instance.keys() - known.keys():
                matches = difflib.get_close_matches(str(key), known, n=1)
                if matches:
                    message += f"; did you mean {matches[0]}?"
        problems.append((path, message))
    try:
        canonical_bytes(value)
    except (ValueError, TypeError):
        problems.append(("/", "must be finite, interoperable canonical JSON"))
    return problems


def validate_protocol(descriptor):
    """Collect closed-schema errors without reading files or writing artifacts."""
    problems = _shape_problems(descriptor, "protocol")
    if problems:
        raise ContractError(problems)
    return descriptor


def particle_identity_hash(manifest):
    """SHA256 of JCS ordered System-row identity UUID strings, including sites."""
    return content_hash([row["identity_id"] for row in manifest["particles"]])


def _artifact(directory, manifest, name):
    root = Path(directory).resolve(strict=True)
    if name not in manifest["artifacts"]:
        raise ValueError("required prepared artifact is absent")
    path = (root / name).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("prepared artifact is outside its trusted directory")
    expected = manifest["artifacts"][name]
    if path.stat().st_size != expected["size_bytes"] or file_digest(path) != expected["sha256"]:
        raise ValueError("prepared artifact bytes differ from verified manifest")
    return path


_GROUPS = frozenset(["grp1", "grp2", "grp3", "grp4", "grp1_idx", "grp2_idx", "grp3_idx",
                    "grp4_idx", "min1_idx1", "min2_idx1", "min_idx2", "particles", "restr_grp",
                    "gate_grp", "pocket_grp", "vec_grp1", "vec_grp2",
                    *[f"{part}_grp{i}" for part in ("rec", "lig") for i in range(1, 4)]])
_RAMP_KEYS = frozenset(["restr_k", "restr_k_per_atom", "min_nm", "max_nm", "min_degree",
                        "max_degree", "order", "maxRMSD_nm", "ref_position_nm"])


def _is_ramp(key, value):
    return key in _RAMP_KEYS and isinstance(value, list) and (
        key != "ref_position_nm" or isinstance(value[0], list))


def _anchors(value):
    return np.asarray(value if isinstance(value, list) else [value], dtype=float)


def _bounds(problems, path, parameters, kind, steering):
    pairs = [("min_nm", "max_nm"), ("min_degree", "max_degree")]
    pairs += [(f"min_{axis}_nm", f"max_{axis}_nm") for axis in "xyz"]
    pairs += [("lower_wall_nm", "upper_wall_nm")]
    for lower, upper in pairs:
        if lower in parameters and upper in parameters and kind != "dihedral":
            lo, hi = _anchors(parameters[lower]), _anchors(parameters[upper])
            def times_for(values):
                if not steering or len(values) == 1:
                    return np.linspace(0, 1, len(values))
                if len(values) > steering + 1:
                    return np.linspace(0, 1, len(values))
                # Native SMD uses floor-sized segments and extends the last.
                segment = steering // (len(values) - 1)
                return np.asarray([i * segment for i in range(len(values) - 1)] + [steering]) / steering

            lo_times, hi_times = times_for(lo), times_for(hi)
            times = np.unique(np.concatenate([lo_times, hi_times]))
            if np.any(np.interp(times, lo_times, lo) > np.interp(times, hi_times, hi)):
                problems.append((path, f"{lower} must not exceed {upper}, including ramp anchors"))
    bounds = [k for k in parameters if k.startswith(("min_", "max_"))]
    if kind in {"distance", "angle", "dist_ref_position", "xyz_box", "distances_pair"} and not bounds:
        problems.append((path, "at least one restraint wall is required"))
    if kind == "dist_ref_position" and sum(k in parameters for k in ("restr_k", "restr_k_per_atom")) != 1:
        problems.append((path, "choose exactly one of restr_k and restr_k_per_atom"))
    if kind == "funnel" and (parameters["width"] < 0 or parameters["buffer"] < 0):
        problems.append((path, "funnel width and buffer must be nonnegative"))
    if kind == "boresch" and (parameters["r0_nm"] <= 0 or any(
            not 0 < parameters[key] < 180 for key in ("thetaA0_degree", "thetaB0_degree"))):
        problems.append((path, "Boresch distance must be positive and equilibrium angles strictly between 0 and 180 degrees"))
    if steering:
        for key, value in parameters.items():
            if _is_ramp(key, value) and len(value) > steering + 1:
                problems.append((path + "/" + key, "ramp has more segments than simulation steps"))


class _ReferenceLimitError(ValueError):
    pass


def _reference_data(reference, kind, manifest, directory, resolver, limits, requirements):
    count = manifest["particle_count"]

    def reserve(frame_count):
        size = frame_count * count * 3 * 8
        if frame_count > limits.max_reference_frames:
            raise _ReferenceLimitError("reference frame count exceeds the host scientific limit")
        if requirements["reference_coordinate_bytes"] + size > limits.max_reference_bytes:
            raise _ReferenceLimitError("total reference coordinate bytes exceed the host scientific limit")
        requirements["reference_coordinate_bytes"] += size
        requirements["max_reference_frames"] = max(requirements["max_reference_frames"], frame_count)

    if reference["kind"] == "prepared_start":
        if kind in {"path_s", "path_z"}:
            raise ValueError("path CV requires a registered reference with at least two frames")
        reserve(1)
        path = _artifact(directory, manifest, manifest["coordinates"]["artifact"])
        return np.memmap(path, mode="r", dtype="<f8", shape=(1, count, 3))
    if resolver is None:
        raise ValueError("registered reference resolver is required")
    result = resolver(reference["reference_id"])
    if not isinstance(result, CoordinateReference):
        raise ValueError("reference resolver must return CoordinateReference")
    meta = dict(result.metadata)
    if _shape_problems(meta, "coordinate-reference"):
        raise ValueError("registered coordinate metadata fails its versioned schema")
    if (meta["reference_id"] != reference["reference_id"]
            or meta["prepared_revision_id"] != manifest["prepared_revision_id"]
            or meta["particle_identity_sha256"] != particle_identity_hash(manifest)
            or meta["particle_count"] != count):
        raise ValueError("registered reference identity or prepared revision does not match")
    # JSON Schema integers include lexical 2.0; use integral dimensions after
    # validation so an interoperable JSON number cannot become a slice error.
    frame_count = int(meta["frame_count"])
    expected_size = frame_count * count * 3 * 8
    reserve(frame_count)
    path = Path(result.path)
    if (meta["size_bytes"] != expected_size or path.stat().st_size != expected_size
            or file_digest(path) != meta["content_sha256"]):
        raise ValueError("registered coordinate bytes do not match metadata")
    frames = np.memmap(path, mode="r", dtype="<f8", shape=(frame_count, count, 3))
    if any(not np.isfinite(frame).all() for frame in frames):
        raise ValueError("registered reference contains nonfinite coordinates")
    if kind in {"path_s", "path_z"}:
        if "frame_index" in reference or len(frames) < 2:
            raise ValueError("path reference requires all frames and at least two frames")
    else:
        if "frame_index" not in reference and len(frames) != 1:
            raise ValueError("multi-frame RMSD reference requires explicit frame_index")
        index = int(reference.get("frame_index", 0))
        if index >= len(frames):
            raise ValueError("reference frame_index is out of range")
        frames = frames[index:index + 1]
    return frames


def _write_reference(path, frames):
    """Coordinate-only full-System-row mmCIF for engine reference readers."""
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write("data_reference\nloop_\n_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n_atom_site.pdbx_PDB_model_num\n")
            for model, frame in enumerate(frames, 1):
                for row in frame:
                    handle.write(" ".join(format(float(x) * 10, ".17g") for x in row) + f" {model}\n")
            handle.write("#\n")
            handle.flush()
            if path.exists():
                if file_digest(path) != file_digest(temporary):
                    raise ValueError("immutable compiled reference already exists with different bytes")
            else:
                # Atomic no-replace publication, safe for same-descriptor retries.
                os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def compile_protocol(descriptor, *, prepared_manifest, prepared_directory, limits: ScientificLimits,
                     output_directory, protocol_directory,
                     reference_resolver: Callable[[str], CoordinateReference] | None = None,
                     materialize_references: bool = True):
    """Resolve stable IDs and construct an immutable Plan from safe science JSON.

    The caller owns authorization, schema/hash verification of the prepared
    bundle, XML type approval and actual System validation, immutable artifact
    lifetime, and trusted local directories. This function neither certifies a
    force field nor admits a job. Context creation remains an execution gate.
    It does not alter the prepared pair; the host injects its verified full-N
    initial coordinates and explicit topology mapping through KernelSpec.
    """
    validate_protocol(descriptor)
    data = copy.deepcopy(descriptor)
    manifest = prepared_manifest
    problems = []
    if not isinstance(limits, ScientificLimits):
        raise TypeError("host must supply ScientificLimits")
    requirements = {"bias_grid_points": 0, "max_reference_frames": 0, "reference_coordinate_bytes": 0}
    if data["method"] in {"metadynamics", "opes"}:
        requirements["bias_grid_points"] = math.prod(int(cv["parameters"]["bins"]) for cv in data["colvars"])
        if requirements["bias_grid_points"] > limits.max_bias_grid_points:
            problems.append(("/colvars", "combined bias grid points exceed the host scientific limit"))
    if data["prepared_revision_id"] != manifest["prepared_revision_id"]:
        problems.append(("/prepared_revision_id", "does not match the authorized prepared bundle"))
    pending = manifest["parameters"].get("system_modifications")
    has_pending = any(value not in (None, {}, []) for value in pending.values()) if isinstance(pending, dict) else bool(pending)
    if has_pending:
        problems.append(("/prepared", "pending System modifications must be materialized and exported before registration"))
    particles = {row["identity_id"]: row for row in manifest["particles"]}
    digest = content_hash(data)
    protocol_root = Path(protocol_directory).resolve()
    generated = {}
    names = {section: {} for section in ("colvars", "restraints", "steering")}
    translated = {section: {} for section in names}
    seen_ids = set()

    def convert(parameters, path):
        result = {}
        for key, value in parameters.items():
            if key in _GROUPS:
                indices = []
                for i, identity in enumerate(value):
                    particle = particles.get(identity)
                    if particle is None:
                        problems.append((f"{path}/{key}/{i}", "unknown prepared AtomId"))
                    elif not particle["control_eligible"] or particle["identity_kind"] != "atom":
                        problems.append((f"{path}/{key}/{i}", "selection requires an eligible massive prepared atom"))
                    else:
                        indices.append(particle["particle_index"])
                result[key] = indices
            elif key == "params":
                result[key] = [convert(pair, f"{path}/params/{i}") for i, pair in enumerate(value)]
                for i, pair in enumerate(value):
                    _bounds(problems, f"{path}/params/{i}", pair, "distances_pair", False)
            elif key != "reference":
                result[key] = value
        return result

    for section, prefix in (("colvars", "cv_"), ("restraints", "r_"), ("steering", "s_")):
        for i, definition in enumerate(data.get(section, [])):
            path = f"/{section}/{i}"
            identity, kind, parameters = definition["id"], definition["type"], definition["parameters"]
            if identity in seen_ids:
                problems.append((path + "/id", "definition UUID must be unique across CVs and restraints"))
            seen_ids.add(identity)
            name = prefix + UUID(identity).hex
            names[section][identity] = name
            spec = {"type": kind, **convert(parameters, path + "/parameters")}
            if section == "colvars":
                suffix = "_degree" if kind in {"angle", "dihedral"} else "" if kind in {"coordination", "path_s"} else "_nm"
                if parameters["min_cv" + suffix] >= parameters["max_cv" + suffix]:
                    problems.append((path + "/parameters", "CV grid lower bound must be below upper bound"))
                if kind == "coordination" and parameters.get("mm", 12) <= parameters.get("nn", 6):
                    problems.append((path + "/parameters", "coordination denominator exponent must exceed numerator exponent"))
            else:
                _bounds(problems, path + "/parameters", parameters, kind,
                        data["steps"] if section == "steering" else False)
            if "reference" in parameters:
                try:
                    frames = _reference_data(parameters["reference"], kind, manifest, prepared_directory,
                                             reference_resolver, limits, requirements)
                    target = protocol_root / digest / f"{name}.pdbx"
                    generated[target] = frames
                    spec["ref_path_file" if kind in {"path_s", "path_z"} else "ref_pos_file"] = str(target)
                except _ReferenceLimitError as error:
                    problems.append((path + "/parameters/reference", str(error)))
                except (ValueError, OSError, KeyError):
                    problems.append((path + "/parameters/reference", "reference unavailable, mismatched, malformed or incompatible with this CV; verify frame selection and registered metadata"))
            translated[section][name] = spec

    physics = data["physics"]
    raw = {"method": data["method"], "temperature": physics["temperature_K"], "seed": physics["seed"],
           "integrator": {"integrator_name": "LangevinIntegrator", "dt": physics["timestep_ps"],
                          "friction_coeff": physics["friction_per_ps"]},
           "input_files": {}, "output": {"output_dir": str(Path(output_directory).resolve()),
                                          "report_interval": 1000}}
    if "barostat" in physics:
        if manifest["box_vectors_nm"] is None:
            problems.append(("/physics/barostat", "barostat requires a periodic prepared System"))
        raw["barostat"] = {"pressure": physics["barostat"]["pressure_bar"],
                           "frequency": physics["barostat"]["frequency_steps"]}
    for key, value in data.get("reporting", {}).items():
        raw["output"][key.removesuffix("_steps")] = value
    if translated["restraints"]:
        raw["restraint"] = translated["restraints"]
    if "steps" in data:
        raw["steps"] = data["steps"]
    if data["method"] == "min":
        raw["min_params"] = {"tolerance": data["minimization"]["tolerance_kj_mol_nm"],
                             "maxiter": data["minimization"]["max_iterations"]}
    if data["method"] == "smd":
        raw["smd"] = translated["steering"]
    if data["method"] in {"metadynamics", "opes"}:
        method = data["method"]
        settings = data[method]
        ids = settings["cv_ids"]
        if set(ids) != set(names["colvars"]):
            problems.append((f"/{method}/cv_ids", "must reference every defined CV exactly once"))
        else:
            raw["colvars"] = {names["colvars"][identity]: translated["colvars"][names["colvars"][identity]] for identity in ids}
        mapping = {"bias_factor": "biasFactor", "height_kj_mol": "height", "frequency_steps": "frequency",
                   "update_context_frequency_steps": "update_context_frequency", "pace_steps": "pace", "barrier_kj_mol": "barrier"}
        raw["meta_set" if method == "metadynamics" else "opes_set"] = {
            mapping.get(key, key): value for key, value in settings.items() if key != "cv_ids"}
    if data["method"] == "gamd":
        settings = data["gamd"]
        if settings["calibration_steps"] >= data["steps"] or settings["calibration_interval_steps"] > settings["calibration_steps"]:
            problems.append(("/gamd", "calibration must sample at least once and end before total steps"))
        raw["gamd"] = {key.removesuffix("_kj_mol").removesuffix("_steps") if key not in {"calibration_steps"} else key: value for key, value in settings.items()}
    try:
        topology = [name for name in manifest["artifacts"] if name in {"topology.pdb", "topology.pdbx"}]
        if len(topology) != 1:
            raise ValueError("one prepared topology artifact is required")
        raw["input_files"] = {"complex": str(_artifact(prepared_directory, manifest, topology[0])),
                              "system": str(_artifact(prepared_directory, manifest, "system.xml"))}
        if data["method"] == "rbfe":
            windows = data["rbfe"]["windows"]
            keys = set(windows[0])
            if any(set(window) != keys for window in windows):
                problems.append(("/rbfe/windows", "all lambda windows must use identical parameter names"))
            if len({content_hash(window) for window in windows}) != len(windows):
                problems.append(("/rbfe/windows", "lambda windows must be distinct"))
            tree = ElementTree.parse(raw["input_files"]["system"])
            available = {node.attrib["name"] for node in tree.findall(".//GlobalParameters/Parameter") if "name" in node.attrib}
            if not keys <= available:
                problems.append(("/rbfe/windows", "lambda parameters are not declared by the prepared System"))
            if raw["output"]["report_interval"] <= 0:
                problems.append(("/reporting/report_interval_steps", "RBFE requires positive energy reporting cadence"))
            raw["alchemical"] = {"lambda_values": windows[0], "ladder": windows}
    except (ValueError, OSError, KeyError, ElementTree.ParseError):
        problems.append(("/prepared", "trusted prepared artifacts are absent, changed or malformed"))
    if problems:
        raise ContractError(problems)
    # Core Plan retains its collect-all immutable validation contract. File paths
    # are host-constructed; generated references are published only after it passes.
    plan = Plan(raw)
    if materialize_references:
        for path, frames in generated.items():
            _write_reference(path, frames)
    return CompiledProtocol(plan, MappingProxyType(names["colvars"]), MappingProxyType(names["restraints"]),
                            MappingProxyType(names["steering"]), digest, MappingProxyType(requirements))
