"""The mlcv featurizer: run-dir data -> named feature columns (ADR-0006).

Registry kinds reuse the PUBLIC cv registry's evaluate implementations; local
kinds ``contact`` (switching function on the group-COM distance) and ``tape``
(passthrough column aligned by step, nan where missing) — see featurize.
Config reference: docs/methods/mlcv.md.  Deterministic, numpy-only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from neomd.errors import (
    ConfigKeyError,
    ConfigValueError,
    NeoUserError,
    PlanValidationErrors,
    suggest,
)

from ._readers import (
    DCD_FILENAME,
    Frames,
    read_dcd_frames,
    read_step_tsv,
    read_system_masses,
    run_system_xml,
)

__all__ = [
    "FEATURE_FORMAT_VERSION",
    "FEATURE_TYPES",
    "MASS_DEPENDENT_TYPES",
    "FeaturizeResult",
    "validate_featurize_config",
    "featurize",
]

#: features.npz format version (bump on any layout change)
FEATURE_FORMAT_VERSION = 1

#: the registry-backed feature kinds (colvars.py vocabulary, reused as-is)
_REGISTRY_TYPES = (
    "distance", "dihedral", "angle", "min_distances", "distance_ref",
    "rmsd", "coordination", "path_s", "path_z",
)

#: the local feature kinds (no registry entry; see module docstring)
_LOCAL_TYPES = ("contact", "tape")

FEATURE_TYPES = _REGISTRY_TYPES + _LOCAL_TYPES

#: feature kinds whose evaluate consumes masses (COM-based geometry)
MASS_DEPENDENT_TYPES = frozenset({
    "distance", "dihedral", "angle", "min_distances", "distance_ref",
    "contact",
})

#: natural unit per feature kind (the cv registry's own unit convention)
_UNITS = {
    "distance": "nm", "min_distances": "nm", "distance_ref": "nm",
    "rmsd": "nm", "path_z": "nm", "angle": "degree", "dihedral": "degree",
    "coordination": "dimensionless", "path_s": "dimensionless",
    "contact": "dimensionless", "tape": "unknown",
}

#: grid keys of the cv schema — required for biasing, irrelevant for features
_GRID_KEYS = frozenset({
    "min_cv_nm", "max_cv_nm", "biasWidth_nm", "min_cv_degree", "max_cv_degree",
    "biasWidth_degree", "min_cv", "max_cv", "biasWidth", "bins", "is_period",
})

#: featurize-config top-level keys
_CONFIG_KEYS = frozenset({
    "run_dirs", "trajectory", "system_xml", "uniform_masses", "stride",
    "output", "features",
})

_TAPE_FILES = ("colvar.tsv", "restraint.tsv", "smd.tsv")


@dataclass(frozen=True)
class FeaturizeResult:
    """What ``featurize`` did (the CLI summary is built from this)."""

    output: str
    n_frames: int
    feature_names: list[str]
    units: list[str]
    runs: list[str]  # the run dirs (or the trajectory file) in row order


# ---------------------------------------------------------------------------
# validation — collect-all with key paths + did-you-mean (the house style)
# ---------------------------------------------------------------------------


def _known_required_keys(feature_type: str) -> dict:
    """required-key vocabulary of one registry feature kind (grid keys
    dropped — they configure biasing, not geometry)."""
    if feature_type in _LOCAL_TYPES:
        if feature_type == "contact":
            return {"grp1_idx": "index group", "grp2_idx": "index group",
                    "r0": "float nm"}
        return {"tape": "artifact filename", "column": "column label"}
    from neomd import (
        colvars,  # noqa: F401 (import = cv registration)
        registry,
    )

    entry = registry.get("cv", feature_type)
    return {key: doc for key, doc in entry.schema["required"].items()
            if key not in _GRID_KEYS}


def validate_featurize_config(config, *, source: str | None = None) -> list:
    """Every problem with a featurize config, in one pass (the collect-all
    discipline; renders through the errors.py family)."""
    errors: list[NeoUserError] = []
    if not isinstance(config, Mapping):
        return [ConfigValueError(
            "featurize config must be a mapping", value=config, source=source)]

    for key in config:
        if key not in _CONFIG_KEYS:
            errors.append(ConfigKeyError(
                f"unknown featurize config key {key!r}", key=key,
                value=key, source=source, known_keys=_CONFIG_KEYS))

    # -- frames source ----------------------------------------------------
    run_dirs = config.get("run_dirs")
    trajectory = config.get("trajectory")
    if run_dirs is None and trajectory is None:
        errors.append(ConfigKeyError(
            "featurize config needs 'run_dirs' (v2 run directories) or "
            "'trajectory' (a .dcd file)", key="run_dirs", source=source,
            known_keys=_CONFIG_KEYS, candidates=[]))  # suggesting the missing
            # key itself would be noise — the message names it
    if run_dirs is not None:
        if isinstance(run_dirs, (str, os.PathLike)):
            run_dirs = [run_dirs]
        if not isinstance(run_dirs, (list, tuple)) or not run_dirs:
            errors.append(ConfigValueError(
                "run_dirs must be a run-directory path or a non-empty list "
                "of them", key="run_dirs", value=run_dirs, source=source))
        else:
            for run_dir in run_dirs:
                if not os.path.isdir(os.fspath(run_dir)):
                    errors.append(ConfigValueError(
                        f"run_dirs entry is not a directory: {run_dir!s}",
                        key="run_dirs", value=str(run_dir), source=source))
                elif not os.path.exists(os.path.join(os.fspath(run_dir),
                                                    "manifest.json")):
                    errors.append(ConfigValueError(
                        f"run_dirs entry has no manifest.json (not a v2 run "
                        f"directory): {run_dir!s}", key="run_dirs",
                        value=str(run_dir), source=source))
    if trajectory is not None:
        if not isinstance(trajectory, str) or not os.path.isfile(trajectory):
            errors.append(ConfigValueError(
                f"trajectory must be an existing .dcd path, got {trajectory!r}",
                key="trajectory", value=trajectory, source=source))

    # -- masses ------------------------------------------------------------
    system_xml = config.get("system_xml")
    if system_xml is not None and not (
            isinstance(system_xml, str) and os.path.isfile(system_xml)):
        errors.append(ConfigValueError(
            f"system_xml must be an existing system.xml path, got "
            f"{system_xml!r}", key="system_xml", value=system_xml,
            source=source))
    uniform = config.get("uniform_masses", False)
    if not isinstance(uniform, bool):
        errors.append(ConfigValueError(
            f"uniform_masses must be a bool, got {uniform!r}",
            key="uniform_masses", value=uniform, source=source))

    # -- stride --------------------------------------------------------------
    stride = config.get("stride", 1)
    if not isinstance(stride, int) or isinstance(stride, bool) or stride < 1:
        errors.append(ConfigValueError(
            f"stride must be an int >= 1, got {stride!r}", key="stride",
            value=stride, source=source))

    # -- features ------------------------------------------------------------
    features = config.get("features")
    if features is None:
        errors.append(ConfigKeyError(
            "featurize config needs a 'features' mapping", key="features",
            source=source, known_keys=_CONFIG_KEYS, candidates=[]))
    elif not isinstance(features, Mapping) or not features:
        errors.append(ConfigValueError(
            "features must be a non-empty mapping of feature name -> spec",
            key="features", value=features, source=source))
    else:
        for name, spec in features.items():
            errors.extend(_validate_feature(name, spec, source=source))

    # -- output --------------------------------------------------------------
    output = config.get("output")
    if output is not None and not isinstance(output, str):
        errors.append(ConfigValueError(
            f"output must be a path string, got {output!r}", key="output",
            value=output, source=source))
    return errors


def _validate_feature(name, spec, *, source) -> list:
    """One feature entry: type known, required keys present, construction
    dry-runs (registry make_cv validates index lists / reference files)."""
    if not isinstance(name, str) or not name:
        return [ConfigValueError("feature names must be non-empty strings",
                                 key="features", value=name, source=source)]
    if not isinstance(spec, Mapping):
        return [ConfigValueError(
            f"feature {name!r} must be a mapping with a 'type' key",
            key="features", value=spec, source=source)]
    if "type" not in spec:
        return [ConfigKeyError(f"feature {name!r} has no 'type'",
                               key=f"features.{name}.type", source=source,
                               known_keys={"type", *_known_keys_any()})]
    feature_type = spec["type"]
    if feature_type not in FEATURE_TYPES:
        return [ConfigKeyError(
            f"unknown feature type {feature_type!r} in feature {name!r}",
            key=f"features.{name}.type", value=feature_type, source=source,
            known_keys=FEATURE_TYPES,
            # the misspelled word is the VALUE here, not the key — suggest
            # from it explicitly (the constructor suggests from `key`)
            candidates=suggest(feature_type, FEATURE_TYPES))]
    if feature_type == "tape":
        return _validate_tape_feature(name, spec, source=source)

    errors: list[NeoUserError] = []
    required = _known_required_keys(feature_type)
    for key in required:
        if key not in spec:
            errors.append(ConfigKeyError(
                f"feature {name!r} ({feature_type}) is missing required key "
                f"{key!r}", key=f"features.{name}.{key}", source=source,
                known_keys=set(required)))
    # dry-run construction (registry kinds only — make_cv parses index
    # lists, reference files, parameter shapes; its errors become
    # collected user errors).  The local contact kernel needs no build.
    if feature_type in _REGISTRY_TYPES:
        try:
            _build_registry_feature(name, spec)
        except (ValueError, KeyError) as error:
            errors.append(ConfigValueError(
                f"feature {name!r} does not construct: {error}",
                key=f"features.{name}", value=spec, source=source))
    return errors


def _known_keys_any() -> set:
    return {"type", *FEATURE_TYPES}


def _validate_tape_feature(name, spec, *, source) -> list:
    errors: list[NeoUserError] = []
    tape = spec.get("tape", "colvar.tsv")
    if tape not in _TAPE_FILES:
        errors.append(ConfigValueError(
            f"feature {name!r}: tape must be one of {list(_TAPE_FILES)}, "
            f"got {tape!r}", key=f"features.{name}.tape", value=tape,
            source=source))
    if "column" not in spec:
        errors.append(ConfigKeyError(
            f"feature {name!r} (tape) is missing required key 'column'",
            key=f"features.{name}.column", source=source,
            known_keys={"tape", "column"}))
    return errors


# ---------------------------------------------------------------------------
# feature construction + evaluation (registry reuse)
# ---------------------------------------------------------------------------


def _build_registry_feature(name: str, spec: Mapping):
    """(cv entry, CVIR) for one registry-backed feature — the PUBLIC
    make_cv route, so geometry, file readers and index parsing are the
    same code the simulation uses."""
    import neomd.colvars  # noqa: F401 (import = cv registration)
    from neomd import registry

    feature_type = spec["type"]
    entry = registry.get("cv", feature_type)
    feature_spec = {key: value for key, value in spec.items()
                    if key not in _GRID_KEYS and key != "type"}
    cv, _grid = entry.make_cv(name, feature_spec)
    return entry, cv


def _switching(distance: float, r0: float, nn: float, mm: float) -> float:
    """The coordination CV's rational switching function applied once (the
    contact feature's kernel — same expression, single distance).

    The raw kernel has a removable 0/0 at ``d == r0`` exactly (analytically
    1/(1+(d/r0)^nn) for the nn/mm=6/12 default); the featurizer evaluates
    it at hand-picked geometries, so the limit (nn/mm) is taken there."""
    x = distance / float(r0)
    den = 1.0 - x ** mm
    if den == 0.0:
        return float(nn / mm)
    return float((1.0 - x ** nn) / den)


def _feature_value(name: str, feature_type: str, spec: Mapping, positions,
                   masses, built) -> float:
    """One frame's feature value through the registry evaluate (or the
    local contact kernel)."""
    if feature_type == "contact":
        from neomd.colvars import _com

        com1 = _com(masses, positions, _parse_indices(spec["grp1_idx"]))
        com2 = _com(masses, positions, _parse_indices(spec["grp2_idx"]))
        d = float(np.linalg.norm(com1 - com2))
        return _switching(d, float(spec["r0"]),
                          float(spec.get("nn", 6)), float(spec.get("mm", 12)))
    entry, cv = built
    return float(entry.evaluate(positions, masses, cv))


def _parse_indices(value):
    """Index-group normalization (the colvars comma-string / list grammar)."""
    if isinstance(value, str):
        return [int(v) for v in value.split(",")]
    return [int(v) for v in value]


# ---------------------------------------------------------------------------
# the featurizer
# ---------------------------------------------------------------------------


def _frames_and_masses(config) -> tuple[Frames, np.ndarray, list[str], list]:
    """(frames, masses, run labels, per-run (start, stop) row ranges) from
    the config's frames source."""
    stride = int(config.get("stride", 1))
    trajectory = config.get("trajectory")
    system_xml = config.get("system_xml")
    uniform = bool(config.get("uniform_masses", False))

    if trajectory is not None:
        frames = read_dcd_frames(trajectory, stride=stride)
        if uniform:
            masses = np.ones(frames.n_atoms)
        elif system_xml is not None:
            masses = read_system_masses(system_xml)
        else:
            raise ConfigValueError(
                "an explicit trajectory needs masses for COM-based features: "
                "give 'system_xml' or set 'uniform_masses: true' (centroid "
                "geometry)", key="system_xml", value=None)
        if masses.shape[0] != frames.n_atoms:
            raise ConfigValueError(
                f"system_xml has {masses.shape[0]} particles, the trajectory "
                f"has {frames.n_atoms}", key="system_xml", value=system_xml)
        return frames, masses, [str(trajectory)], [(0, frames.positions.shape[0])]

    run_dirs = config["run_dirs"]
    if isinstance(run_dirs, (str, os.PathLike)):
        run_dirs = [run_dirs]
    positions_blocks: list[np.ndarray] = []
    steps_blocks: list[np.ndarray] = []
    masses: np.ndarray | None = None
    labels: list[str] = []
    ranges: list[tuple[int, int]] = []
    total = 0
    for run_dir in run_dirs:
        run_dir = os.fspath(run_dir)
        dcd = os.path.join(run_dir, DCD_FILENAME)
        if not os.path.exists(dcd):
            raise ConfigValueError(
                f"run directory {run_dir} has no {DCD_FILENAME} — its plan "
                f"set trajectory_interval to 0, so no per-frame positions "
                f"exist for geometry features (tape features still work if "
                f"you drop the position-dependent ones)",
                key="run_dirs", value=run_dir)
        frames = read_dcd_frames(dcd, stride=stride)
        if uniform:
            run_masses = np.ones(frames.n_atoms)
        else:
            source = system_xml or run_system_xml(run_dir)
            if source is None:
                raise ConfigValueError(
                    f"run directory {run_dir}: no system.xml for masses "
                    f"(neither the config's 'system_xml' key nor the run's "
                    f"manifest input_files.system names one); set "
                    f"'uniform_masses: true' for centroid geometry",
                    key="system_xml", value=str(run_dir), source=run_dir)
            run_masses = read_system_masses(source)
        if masses is None:
            masses = run_masses
        elif run_masses.shape != masses.shape or not np.array_equal(
                run_masses, masses):
            raise ConfigValueError(
                f"run directory {run_dir}: particle masses differ from the "
                f"first run's — featurized runs must share one system",
                key="run_dirs", value=str(run_dir))
        positions_blocks.append(frames.positions)
        steps_blocks.append(frames.steps)
        labels.append(run_dir)
        ranges.append((total, total + frames.positions.shape[0]))
        total += frames.positions.shape[0]
    if total == 0:
        raise ConfigValueError(
            "the run directories hold no trajectory frames (empty DCDs)",
            key="run_dirs", value=[str(d) for d in run_dirs])
    positions = (positions_blocks[0] if len(positions_blocks) == 1
                 else np.concatenate(positions_blocks, axis=0))
    steps = (steps_blocks[0] if len(steps_blocks) == 1
             else np.concatenate(steps_blocks, axis=0))
    return (Frames(positions=positions, steps=steps, n_atoms=masses.shape[0],
                   periodic=False),
            masses, labels, ranges)


def _tape_column(run_dir: str, tape: str, column: str,
                 steps: np.ndarray) -> np.ndarray:
    """One tape feature column aligned to the frame steps (nan where the
    tape has no row for a frame step)."""
    path = os.path.join(run_dir, tape)
    if not os.path.exists(path):
        raise ConfigValueError(
            f"tape feature needs {tape} inside {run_dir} — the file does "
            f"not exist", key="features", value=str(path))
    tape_data = read_step_tsv(path)
    if column not in tape_data.columns:
        matches = suggest(column, tape_data.columns)
        hint = (f"did you mean: {', '.join(matches)}?" if matches else
                f"columns: {tape_data.columns}")
        raise ConfigValueError(
            f"tape {path} has no column {column!r} ({hint})",
            key="column", value=column)
    lookup = {int(step): value for step, value in zip(
        tape_data.steps.tolist(), tape_data.values[:, tape_data.columns.index(
            column)].tolist())}
    return np.asarray([lookup.get(int(step), float("nan"))
                       for step in steps.tolist()], dtype=np.float64)


def featurize(config: Mapping, output: str | None = None) -> FeaturizeResult:
    """Compute every configured feature over the frames of the config's
    run dirs / trajectory and write the features.npz cache.

    Raises :class:`~neomd.errors.PlanValidationErrors` (collect-all) on
    config problems, single :class:`~neomd.errors.NeoUserError` family
    members on data problems found while running.
    """
    errors = validate_featurize_config(config)
    if errors:
        raise PlanValidationErrors(
            errors, footer="nothing was written — fix the problems above and "
            "re-run `neomd mlcv featurize`")

    features: Mapping = config["features"]
    frames, masses, labels, ranges = _frames_and_masses(config)
    n_frames = int(frames.positions.shape[0])
    _check_index_bounds(features, frames.n_atoms)

    built: dict[str, object] = {}
    for name, spec in features.items():
        if spec["type"] in _REGISTRY_TYPES:
            built[name] = _build_registry_feature(name, spec)

    columns: list[np.ndarray] = []
    names: list[str] = []
    types: list[str] = []
    units: list[str] = []
    for name, spec in features.items():
        feature_type = spec["type"]
        if feature_type == "tape":
            tape = spec.get("tape", "colvar.tsv")
            if "trajectory" in config:
                raise ConfigValueError(
                    f"feature {name!r}: tape features need run_dirs (a "
                    f"bare trajectory has no tapes)",
                    key=f"features.{name}", value=spec)
            column = np.full(n_frames, np.nan)
            filled = np.zeros(n_frames, dtype=bool)
            for run_label, (start, stop) in zip(labels, ranges):
                run_steps = frames.steps[start:stop]
                values = _tape_column(run_label, tape,
                                      spec["column"], run_steps)
                column[start:stop] = values
                filled[start:stop] = np.isfinite(values)
            if not filled.any():
                raise ConfigValueError(
                    f"tape feature {name!r}: no tape row matched any frame "
                    f"step (steps on tape and DCD frame cadences disagree?)",
                    key=f"features.{name}", value=spec)
            columns.append(column)
        else:
            values = np.empty(n_frames, dtype=np.float64)
            for i in range(n_frames):
                values[i] = _feature_value(
                    name, feature_type, spec, frames.positions[i], masses,
                    built.get(name))
            columns.append(values)
        names.append(str(name))
        types.append(str(feature_type))
        units.append(_UNITS[feature_type])

    matrix = (np.stack(columns, axis=1) if columns
              else np.zeros((n_frames, 0)))
    out = str(output or config.get("output") or "features.npz")
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    np.savez(
        out,
        format_version=np.int64(FEATURE_FORMAT_VERSION),
        values=matrix,
        steps=frames.steps.astype(np.int64),
        run_index=_run_index(frames, ranges),
        feature_names=np.array(json.dumps(names)),
        feature_types=np.array(json.dumps(types)),
        units=np.array(json.dumps(units)),
        masses_source=np.array(json.dumps(
            "uniform" if config.get("uniform_masses") else "system_xml")),
    )
    return FeaturizeResult(output=out, n_frames=n_frames,
                           feature_names=names, units=units, runs=labels)


def _run_index(frames: Frames, ranges) -> np.ndarray:
    """(n_frames,) run ordinal per frame row."""
    index = np.zeros(frames.positions.shape[0], dtype=np.int64)
    for ordinal, (start, stop) in enumerate(ranges):
        index[start:stop] = ordinal
    return index


def _check_index_bounds(features: Mapping, n_atoms: int) -> None:
    """Every index group of every feature inside the trajectory's particle
    count (the registry dry-run cannot know it — make_cv only parses)."""
    for name, spec in features.items():
        for key, value in spec.items():
            if not (key.endswith("_idx") or key == "restr_grp"):
                continue
            try:
                indices = _parse_indices(value)
            except (TypeError, ValueError):
                continue  # shape problems are the validator's job
            for index in indices:
                if not 0 <= index < n_atoms:
                    raise ConfigValueError(
                        f"feature {name!r} indexes particle {index}, but the "
                        f"trajectory has {n_atoms} atoms",
                        key=f"features.{name}.{key}", value=index)
