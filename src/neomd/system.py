"""system — the kernel-agnostic system description (v2 migration plan §5
Phase 2 item 2.3; split per the v2 improvements list item 6).

This module is now ONE-headed: :class:`SystemBundle` — pure data + loading +
validation — plus its openmm-free helpers.  It NEVER imports openmm: the
openmm kernel deserializes ``system.xml`` itself (see ``kernel/openmm.py``),
so the bundle only carries file paths, ligand molecules (openff, lazily)
and the *modification IR* (raw barostat dict + normalized particle-mass
overrides).  ``run.compile`` is the consumer that turns the same IR into a
:class:`~neomd.kernel.port.KernelSpec`; the bundle exists so callers can
inspect/describe a system without a kernel.

The v1-ported PREPARATION WORKFLOW moved to :mod:`neomd.prepare`
(prepare_system / make_system / loaders / the ForceFieldBuilder seam), and
every openmm PRIVATE attribute it needs lives in
:mod:`neomd.openmm_privates` behind a pinned-version gate.  The workflow
names below remain importable from ``neomd.system`` (the historical import
path tests and callers used) — they are re-exports, not copies.

The fgroup write-back is DEAD (plan §2)
---------------------------------------
v1's ``NeoSystem.system_add_restraints`` mutated the freshly deserialized
openmm System (adding restraint forces, assigning force groups) and then
wrote the assigned groups back into the user's config
(``config.restraint[name]["fgroup"] = fgroup`` — neosystem.py:122).  v2 kills
both halves:

* restraints never touch a SystemBundle — nothing in this module adds a
  Force, assigns a force group, or knows the word "fgroup";
* plan restraint entries flow through the registry knowledge triples to
  ``kernel.install_bias`` (``driver.drive``), and the assigned force-group
  ids come back as ``RunOutcome.fgroups`` (name -> list[int]) — a return
  value of the interface, never a config mutation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Mapping

from .errors import ConfigValueError
from .plan import Plan

# the preparation workflow lives in prepare.py; re-exported here because
# callers historically imported it from neomd.system (the split kept the
# public import paths stable)
from .prepare import (  # noqa: F401  (re-exports)
    ForceFieldBuilder,
    PlainForceFieldBuilder,
    custom_addH,
    custom_bonds,
    load_complex,
    prepare_system,
    sys_params_from_config,
    system_from_amber,
    system_from_gromacs,
)

__all__ = [
    "SystemBundle",
    "ForceFieldBuilder",
    "PlainForceFieldBuilder",
    "prepare_system",
    "load_complex",
    "system_from_gromacs",
    "system_from_amber",
    "custom_bonds",
    "custom_addH",
    "sys_params_from_config",
]


# ---------------------------------------------------------------------------
# SystemBundle — the kernel-agnostic description (openmm-free)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SystemBundle:
    """Everything about a prepared system except the physics kernel.

    Fields
    ------
    topology_file:
        path of the .pdb/.pdbx carrying topology + positions.
    system_xml:
        path of the serialized openmm System (deserialized by the kernel).
    ligands:
        list of openff Molecule objects, or None when no ligands are bound.
    templates:
        ligand template identifiers (ffxml paths from a plan, or the template
        names the builder used during preparation).
    modifications:
        the modification IR — ``{"barostat": raw dict | None,
        "particle_masses": {int index: float dalton} | None}``.  The barostat
        dict is kept RAW (v1 ``NeoSystem.add_barostat`` defaults and the
        seed/temperature augmentation happen in ``run.compile`` -> KernelSpec,
        never here).
    """

    topology_file: str | None = None
    system_xml: str | None = None
    ligands: list | None = None
    templates: list[str] | None = None
    modifications: dict | None = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_plan(cls, plan) -> "SystemBundle":
        """Validate the plan's input_files and load the ligand molecules.

        * ``complex`` must exist and end in ``.pdb``/``.pdbx`` (the v1
          ``load_complex`` suffix check and message);
        * ``system`` must exist;
        * ``ligands`` (optional) is a json list of openff molecule dicts —
          openff.toolkit is imported lazily inside the loader because
          ligands are optional;
        * ``templates`` comes comma-split from ``plan.derived``;
        * ``barostat``/``system_modification`` are extracted into the
          modification IR (see module docstring).

        ``plan`` may be a :class:`~neomd.plan.Plan` or a plain plan dict
        (validated through ``Plan.from_dict``).
        """
        if isinstance(plan, Plan):
            resolved = plan
        elif isinstance(plan, Mapping):
            resolved = Plan.from_dict(dict(plan))
        else:
            raise TypeError(
                f"SystemBundle.from_plan expects a Plan or a plan mapping, "
                f"got {type(plan).__name__}")

        input_files = resolved.input_files

        complex_path = input_files.get("complex")
        if complex_path is None:
            raise ConfigValueError(
                "input_files.complex is required to describe a system",
                key="complex")
        _check_complex_suffix(complex_path)
        _check_exists(complex_path, "input_files.complex")

        system_path = input_files.get("system")
        if system_path is None:
            raise ConfigValueError(
                "input_files.system is required to describe a system",
                key="system")
        _check_exists(system_path, "input_files.system")

        ligand_path = input_files.get("ligands")
        ligands = _load_ligands(ligand_path) if ligand_path else None

        return cls(
            topology_file=str(complex_path),
            system_xml=str(system_path),
            ligands=ligands,
            templates=getattr(resolved, "templates", None),
            modifications=_extract_modifications(resolved.raw),
        )

    # -- reporting -----------------------------------------------------------

    def describe(self) -> str:
        """A compact multi-line summary for logs and manifests."""
        ligands = self.ligands
        if ligands:
            names = ", ".join(
                getattr(mol, "name", "") or f"ligand{i + 1}"
                for i, mol in enumerate(ligands))
            ligand_line = f"ligands: {len(ligands)} ({names})"
        else:
            ligand_line = "ligands: none"
        modifications = self.modifications or {}
        barostat = modifications.get("barostat")
        masses = modifications.get("particle_masses")
        parts = []
        if barostat:
            items = ", ".join(
                f"{key}={value!r}" for key, value in sorted(barostat.items()))
            parts.append(f"barostat({items})")
        if masses:
            parts.append(f"{len(masses)} particle-mass overrides")
        mod_line = "modifications: " + ("; ".join(parts) if parts else "none")
        templates = (
            f"templates: {', '.join(self.templates)}" if self.templates
            else "templates: none")
        return "\n".join((
            "SystemBundle:",
            f"  topology: {self.topology_file}",
            f"  system:   {self.system_xml}",
            f"  {ligand_line}",
            f"  {templates}",
            f"  {mod_line}",
        ))


# ---------------------------------------------------------------------------
# loading / validation helpers (openmm-free)
# ---------------------------------------------------------------------------


def _check_complex_suffix(complex_path: str) -> None:
    """The v1 ``load_complex`` suffix gate, message verbatim."""
    if not (complex_path.endswith(".pdb") or complex_path.endswith(".pdbx")):
        raise ConfigValueError(
            "In config.input_files.complex, unrecognized file type:{}".format(
                complex_path),
            key="complex",
            value=complex_path)


def _check_exists(path: str, key: str) -> None:
    if not os.path.isfile(path):
        raise ConfigValueError(
            f"{key} file not found: {path}", key=key, value=path)


def _load_ligands(ligand_path: str) -> list:
    """Load the ligand json (a list of openff molecule dicts).

    openff.toolkit is imported lazily HERE: ligands are optional, and the
    rest of this module works without the toolkit.
    """
    _check_exists(ligand_path, "input_files.ligands")
    try:
        with open(ligand_path, "r") as handle:
            ligands_json = json.load(handle)
    except json.JSONDecodeError as error:
        raise ConfigValueError(
            f"input_files.ligands is not valid json: {error}",
            key="ligands", value=ligand_path) from error
    if not isinstance(ligands_json, list):
        raise ConfigValueError(
            "input_files.ligands must be a json LIST of openff molecule "
            f"dicts, got {type(ligands_json).__name__}",
            key="ligands", value=ligand_path)
    try:
        from openff.toolkit.topology import Molecule as openff_Molecule
    except ImportError as error:
        raise ConfigValueError(
            "input_files.ligands requires openff-toolkit, which is not "
            "importable in this environment (ligands are optional; remove "
            "the key to prepare a protein-only system)",
            key="ligands", value=ligand_path) from error
    return [
        openff_Molecule.from_json(json.dumps(liginfo))
        for liginfo in ligands_json
    ]


def _normalize_particle_masses(system_modification) -> dict[int, float] | None:
    """``{particle index: mass in dalton}`` from raw ``system_modification``.

    Verbatim port of v1 ``neosystem.py:74-78`` semantics (a mapping of
    ``{index: {"mass": value}}``; entries without a "mass" key are ignored),
    accepting BOTH spellings the plan schema allows — the v1 mapping and the
    list-of-dicts ``[{"index": i, "mass": m}, ...]`` (normalized exactly like
    ``run._particle_masses`` compiles into the KernelSpec).
    """
    if not system_modification:
        return None
    if isinstance(system_modification, Mapping):
        entries = system_modification.items()
    else:
        entries = (
            (entry.get("index"), entry)
            for entry in system_modification
            if isinstance(entry, Mapping)
        )
    masses: dict[int, float] = {}
    for index, info in entries:
        if isinstance(info, Mapping) and "mass" in info:
            masses[int(index)] = float(info["mass"])
    return masses or None


def _extract_modifications(source: Mapping) -> dict:
    """The modification IR from a plan's RAW view (or a prepare config).

    The barostat dict stays RAW — the temperature/seed augmentation v1 did
    in ``add_barostat`` (``config.get("temperature", 298)`` /
    ``setRandomNumberSeed(config.seed)``) is ``run.compile``'s job, so the
    same source of truth feeds both the bundle and the KernelSpec.
    """
    if not isinstance(source, Mapping):  # pragma: no cover - defensive
        raise TypeError(
            f"modification source must be a mapping, "
            f"got {type(source).__name__}")
    barostat = source.get("barostat")
    if barostat is not None:
        barostat = dict(barostat)
    return {
        "barostat": barostat,
        "particle_masses": _normalize_particle_masses(
            source.get("system_modification")),
    }


def _templates_value(value) -> list[str] | None:
    """Comma-split a templates string (plans already did this in derive)."""
    if not value:
        return None
    if isinstance(value, str):
        return value.split(",")
    return [str(item) for item in value]
