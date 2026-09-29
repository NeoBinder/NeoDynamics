"""Read-only protein variant inspection in a complete canonical complex.

This preflight binds explicit choices to source residue/atom IDs and installed
ff14SB/PDBFixer template bytes. It does not repair, protonate, parameterize or
convert a pH value into a chemical decision.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace

from .component_contract import canonical_sha256
from .errors import PlanValidationErrors
from .preparation import STANDARD_RESIDUES
from .topology_records import TopologyRecordError, validate_topology

_VARIANTS = {
    "ASP": ("ASP", "ASH"), "GLU": ("GLU", "GLH"),
    "LYS": ("LYS", "LYN"), "CYS": ("CYS", "CYM", "CYX"),
    "HIS": ("HID", "HIE", "HIP"),
}
_DIAGNOSTIC_CODES = {
    "invalid_topology": "invalid_reference",
    "connectivity_incomplete": "chemistry_unresolved",
    "no_protein": "capability_unavailable",
    "unsupported_residue": "capability_unavailable",
    "invalid_assignment": "mapping_mismatch",
    "template_unavailable": "capability_unavailable",
    "unsupported_attachment": "capability_unavailable",
    "unsupported_repair": "capability_unavailable",
    "variant_required": "chemistry_unresolved",
    "unsupported_variant": "capability_unavailable",
    "repair_required": "chemistry_unresolved",
    "protonation_conflict": "chemistry_unresolved",
    "external_bond_mismatch": "mapping_mismatch",
    "missing_bond": "mapping_mismatch",
}


@dataclass(frozen=True)
class ProteinDiagnostic:
    code: str
    path: str
    detail: str


@dataclass(frozen=True)
class ProteinResidueInspection:
    residue_id: str
    name: str
    variant: str | None
    source_atom_ids: tuple[str, ...]
    missing_heavy_atoms: tuple[str, ...]
    missing_hydrogens: tuple[str, ...]
    external_bonds: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class ProteinContextInspection:
    source_revision_id: str | None
    source_topology_sha256: str | None
    protein_resource_sha256: str | None
    hydrogen_definitions_sha256: str | None
    residues: tuple[ProteinResidueInspection, ...]
    diagnostics: tuple[ProteinDiagnostic, ...]

    @property
    def ready(self) -> bool:
        """Source preflight passed; listed missing H still need explicit completion."""
        return not self.diagnostics


def _diagnostic(code, path, detail):
    return ProteinDiagnostic(_DIAGNOSTIC_CODES[code], path, f"{code}: {detail}")


def _variant_names(base):
    cores = _VARIANTS.get(base, (base,))
    return {name for core in cores for name in (core, "N" + core, "C" + core)}


def _trusted_templates(protein_residues):
    """Use the same bounded installed inventories as repair and H completion."""
    from openmm import app

    from .protein_hydrogens import _trusted_inventory
    from .repair_rules import template_atoms

    # template_atoms reads only residue names. Filter the full complex's
    # nonprotein residues without editing its source or inventing bonds.
    names = SimpleNamespace(residues=lambda: (
        SimpleNamespace(name=row["name"]) for row in protein_residues
    ))
    heavy = template_atoms(names, app)
    resources, variants, _, hydrogen_definitions = _trusted_inventory()
    protein_resource = next(metadata for metadata, _ in resources
                            if metadata["role"] == "protein")
    return heavy, variants, protein_resource["sha256"], hydrogen_definitions["sha256"]


def inspect_protein_variants(records: dict, assignments) -> ProteinContextInspection:
    """Inspect every polymer residue against explicit ff14SB variant choices.

    ``assignments`` contains exactly ``{residue_id, variant}`` for each standard
    protein residue; variant is the installed template name (for example NALA,
    ASH, HID, CYX). No assignment is inferred from pH or residue label. All
    original bonds, including links into a ligand, remain in this full-complex
    view. A ready result is structural preflight, not a prepared System receipt.
    """
    try:
        validate_topology(records)
        source_hash = canonical_sha256(records)
    except (TopologyRecordError, TypeError, ValueError) as error:
        return ProteinContextInspection(None, None, None, None, (), (
            _diagnostic("invalid_topology", "", f"canonical topology rejected: {type(error).__name__}"),
        ))
    revision = records["revision_id"]
    if records["coverage"]["state"] != "complete":
        return ProteinContextInspection(revision, source_hash, None, None, (), (
            _diagnostic("connectivity_incomplete", "/coverage/state",
                        "protein variant inspection requires complete stored bonds"),
        ))

    diagnostics = []
    protein = [row for row in records["residues"]
               if row["entity_kind"] == "polymer" and row["name"] in STANDARD_RESIDUES]
    protein_ids = {row["id"] for row in protein}
    if not protein:
        diagnostics.append(_diagnostic("no_protein", "/residues",
                                       "full complex has no qualified standard protein residue"))
    for index, row in enumerate(records["residues"]):
        if row["entity_kind"] == "polymer" and row["name"] not in STANDARD_RESIDUES:
            diagnostics.append(_diagnostic("unsupported_residue", f"/residues/{index}/name",
                                           "polymer residue needs a separately qualified template"))

    choices = {}
    choice_paths = {}
    if not isinstance(assignments, (list, tuple)):
        assignments = ()
        diagnostics.append(_diagnostic("invalid_assignment", "/assignments",
                                       "requires explicit residue and variant rows"))
    for index, row in enumerate(assignments):
        path = f"/assignments/{index}"
        if (not isinstance(row, dict) or set(row) != {"residue_id", "variant"}
                or not isinstance(row.get("residue_id"), str)
                or not isinstance(row.get("variant"), str)):
            diagnostics.append(_diagnostic("invalid_assignment", path,
                                           "requires residue_id and exact template variant"))
            continue
        residue_id = row["residue_id"]
        if residue_id not in protein_ids or residue_id in choices:
            diagnostics.append(_diagnostic("invalid_assignment", f"{path}/residue_id",
                                           "unknown or duplicate protein residue"))
            continue
        choices[residue_id] = row["variant"]
        choice_paths[residue_id] = path

    try:
        heavy_templates, variants, protein_hash, hydrogen_hash = _trusted_templates(protein)
    except (ImportError, LookupError, OSError, ValueError, PlanValidationErrors) as error:
        diagnostics.append(_diagnostic("template_unavailable", "",
                                       f"trusted protein template inventory unavailable: {type(error).__name__}"))
        return ProteinContextInspection(revision, source_hash, None, None, (), tuple(diagnostics))

    atom_by_id = {row["id"]: row for row in records["atoms"]}
    residue_by_id = {row["id"]: row for row in records["residues"]}
    atoms_by_residue = defaultdict(list)
    for atom in records["atoms"]:
        atoms_by_residue[atom["residue_id"]].append(atom)
    external = defaultdict(list)
    internal = defaultdict(set)
    for index, bond in enumerate(records["bonds"]):
        left, right = (atom_by_id[atom_id] for atom_id in bond["atom_ids"])
        left_residue, right_residue = left["residue_id"], right["residue_id"]
        if left_residue == right_residue:
            internal[left_residue].add(frozenset((left["name"], right["name"])))
            continue
        for own, other in ((left, right), (right, left)):
            if own["residue_id"] in protein_ids:
                external[own["residue_id"]].append((own["id"], other["id"], other["residue_id"],
                                                     own["name"], other["name"], index))
        if left_residue not in protein_ids and right_residue not in protein_ids:
            continue
        left_name = residue_by_id[left_residue]["name"] if left_residue else None
        right_name = residue_by_id[right_residue]["name"] if right_residue else None
        peptide = (left_residue in protein_ids and right_residue in protein_ids
                   and ((left["name"], right["name"]) == ("C", "N")
                        or (left["name"], right["name"]) == ("N", "C")))
        disulfide = (left_name == right_name == "CYS"
                     and left_residue in protein_ids and right_residue in protein_ids
                     and left["name"] == right["name"] == "SG")
        allowed_kind = (peptide and bond["kind"]["kind"] in ("covalent", "unknown")
                        or disulfide and bond["kind"]["kind"] in
                        ("covalent", "disulfide", "unknown"))
        if (not allowed_kind or bond["order"]["kind"] not in ("single", "unknown")):
            diagnostics.append(_diagnostic("unsupported_attachment", f"/bonds/{index}",
                                           "external protein bond needs a qualified joint template"))

    inspected = []
    for index, residue in enumerate(records["residues"]):
        if residue["id"] not in protein_ids:
            continue
        rid, base = residue["id"], residue["name"]
        path = f"/residues/{index}"
        atoms = atoms_by_residue[rid]
        names = {atom["name"]: atom for atom in atoms}
        if len(names) != len(atoms):
            diagnostics.append(_diagnostic("unsupported_repair", path,
                                           "duplicate atom names prevent template mapping"))
        variant = choices.get(rid)
        if variant is None:
            diagnostics.append(_diagnostic("variant_required", path,
                                           "explicit protein variant assignment is required"))
        elif variant not in _variant_names(base) or variant not in variants:
            diagnostics.append(_diagnostic("unsupported_variant", f"{choice_paths[rid]}/variant",
                                           "variant is not a qualified ff14SB template for this residue"))
        expected = variants.get(variant) if variant in _variant_names(base) else None
        missing_heavy = ()
        missing_hydrogens = ()
        if expected:
            expected_atoms = dict(expected["atoms"])
            expected_bonds = expected["bonds"]
            # Pinned OpenMM PDB loading can normalize N-terminal H1 to H.
            # Match the existing hydrogen-completion rule without renaming source.
            if variant.startswith("N") and "H1" not in names and "H1" in expected_atoms:
                expected_atoms["H"] = expected_atoms.pop("H1")
                expected_bonds = {
                    frozenset("H" if name == "H1" else name for name in pair)
                    for pair in expected_bonds
                }
            allowed_heavy = {**heavy_templates[base],
                             **({"OXT": "O"} if "OXT" in expected_atoms else {})}
            actual_heavy = {name: atom["element"] for name, atom in names.items()
                            if atom["element"] != "H"}
            for name, element in actual_heavy.items():
                if name not in allowed_heavy or allowed_heavy[name] != element:
                    diagnostics.append(_diagnostic("unsupported_repair", path,
                                                   "retained heavy atom differs from the qualified template"))
                    break
            missing_heavy = tuple(sorted(set(allowed_heavy) - set(actual_heavy)))
            if missing_heavy:
                diagnostics.append(_diagnostic("repair_required", path,
                                               "missing standard heavy atoms require qualified repair before assignment"))
            for name, atom in names.items():
                if name in expected_atoms and expected_atoms[name] == atom["element"]:
                    continue
                if atom["element"] == "H":
                    diagnostics.append(_diagnostic("protonation_conflict", path,
                                                   "retained hydrogen is incompatible with the explicit variant"))
                    break
            missing_hydrogens = tuple(sorted(name for name, element in expected_atoms.items()
                                             if element == "H" and name not in names))
            observed_external = Counter(row[3] for row in external[rid])
            if observed_external != expected["external"]:
                diagnostics.append(_diagnostic("external_bond_mismatch", path,
                                               "stored external bonds differ from the explicit template"))
            present = set(names)
            expected_pairs = {bond for bond in expected_bonds if bond <= present}
            if internal[rid] != expected_pairs:
                diagnostics.append(_diagnostic("missing_bond", path,
                                               "stored internal bonds differ from the explicit template"))
        inspected.append(ProteinResidueInspection(
            rid, base, variant, tuple(atom["id"] for atom in atoms), missing_heavy,
            missing_hydrogens,
            tuple((own, other, neighbor) for own, other, neighbor, _, _, _ in
                  sorted(external[rid], key=lambda row: (int(row[0]), int(row[1])))),
        ))
    return ProteinContextInspection(revision, source_hash, protein_hash, hydrogen_hash,
                                    tuple(inspected), tuple(diagnostics))
