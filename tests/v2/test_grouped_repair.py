"""Real grouped heavy repair: explicit original identities, no full preparation."""

import copy

import numpy as np
import pytest
from openmm import app, unit

from neomd.errors import PlanValidationErrors
from neomd_preparation.source import load_protein_source
from tests.v2.test_protein_grouping import BOND_METADATA, canonicalize, mixed_source
from tests.v2.test_protein_grouping_cases import _complete_standard_peptide_source
from tests.v2.test_protein_preparation import descriptor, source_fixture


def complete_source(tmp_path, *, missing_cb=True, reverse=False, coincident=False):
    """Canonical water-first uncapped peptide, with optional missing heavy CB.

    The preexisting terminal OXT is source evidence, never a guessed addition.
    Retained HB atoms are intentionally kept when their missing CB is removed.
    """
    source = _complete_standard_peptide_source(tmp_path)
    # TEST-ONLY coordinate snapshot: public PDBFixer 1.12.0, CPU, seed=123
    # repaired a heavy-only two-chain peptide/water fixture; public ff14SB/TIP3P
    # Modeller.addHydrogens(pH=7, CPU) then placed H around that heavy geometry.
    # Store the measured source coordinates, not a production repair result.
    # The old hand-drawn HB geometry stretched a regenerated CB–HB to 0.21 nm.
    qualified = {
        (1, "H"): [0.46523, 0.56306, 0.42880],
        (1, "H2"): [0.46553, 0.40692, 0.48159],
        (1, "H3"): [0.46582, 0.53116, 0.59005],
        (1, "HA"): [0.65720, 0.48592, 0.39258],
        (1, "CB"): [0.73360, 0.37827, 0.53627],
        (1, "HB1"): [0.74456, 0.37161, 0.64453],
        (1, "HB2"): [0.83245, 0.38960, 0.49147],
        (1, "HB3"): [0.68793, 0.28654, 0.49908],
        (2, "H"): [0.50003, 0.71576, 0.50499],
        (2, "HA"): [0.69020, 0.89812, 0.40744],
        (2, "HB1"): [0.83421, 0.87522, 0.60036],
        (2, "HB2"): [0.72634, 1.00741, 0.63922],
        (2, "HB3"): [0.69425, 0.84891, 0.70945],
    }
    top, positions, mapping = app.Topology(), [], {}
    for old_chain in source.topology.chains():
        chain = top.addChain(old_chain.id)
        for old_residue in old_chain.residues():
            residue = top.addResidue(old_residue.name, chain, old_residue.id, old_residue.insertionCode)
            atoms = list(old_residue.atoms())
            for old in reversed(atoms) if reverse else atoms:
                if missing_cb and old_residue.index == 1 and old.name == "CB":
                    continue
                mapping[old] = top.addAtom(old.name, old.element, residue, old.id,
                                         formalCharge=old.formalCharge)
                positions.append(np.asarray(qualified.get((old_residue.index, old.name),
                                                          source.positions_nm[old.index])).copy())
    for bond in source.topology.bonds():
        if all(atom in mapping for atom in bond):
            top.addBond(mapping[bond[0]], mapping[bond[1]], order=bond.order, type=bond.type)
    top.setPeriodicBoxVectors(source.topology.getPeriodicBoxVectors())
    positions = np.asarray(positions)
    if coincident:
        retained_h = [a.index for a in top.atoms() if a.element == app.element.hydrogen]
        positions[retained_h[1]] = positions[retained_h[0]]
    return canonicalize(tmp_path, top, positions, name="missing-standard.pdbx")


def test_real_grouped_heavy_repair_preserves_source_water_hydrogens_and_stage_uuids(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = complete_source(tmp_path)
    before_bytes = source.source_path.read_bytes()
    before_descriptor = copy.deepcopy(source.descriptor)
    before_positions = source.positions_nm.tobytes()
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    assert [s["operation"] for s in repaired.lineage.steps] == ["load", "group_protein_water", "repair_standard_residue"]
    assert [[r.name for r in c.residues()] for c in repaired.topology.chains()] == [["ALA", "ALA"], ["HOH"]]
    generated = [a for a in repaired.lineage.atoms if a["origin"] == "generated_heavy_atom"]
    assert [(a["name"], a["element"]) for a in generated] == [("CB", "C")]
    group_step = repaired.lineage.steps[1]["grouping"]
    repair_step = repaired.lineage.steps[2]["repair"]
    by_stage = {a["output_atom_index"]: a for a in group_step["atoms"]}
    for row in repair_step["retained_atoms"]:
        grouped = by_stage[row["input_atom_index"]]
        assert row["atom_id"] == grouped["atom_id"]
        assert row["source_atom_id"] == grouped["source_atom_id"]
        assert repaired.positions_nm[row["output_atom_index"]].tobytes() == source.positions_nm[grouped["input_atom_index"]].tobytes()
    assert repaired.lineage.sources[0]["bonds"] == source.descriptor["bonds"]
    assert repaired.lineage.sources[0]["polymer_sequences"] == source.descriptor["polymer_sequences"]
    assert all(a["origin"] == "source" for a in repaired.lineage.atoms if a["element"] == "H")
    assert source.source_path.read_bytes() == before_bytes
    assert source.descriptor == before_descriptor
    assert source.positions_nm.tobytes() == before_positions
    assert repaired.positions_nm.flags.writeable is False
    assert np.asarray(repaired.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)).tobytes() == np.asarray(source.topology.getPeriodicBoxVectors().value_in_unit(unit.nanometer)).tobytes()


def known_terminal_source(tmp_path, *, evidence="client_declared", gap=False):
    """Two observed alanines: terminal evidence plus recoverable actual geometry."""
    original = complete_source(tmp_path)
    modeller = app.Modeller(original.topology, original.positions_nm * unit.nanometer)
    modeller.delete([a for a in original.topology.atoms() if a.residue.name == "HOH" or a.name == "OXT"])
    source = canonicalize(tmp_path, modeller.topology,
                          np.asarray(modeller.positions.value_in_unit(unit.nanometer)), name="terminal-evidence.pdbx")
    desc = copy.deepcopy(source.descriptor)
    sequence = desc["polymer_sequences"][0]
    sequence.update(evidence=evidence, residue_names=["ALA"] * (3 if gap else 2),
                    residue_bindings=[{"sequence_index": 2 if gap == "internal" and i == 1 else i,
                                       "source_residue_id": row["source_residue_id"]}
                                      for i, row in enumerate(desc["residue_bindings"])])
    canonical = ({0: {"residue_names": sequence["residue_names"],
                      "residue_bindings": sequence["residue_bindings"]}}
                 if evidence == "canonical_file" else None)
    return load_protein_source(source.source_path, desc, canonical_sequences=canonical)


@pytest.mark.parametrize("evidence", ["client_declared", "canonical_file"])
def test_complete_explicit_terminal_evidence_allows_missing_oxt(tmp_path, evidence):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = known_terminal_source(tmp_path, evidence=evidence)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    generated = [a for a in repaired.lineage.atoms if a["origin"] == "generated_heavy_atom"]
    assert {a["name"] for a in generated} == {"CB", "OXT"}
    assert repaired.lineage.sources[0]["polymer_sequences"] == source.descriptor["polymer_sequences"]
    assert repaired.report["generated_count"] == 2


def test_unknown_terminal_evidence_blocks_automatic_oxt_without_source_changes(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    path, desc = source_fixture(tmp_path)
    source = load_protein_source(path, desc)
    before = (path.read_bytes(), copy.deepcopy(desc), source.positions_nm.tobytes())
    with pytest.raises(PlanValidationErrors, match="terminal|OXT|evidence"):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert (path.read_bytes(), source.descriptor, source.positions_nm.tobytes()) == before


def test_known_missing_sequence_stretch_blocks_even_with_explicit_terminal_claim(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = known_terminal_source(tmp_path, gap=True)
    before = source.source_path.read_bytes()
    with pytest.raises(PlanValidationErrors, match="gap|sequence|stretch"):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert source.source_path.read_bytes() == before


@pytest.mark.parametrize("evidence", ["client_declared", "canonical_file"])
@pytest.mark.parametrize("gap,missing", [("internal", 1), ("terminal", 2)])
@pytest.mark.parametrize("repair", ["grouped", "source_only"])
def test_unpositioned_known_residue_fails_before_fixer(tmp_path, monkeypatch, evidence, gap, missing, repair):
    from pdbfixer import PDBFixer

    from neomd_preparation.protein.repair import group_and_repair_protein
    from neomd_preparation.source import inspect_protein, repair_standard_residues

    source = known_terminal_source(tmp_path, evidence=evidence, gap=gap)
    before = (source.source_path.read_bytes(), copy.deepcopy(source.descriptor),
              source.positions_nm.tobytes())
    assert inspect_protein(source)["findings"] == [
        {"code": "detected_gap", "input_chain_index": 0, "sequence_indices": [missing]},
    ]

    def must_not_run(*args, **kwargs):
        raise AssertionError("missing-residue preparation reached PDBFixer")

    monkeypatch.setattr(PDBFixer, "findMissingAtoms", must_not_run)
    producer = group_and_repair_protein if repair == "grouped" else repair_standard_residues
    with pytest.raises(PlanValidationErrors) as caught:
        producer(source, platform="CPU", seed=123)
    sequence_errors = [error for error in caught.value.errors if error.key == "repair.sequence"]
    assert len(sequence_errors) == 1
    assert f"[{missing}] have no residue coordinates" in sequence_errors[0].message
    assert "model missing residues upstream" in sequence_errors[0].message
    assert (source.source_path.read_bytes(), source.descriptor, source.positions_nm.tobytes()) == before


@pytest.mark.parametrize("replacement", ["client_declared", "unknown"])
def test_trusted_missing_residue_sequence_cannot_be_downgraded_to_observed_fragment(tmp_path, replacement):
    original = known_terminal_source(tmp_path, evidence="canonical_file", gap="internal")
    canonical = original.canonical_sequences
    desc = copy.deepcopy(original.descriptor)
    sequence = desc["polymer_sequences"][0]
    sequence["evidence"] = replacement
    sequence["residue_names"] = ["ALA", "ALA"] if replacement == "client_declared" else None
    sequence["residue_bindings"] = (
        [{"sequence_index": index, "source_residue_id": row["source_residue_id"]}
         for index, row in enumerate(desc["residue_bindings"])]
        if replacement == "client_declared" else []
    )
    with pytest.raises(PlanValidationErrors, match="trusted canonical sequence"):
        load_protein_source(original.source_path, desc, canonical_sequences=canonical)


@pytest.mark.parametrize("fault", ["missing_N", "missing_C", "unknown_heavy", "heavy_name_hydrogen"])
def test_unsupported_backbone_or_heavy_chemistry_blocks_without_deletion(tmp_path, fault):
    from neomd_preparation.protein.repair import group_and_repair_protein

    original = complete_source(tmp_path, missing_cb=False)
    topology = copy.deepcopy(original.topology)
    positions = original.positions_nm.copy()
    first = list(topology.residues())[1]
    if fault.startswith("missing_"):
        name = fault.removeprefix("missing_")
        modeller = app.Modeller(topology, positions * unit.nanometer)
        modeller.delete([a for a in first.atoms() if a.name == name])
        topology = modeller.topology
        positions = np.asarray(modeller.positions.value_in_unit(unit.nanometer))
    else:
        cb = next(a for a in first.atoms() if a.name == "CB")
        if fault == "unknown_heavy":
            cb.name = "GHOST"
        else:
            cb.element = app.element.hydrogen
    source = canonicalize(tmp_path, topology, positions, name="unsupported.pdbx")
    before = (source.source_path.read_bytes(), source.positions_nm.tobytes(), copy.deepcopy(source.descriptor))
    with pytest.raises(PlanValidationErrors):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert (source.source_path.read_bytes(), source.positions_nm.tobytes(), source.descriptor) == before


@pytest.mark.parametrize("seed", [True, 0, -1, 2147483648, 1.0, "1"])
def test_grouped_repair_requires_explicit_valid_seed(tmp_path, seed):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = complete_source(tmp_path)
    with pytest.raises(PlanValidationErrors, match="seed"):
        group_and_repair_protein(source, platform="CPU", seed=seed)


def separated_runs(tmp_path, *, existing_last_oxt):
    original = mixed_source(tmp_path)
    top, xyz, mapping = app.Topology(), [], {}
    for old_chain in original.topology.chains():
        chain = top.addChain(old_chain.id)
        for old_residue in old_chain.residues():
            residue = top.addResidue(old_residue.name, chain, old_residue.id, old_residue.insertionCode)
            for old in old_residue.atoms():
                mapping[old] = top.addAtom(old.name, old.element, residue, old.id,
                                         formalCharge=old.formalCharge)
                xyz.append(original.positions_nm[old.index].copy())
            if existing_last_oxt and old_residue.index == 2:
                oxt = top.addAtom("OXT", app.element.oxygen, residue)
                carbon = next(a for a in old_residue.atoms() if a.name == "C")
                top.addBond(mapping[carbon], oxt)
                xyz.append(original.positions_nm[carbon.index] + [0, 0.12, 0])
    for bond in original.topology.bonds():
        if all(a in mapping for a in bond):
            top.addBond(mapping[bond[0]], mapping[bond[1]], order=bond.order, type=bond.type)
    top.setPeriodicBoxVectors(original.topology.getPeriodicBoxVectors())
    return canonicalize(tmp_path, top, np.asarray(xyz), name="separated-runs.pdbx")


def test_internal_protein_run_does_not_gain_oxt_from_group_chain_end(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = separated_runs(tmp_path, existing_last_oxt=True)
    repaired = group_and_repair_protein(source, platform="CPU", seed=123)
    residues = list(repaired.topology.residues())
    assert [r.name for r in residues] == ["ALA", "ALA", "HOH", "HOH"]
    assert "OXT" not in {a.name for a in residues[0].atoms()}
    assert "OXT" in {a.name for a in residues[1].atoms()}
    # Complete source heavy atoms isolate GROUP terminal suppression from
    # unrelated synthetic sidechain geometry; no repair is manufactured here.
    assert repaired.lineage.steps[-1]["generated_count"] == 0


def test_water_inclusive_sequence_cannot_impersonate_biological_terminal_evidence(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    source = separated_runs(tmp_path, existing_last_oxt=False)
    desc = copy.deepcopy(source.descriptor)
    rows = desc["residue_bindings"]
    desc["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=[r["name"] for r in rows],
        residue_bindings=[{"sequence_index": i, "source_residue_id": r["source_residue_id"]}
                         for i, r in enumerate(rows)])
    declared = load_protein_source(source.source_path, desc)
    with pytest.raises(PlanValidationErrors, match="terminal|sequence|evidence"):
        group_and_repair_protein(declared, platform="CPU", seed=123)


def test_known_sequence_cannot_reverse_original_residue_order_even_with_existing_oxt(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    original = complete_source(tmp_path)
    modeller = app.Modeller(original.topology, original.positions_nm * unit.nanometer)
    modeller.delete([r for r in original.topology.residues() if r.name == "HOH"])
    source = canonicalize(tmp_path, modeller.topology,
                          np.asarray(modeller.positions.value_in_unit(unit.nanometer)), name="protein-only.pdbx")
    desc = copy.deepcopy(source.descriptor)
    rows = desc["residue_bindings"]
    assert len(rows) == 2 and [r["name"] for r in rows] == ["ALA", "ALA"]
    desc["polymer_sequences"][0].update(
        evidence="client_declared", residue_names=["ALA", "ALA"],
        residue_bindings=[{"sequence_index": 1 - i, "source_residue_id": r["source_residue_id"]}
                         for i, r in enumerate(rows)])
    declared = load_protein_source(source.source_path, desc)
    with pytest.raises(PlanValidationErrors, match="sequence"):
        group_and_repair_protein(declared, platform="CPU", seed=123)


def test_actual_retained_bond_order_loss_is_not_hidden_by_restoring_source_graph(tmp_path):
    from neomd_preparation.protein.repair import group_and_repair_protein

    original = complete_source(tmp_path)
    path = original.source_path
    path.write_text(path.read_text() + BOND_METADATA)
    loaded = app.PDBxFile(str(path))
    assert any(b.order == 1 for b in loaded.topology.bonds())
    source = load_protein_source(path, descriptor(path, loaded))
    before = path.read_bytes()
    with pytest.raises(PlanValidationErrors, match="bonds"):
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert path.read_bytes() == before
    assert any(b.order == 1 for b in source.topology.bonds())


def cysteine_pair(tmp_path, *, missing_sg=False):
    import importlib.resources

    template = app.PDBFile(str(importlib.resources.files("pdbfixer") / "templates/CYS.pdb"))
    top, xyz = app.Topology(), []
    coords = np.asarray(template.positions.value_in_unit(unit.nanometer))
    for index in range(2):
        chain = top.addChain(f"original{index}")
        residue = top.addResidue("CYS", chain, str(10 + index))
        for old in template.topology.atoms():
            if missing_sg and index == 0 and old.name == "SG":
                continue
            top.addAtom(old.name, old.element, residue)
            xyz.append(coords[old.index] + [index * 0.2, 0, 0])
        carbon = next(a for a in template.topology.atoms() if a.name == "C")
        top.addAtom("OXT", app.element.oxygen, residue)
        xyz.append(coords[carbon.index] + [index * 0.2, 0.12, 0])
    top.createStandardBonds()
    top.setPeriodicBoxVectors(np.eye(3) * 4 * unit.nanometer)
    source = canonicalize(tmp_path, top, np.asarray(xyz), name="cysteines.pdbx")
    assert not any(a.name == b.name == "SG" for a, b in source.topology.bonds())
    return source


@pytest.mark.parametrize("missing_sg", [False, True])
def test_actual_pdbfixer_inferred_disulfide_is_not_an_approved_repair(tmp_path, monkeypatch, missing_sg):
    from pdbfixer import PDBFixer

    from neomd_preparation.protein.repair import group_and_repair_protein

    source = cysteine_pair(tmp_path, missing_sg=missing_sg)
    before = source.source_path.read_bytes()
    add_missing = PDBFixer.addMissingAtoms
    observed = []

    def observe_actual_backend(fixer, *args, **kwargs):
        result = add_missing(fixer, *args, **kwargs)
        observed.append(any(a.name == b.name == "SG" for a, b in fixer.topology.bonds()))
        return result

    monkeypatch.setattr(PDBFixer, "addMissingAtoms", observe_actual_backend)
    with pytest.raises(PlanValidationErrors, match="bonds") as error:
        group_and_repair_protein(source, platform="CPU", seed=123)
    assert observed == [True]  # The real public backend actually inferred the edge.
    assert ("generated bonds" if missing_sg else "raw bonds between retained") in str(error.value)
    assert source.source_path.read_bytes() == before
    assert not any(a.name == b.name == "SG" for a, b in source.topology.bonds())
