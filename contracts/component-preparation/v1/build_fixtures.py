"""Regenerate deterministic, synthetic contract fixtures and JCS link hashes."""

import copy
import hashlib
import json
from pathlib import Path

import rfc8785

ROOT = Path(__file__).resolve().parent / "fixtures"
POSITIVE = ROOT / "positive"
NEGATIVE = ROOT / "negative"
POSITIVE.mkdir(parents=True, exist_ok=True)
NEGATIVE.mkdir(parents=True, exist_ok=True)
BASE = "urn:neodyn:component-preparation:v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def jcs(document):
    return digest(rfc8785.dumps(document))


def doc(document_kind, **fields):
    return {"schema_id": BASE + ":" + document_kind, "schema_version": 1, **fields}


def artifact(kind, number, content=b"{}", media_type="application/json"):
    return doc("artifact-reference",
               artifact_id=f"00000000-0000-4000-8000-{number:012x}", kind=kind,
               content_sha256=digest(content), size_bytes=len(content),
               decoded_size_bytes=len(content), encoding="identity", media_type=media_type)


def write(directory, name, value):
    (directory / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")


state = doc("chemical-state",
            state_id="00000000-0000-4000-8000-000000000001",
            atoms=[{"id": "1", "element": "H", "formal_charge": 0, "stereo": "none"},
                   {"id": "2", "element": "O", "formal_charge": 0, "stereo": "none"},
                   {"id": "3", "element": "H", "formal_charge": 0, "stereo": "none"}],
            bonds=[{"atom_ids": ["1", "2"], "order": "single", "stereo": "none"},
                   {"atom_ids": ["2", "3"], "order": "single", "stereo": "none"}],
            net_formal_charge=0, hydrogen_policy="explicit_all",
            tautomer_id="water", protonation_id="neutral-water",
            review={"status": "accepted", "decision_id": "00000000-0000-4000-8000-000000000002",
                    "evidence": [{"kind": "calculation",
                                  "source_sha256": digest(b"synthetic contract water only"),
                                  "detail": "Synthetic contract fixture only; no provider qualification."}]})
instance = doc("component-instance",
               instance_id="00000000-0000-4000-8000-000000000003",
               source_topology_id="00000000-0000-4000-8000-000000000004",
               source_revision_id="00000000-0000-4000-8000-000000000005",
               state_id=state["state_id"], state_sha256=jcs(state), role="water",
               included=True, frame_id="00000000-0000-4000-8000-000000000006",
               atom_bindings=[{"state_atom_id": str(i), "source_atom_id": str(i)}
                              for i in range(1, 4)])
package = doc("parameter-package",
              package_id="00000000-0000-4000-8000-000000000007",
              state_id=state["state_id"], state_sha256=jcs(state), origin="generated",
              method={"method_id": "synthetic.contract.fixture", "version": "0",
                      "charge_method_id": "fixture-exact", "tool_versions": [
                          {"tool": "fixture-builder", "version": "1"}],
                      "resource_hashes": [{"resource_id": "synthetic-water",
                                           "sha256": digest(b"synthetic-water")}],
                      "options_sha256": digest(b"synthetic-options"),
                      "conformer_sha256": None, "seed": None},
              atom_map=artifact("atom_map", 8),
              assets=[artifact("parameter_asset", 9, b"abc", "application/octet-stream")],
              compatibility={"protein_family": "fixture", "ligand_family": "fixture",
                             "water_family": "fixture", "ion_family": "fixture",
                             "combining_rule": "lorentz-berthelot",
                             "electrostatic_14_scale": 1, "vdw_14_scale": 1,
                             "unit_system": "openmm_nm_kj_mol_e",
                             "nonbonded_method": "pme", "cutoff_nm": 1,
                             "switch_distance_nm": None, "constraints": "HBonds",
                             "hydrogen_mass_amu": 4, "virtual_sites": False},
              charge_sum_e=0, qc={"status": "unreviewed", "report": artifact("qc_report", 10)})
recipe = doc("preparation-recipe",
             recipe_id="00000000-0000-4000-8000-00000000000b",
             source_topology={"schema_id": "urn:neodyn:molecular-topology:v1",
                              "topology_id": instance["source_topology_id"],
                              "revision_id": instance["source_revision_id"],
                              "content_sha256": digest(b"synthetic topology bytes"),
                              "size_bytes": 32, "decoded_size_bytes": 64, "encoding": "gzip"},
             assignments=[{"instance_id": instance["instance_id"],
                           "instance_sha256": jcs(instance),
                           "state_id": state["state_id"], "state_sha256": jcs(state),
                           "package_id": package["package_id"],
                           "package_sha256": jcs(package)}],
             assembly={"box_policy": "generate_orthorhombic", "padding_nm": 1,
                       "retained_water_policy": "retain_and_solvate", "water_model": "tip3p",
                       "ionic_strength_mol_per_l": 0.1, "neutralize": True,
                       "solvation_seed": 20260926,
                       "positive_ion": "Na+", "negative_ion": "Cl-", "constraints": "HBonds",
                       "hydrogen_mass_amu": 4, "nonbonded_method": "pme", "cutoff_nm": 1,
                       "switch_distance_nm": None})
catalog = doc("capability-catalog", catalog_revision_sha256=digest(b"fixture catalog"),
              methods=[{"method_id": "synthetic.contract.fixture", "version": "0",
                        "chemistry_scope": ["test-water-only"], "import_mode": "generated",
                        "resource_hashes": package["method"]["resource_hashes"],
                        "availability": "unavailable", "qualification": None,
                        "reason": "Synthetic contract fixture has no scientific qualification."}])
diagnostic = doc("diagnostic", code="invalid_reference", path="/assignments/0/package_sha256",
                 detail="Referenced package identity differs.")

positive = {"artifact-reference": package["assets"][0], "chemical-state": state,
            "component-instance": instance, "parameter-package": package,
            "preparation-recipe": recipe, "capability-catalog": catalog,
            "diagnostic": diagnostic}
for name, value in positive.items():
    write(POSITIVE, name, value)


def invalid(name, kind, mutation, code):
    value = copy.deepcopy(positive[kind])
    mutation(value)
    write(NEGATIVE, name, value)
    negative_index[name] = {"kind": kind, "code": code}


negative_index = {}
invalid("artifact-reference-unsafe-path", "artifact-reference",
        lambda d: d.update(path="../../private"), "invalid_reference")
invalid("artifact-reference-oversized", "artifact-reference",
        lambda d: d.update(size_bytes=536870913, decoded_size_bytes=536870913), "oversized_payload")
invalid("artifact-reference-hash-shape", "artifact-reference",
        lambda d: d.update(content_sha256="A" * 64), "invalid_reference")
invalid("chemical-state-charge-sum", "chemical-state",
        lambda d: d.update(net_formal_charge=-1), "mapping_mismatch")
invalid("chemical-state-duplicate-atom", "chemical-state",
        lambda d: d["atoms"].append(copy.deepcopy(d["atoms"][0])), "mapping_mismatch")
invalid("chemical-state-duplicate-bond", "chemical-state",
        lambda d: d["bonds"].append(copy.deepcopy(d["bonds"][0])), "mapping_mismatch")
invalid("chemical-state-unresolved-stereo", "chemical-state",
        lambda d: d["atoms"][0].update(stereo="unresolved"), "chemistry_unresolved")
invalid("component-instance-duplicate-source", "component-instance",
        lambda d: d["atom_bindings"][1].update(source_atom_id="1"), "mapping_mismatch")
invalid("component-instance-missing-binding", "component-instance",
        lambda d: d["atom_bindings"].pop(), "mapping_mismatch")
invalid("parameter-package-unsafe-asset", "parameter-package",
        lambda d: d["assets"][0].update(url="https://invalid.example/asset"), "invalid_reference")
invalid("parameter-package-charge-sum", "parameter-package",
        lambda d: d.update(charge_sum_e=-1), "mapping_mismatch")
invalid("preparation-recipe-duplicate-assignment", "preparation-recipe",
        lambda d: d["assignments"].append(copy.deepcopy(d["assignments"][0])), "mapping_mismatch")
invalid("preparation-recipe-version", "preparation-recipe",
        lambda d: d.update(schema_version=2), "contract_version_mismatch")
invalid("preparation-recipe-wrong-package-hash", "preparation-recipe",
        lambda d: d["assignments"][0].update(package_sha256="0" * 64), "mapping_mismatch")
invalid("preparation-recipe-missing-solvation-seed", "preparation-recipe",
        lambda d: d["assembly"].pop("solvation_seed"), "invalid_reference")
invalid("preparation-recipe-oversized-solvation-seed", "preparation-recipe",
        lambda d: d["assembly"].update(solvation_seed=4294967296), "invalid_reference")
invalid("capability-catalog-false-qualification", "capability-catalog",
        lambda d: d["methods"][0].update(availability="qualified"), "invalid_reference")
invalid("diagnostic-bad-pointer", "diagnostic",
        lambda d: d.update(path="/bad~2escape"), "invalid_reference")
(ROOT / "negative-index.json").write_text(json.dumps(negative_index, indent=2) + "\n")
