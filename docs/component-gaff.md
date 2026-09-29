# Generated GAFF component package

`neomd.component_gaff.build_gaff_package` turns one explicitly accepted,
all-atom `ChemicalState` into an immutable `ParameterPackage`. The first
profile accepts a complete reviewed state with fully specified tetrahedral
stereocenters. Potential but unspecified stereocenters and double-bond stereo
remain unresolved. A source or bound
`ComponentInstance` pose never enters generation. The provider embeds a
separate RDKit ETKDGv3 conformer with a recorded seed. For stereogenic states,
the generation conformer is relaxed with MMFF94s, and all declared CIP labels
are checked before and after relaxation. The provider fixes that conformer
through the legacy antechamber backend, and retains the exact SDF bytes sent
to the tool. The method records the observed `gaff-2.2.20` resource and
`-at gaff2 -c abcg2` command. `abcg2` is the backend option name here; it is
not relabeled as another charge method.

The backend starts with the antechamber MOL2 partial charges, then distributes
the difference from the reviewed net formal charge in proportion to each
atom's absolute partial charge when their absolute sum is nonzero. The
resulting template charges, atom map and complete OpenMM force terms are
stored as immutable assets. Assembly consumes those terms and never reruns
antechamber. The `integrity_passed` QC status confirms structural integrity
only. Independent numerical and scientific comparisons remain separate gates.
The OpenMM parameter system loads ff14SB, TIP3P, GAFF2, then the generated
template, matching the existing combined builder. XML order affects parameter
precedence; the order and all four exact resource hashes are recorded.

## Use

```python
from neomd.component_gaff import build_gaff_package, gaff_cache_key

result = build_gaff_package(
    accepted_state,
    package_id=package_uuid,
    artifact_ids={
        "atom_map": atom_map_uuid,
        "force": force_uuid,
        "template": template_uuid,
        "gaff_xml": gaff_xml_uuid,
        "gaff_dat": gaff_dat_uuid,
        "protein_xml": protein_xml_uuid,
        "water_xml": water_xml_uuid,
        "conformer": conformer_uuid,
        "provenance": provenance_uuid,
        "qc": qc_uuid,
    },
    seed=20260926,
)
manifest = result.package
encoded_assets = result.asset_bytes
cache_key = gaff_cache_key(manifest)
```

The caller persists `package_bytes` and every UUID-addressed asset byte string.
The force JSON uses the shared component vocabulary; the generated residue
ffxml, GAFF XML/dat resources, ff14SB and TIP3P compatibility XML, actual input SDF,
and a canonical provenance JSON are retained with content hashes. The
provenance records the exact command arguments and charge-correction rule.
The cache key binds the state hash, method/tool/resource identities, options,
actual conformer hash and seed. Package UUID and source pose are absent.

The pinned runtime is AmberTools 26.0 from an installed package receipt.
The provider verifies antechamber, parmchk2 and the helper executables
observed in the real JZ4 ABCG2 path (`sqm`, `am1bcc`, `atomtype`, `bondtype`),
their wrappers, `amber.sh`, and a hash of the installed package inventory.
Each top-level subprocess runs in an isolated temporary directory with a
180-second bound for achiral states and a 600-second bound for stereogenic
states. A timeout terminates the entire AmberTools process group. Missing or unverifiable tools return a typed unavailable
error; tool output is excluded from public diagnostics. Other molecules may
exercise additional helper paths, so qualification remains specific to the
measured chemistry scope.

The deprotonated TCH (−1) state with 11 reviewed stereocenters has a retained
79-atom GAFF package and a direct same-method AmberTools/OpenMM reference in
the P16 task record. This verifies the isolated ligand terms and CPU energy
and forces at the generation conformer. Full 3QQA bound-pose assembly and
System qualification are separate gates. The deposited full 3QQA source has
16 polymer residues without coordinates in each chain, so preparation raises
a missing-position error. A modeling algorithm must supply a new
coordinate-complete source before full System qualification.

The installed [openmmforcefields GAFF generator source](https://github.com/openmm/openmmforcefields/blob/main/openmmforcefields/generators/template_generators.py)
and [OpenFF GAFF preparation example](https://docs.openforcefield.org/en/latest/examples/openforcefield/openff-interchange/experimental/openmmforcefields/gaff.html)
describe the upstream generator context. The exact local binary, resource and
charge behavior are determined from the locked runtime and retained receipt.
