# Declarative ligand package import

`import_declarative_ligand` turns a reviewed `ChemicalState` and two local,
declarative JSON assets into a version 1 `ParameterPackage`. It verifies exact
input bytes, complete atom and bond mapping, typed force terms, exception
coverage and the decimal charge residual before returning any package. It
does not construct or run an OpenMM system. Its `integrity_passed` report means
that these structural checks passed; numerical validation and scientific
qualification remain separate gates.

## Input boundary

Pass a canonical RFC 8785 JSON manifest as bytes and a list of exactly two
`(name, bytes)` members named `structure.json` and `parameters.json`:

```python
from pathlib import Path
from neomd.component_import import import_declarative_ligand

source = Path("reviewed-ligand")
result = import_declarative_ligand(
    accepted_state,
    (source / "manifest.json").read_bytes(),
    [(name, (source / name).read_bytes())
     for name in ("structure.json", "parameters.json")],
)
package, assets = result.package, result.assets
```

The caller chooses and reads its local files. The importer accepts bytes only;
it never follows an input path, URL, include, script or plugin. It rejects
extra and duplicate members, XML, unsupported JSON fields and noncanonical
JSON. The manifest is limited to 1 MiB and each member to 32 MiB. These bounds
and member names are part of this first import profile.

The manifest has exactly these fields:

| Field | Meaning |
| --- | --- |
| `format` | `neomd-declarative-ligand-import-v1` |
| `state_id`, `state_sha256` | Accepted state UUID and RFC 8785 canonical JSON SHA-256 |
| `tool_versions` | Nonempty list of unique `{tool, version}` source declarations |
| `compatibility` | The version 1 package compatibility profile |
| `dependencies` | Empty array; remote or nested dependencies are unsupported |
| `assets` | Exactly two `{name, sha256, size_bytes}` rows for the named members |

Each asset SHA-256 and size describes its exact encoded input bytes. The
`structure.json` object has `format: "neomd-mapped-ligand-structure-v1"`, the
same state ID/hash, and complete `atoms` and `bonds` arrays. Atom rows have
`state_atom_id`, `name`, `element`, `formal_charge`, `stereo`; bond rows have
`atom_ids`, `order`, `stereo`. These must reproduce the accepted state graph.
Names are retained in the emitted atom map, in force-particle order.

`parameters.json` uses `format: "neomd-component-parameters-v1"` and explicit
`atoms`, `bonds`, `angles`, `torsions`, `exceptions`, `constraints`, and
`virtual_sites` arrays. Its particle rows are in parameter order and each
contains `state_atom_id`, `mass_amu`, `charge_e`, `sigma_nm`, and
`epsilon_kj_mol`. `charge_e` is an exact decimal string. The importer preserves
the original parameter bytes and charge strings. All state bonds need exactly
one bond term or constraint; exception pairs must cover graph distances 1–3.
Nonempty virtual sites are not supported by this profile. The complete force
row vocabulary and numeric bounds are enforced by `validate_package_assets`.

The returned `assets` mapping is keyed by opaque artifact UUID, with exact
bytes for every package reference. Its atom map, force asset and QC report are
canonical JSON. A package can be consumed only after its versioned manifest
and referenced bytes are checked together. Artifact `content_sha256` values
hash exact encoded transfer bytes; `state_sha256` and other document hashes
hash RFC 8785 canonical decoded JSON.

## Explicit charge normalization

Ordinary import sums the supplied decimal charges without changing their
strings. The residual from the accepted state's net formal charge must be at
most `1e-6 e`; the observed sum and residual are retained in the package and
QC report. A caller may submit an explicit reviewed
charge correction through `normalize_imported_ligand`:

```python
from neomd.component_import import normalize_imported_ligand

result = normalize_imported_ligand(
    accepted_state, manifest_bytes, members,
    charge_adjustments=[{"state_atom_id": "1", "charge_e": "0.4000"}],
    decision_sha256=reviewed_decision_sha256,
)
```

This creates a **different package** with a distinct charge method, cache key
and provenance asset. The original `parameters.json` bytes remain referenced
as an opaque input asset; the corrected force asset is separate. The
normalization record binds the original byte hash, decision hash and each
before/after string. The caller remains responsible for the scientific basis
of the correction; import integrity alone does not qualify the parameters for
simulation or public availability.
