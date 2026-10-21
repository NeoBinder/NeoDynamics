# Native preparation v1

`neomd.native-prepare/v1` describes preparation configuration, distinct from a
Run Plan. `native-fields.json` declares fields, types, units, executor defaults
and path/resource roles. Defaults describe executor behavior; omitted fields
remain omitted. Fixtures are loader and protocol examples, not scientific
qualification or usable molecular inputs.

The public Python entrypoints in `neomd.native_preparation` and CLI
`neomd inspect-prepare` inspect bounded native YAML. The original text remains
available even when unknown fields fail validation. Integer keys are admitted
only in `protein.custom_resname_dict` list entries. Mapping order, sequence
order, types, omission and explicit null remain significant.

Typed nodes are `[tag,value]`; tags are `null`, `bool`, `int`, `float`,
`string`, `seq` and `map`. Mapping values are ordered pairs of typed keys
and values, sequences are lists, integers are decimal strings, floats are
RFC8785 binary64 strings. SHA256 hashes compact UTF8 JSON of the tree. Input
limits are 32768 UTF8 bytes, depth 32 and 10000 nodes; aliases, anchors, explicit
tags, merge keys, nonfinite values and duplicate keys are rejected. Integers
must fit ±9007199254740991. Unknown and unsupported fields cannot be exported
as target compatible. JSON export must produce the same tree under the native
YAML loader; otherwise `json_native_type_mismatch` requires retaining YAML.

The optional `classic-source-box/v1` profile restricts fields and builtin
resources. Structural success does not check source contents, periodic boxes,
residue completeness, covalent topology, forcefield success or prepared-System
fitness; those are separate executor checks. No metadata is inserted into the
scientific configuration. Preparation entrypoints preserve native units and
executor defaults, including omission behavior.
