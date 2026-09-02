# NeoDynamics

An MD SDK on OpenMM: generic dynamics, well-tempered metadynamics, and
steered MD, driven by one immutable Plan per run. This glossary fixes the
domain language; architecture and decisions live in AGENTS.md and the
docs/ tree.

## Language

### Steering

**SMD (steered MD)**:
A sampling method that drives an experiment along restraint parameters
over time (`method: smd`); not a specific pull protocol.
_Avoid_: pulling, guided MD

**Parameter ramp**:
A restraint parameter spelled as a LIST of values in a `smd:` entry,
piecewise-linearly interpolated over `steps`. A scalar spelling is a
constant parameter. Classic constant-velocity pulling is a ramp on the
reference or bound, not a separate mode.
_Avoid_: schedule, protocol (too generic)

**Update cadence**:
The fixed 5000-step granularity at which interpolated ramp values are
pushed to the kernel — a staircase approximation of the ramp, v1 behavior
kept verbatim. Not configurable. On a resume the initial push snaps to
the enclosing boundary, so the resumed staircase equals an uninterrupted
run's.
_Avoid_: update interval (suggests a knob)

**Ramp value at a step**:
The piecewise-linear value anchored at `int(steps / (len(values) - 1))`
per segment, last anchor forced to `steps`; the value a pushed parameter
carries between update boundaries.

**smd tape (`smd.tsv`)**:
The steered-MD artifact: step, the entry's geometric observable, the
current ramp values (spec units; a reference-position ramp expands to
x/y/z columns), and the entry's bias energy. Replaces v1's `smd.csv`
(never parsed by old tools anyway — new format, acknowledged break).
Switched by `output.report_smd` (bool, default on) — the driver reads
the switch, the method never does.

**Static restraint (during SMD)**:
A `restraint:` section entry running alongside the smd entries at fixed
parameters (e.g. holding the protein while the ligand is steered);
reported to `restraint.tsv` like in any MD run — attached by the driver,
not the method.

### Surfaces

**BiasParamOps**:
The optional kernel capability (negotiated via `provides()`) to update one
installed bias's global parameter mid-run — the port-level spelling of
v1's `context.setParameter` loop.
_Avoid_: setParameter (that is the openmm adapter's mechanism, not the seam)

**Method**:
A registry entry owning a full sampling workflow (metadynamics, smd).
`drive()` calls `entry.prepare(...) -> PreparedMethod` — biases installed,
resume planned, tapes built — then runs the loop itself; the method does
physics, the driver owns reporting (which artifacts run).

**PreparedMethod**:
What a method hands the driver before the loop: the `on_step` physics hook
+ interval, its tape probes (keyed by filename), its resume plan, an
optional per-hook artifact-progress reporter, and a `finish` writing the
end-of-run artifacts. `driver.run_prepared_method` is the one place that
assembles the probe list (plan defaults + restraint tape + switch-gated
method tapes) and runs the loop.

**Tape switch**:
A driver-owned output key gating one method tape's inclusion in a run
(`driver._TAPE_SWITCHES`): `report_smd` → `smd.tsv`, default on. The
method still builds the tape; the driver decides whether it runs.

### Restraints & boxes

**Multi-bond bias (`BondIR`)**:
One `CustomCentroidBondForce` holding N bonds with per-bond parameters
(`BiasIR.bonds`); each bond evaluates the same expression with its own
values. Exists for the group economy: N distance pairs cost ONE force
group per side, not N.
_Avoid_: multi-restraint force (the restraint is one entry; the force is
the packing detail)

**distances restraint**:
A `restraint:` entry whose `params` is a list of per-pair entries
(`grp1`/`grp2`/`restr_k`/`min_nm`/`max_nm`/`order`), packed into one
min-wall + one max-wall force. v1 179ae35 behavior, ported post-flip.
A `0.0` bound is a real bound here (`is not None` check), unlike the
single `distance` type where 0.0 means absent.
_Avoid_: distance (singular — that is the one-pair type)

**Runtime-box header**:
`last.pdbx` carries the periodic box the context holds at write time, not
the input file's header box; a fresh start takes its initial box from the
structure file's header, falling back to the System default. Vacuum
systems never gain a box record. v1 8d04b0c semantics.
_Avoid_: box correction (implies a transformation), CRYST1 fix
