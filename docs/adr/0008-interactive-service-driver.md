# ADR-0008: Separate interactive service driver and reusable engine stepping

Status: implemented. The CLI and the service are separate entry points and
packages: `neomd` owns science and the CLI; `neomd_server` owns session policy.
The HTTP entry point, durable ownership and transport are service concerns.

## Boundaries

`neomd.driver.MDStepper` owns one persistent boundary scheduler. `advance(n)`
performs at most `n` integration steps; it does not rebuild a System, reset an
integrator, close probes or finalize a method. Absolute probe/deposition/ramp
cadences survive arbitrary budget boundaries. Probes still run before method
hooks at a shared boundary. `finish()` finalizes once. The CLI's `run_md` and
prepared-method runner drain this same scheduler to completion.

`PreparedMethod` additionally exposes optional scientific preparation,
parameter descriptors, a parameter callback and a method-state checkpoint hook.
GaMD preparation advances its original cMD sample schedule in bounded portions;
its partial samples are saved in `gamd_preparation.json`. A production resume
still requires `gamd_calibration.json`; missing production calibration is an
error. Public engine helpers provide timestep conversion, coordinate hashing,
minimization settings, final structure writing and scientific minimization QC.
The engine never imports `neomd_server`.

`neomd_server.driver.open_session` creates the appropriate service driver.
`InteractiveSession` owns one prepared method and Context; RBFE opens an
`InteractiveLadderSession`, with independent Contexts and `seed + window_index`
for its windows. A window transition occurs at an advance boundary. The public
status/frame has cumulative `step` and `time_ps`; RBFE also reports window
index, window step and window time. Scientific window artifacts retain their
ordinary `window_XX` layout. Lambda cannot change inside a window.

## Ownership and method controls

Construction and every session call must occur on the same owner thread.
Transport handlers submit commands to that owner; they never touch a Context.
Client disconnection does not pause simulation. Pause, resume and stop are
explicit commands, applied between bounded advances.

| Method | Phase/advance | Qualified live parameters |
| --- | --- | --- |
| min | Bounded minimization calls | Particle forces and named positional tethers |
| eq / md / prod | Dynamics | Particle forces and named positional tethers |
| metadynamics | Absolute hill cadence | Future hill height (kJ/mol); deposition enabled |
| smd | Original 5000-step ramp cadence | Advertised existing distance/angle ramp targets; explicit remaining-schedule replacement |
| opes | Absolute deposit cadence | Deposition enabled; the existing KDE model is retained |
| gamd | Bounded cMD calibration then production | Sigma0 during calibration only; particle manipulation starts in production |
| rbfe | Independent ordered windows | Particle manipulation within the current window; fixed window lambda |

Static CV definitions, temperatures defining bias models, WT bias factors, OPES
barriers and GaMD production calibration parameters are not live editable.
Changing those would change the meaning of historical sampling state. The
method's advertised descriptors are authoritative; an unknown or unavailable
parameter is rejected. SMD target edits require `replace_schedule=True` and
remain in force at later scheduled update boundaries.

Minimization advances restart the optimizer's memory between calls. Its finite
iteration budget is reported as `minimization_iterations_budgeted`, not an
observed iteration count or proof of convergence; `time_ps` is null. Bounded
integration/optimization calls are cancellation points, not hard wall-clock
interrupts inside OpenMM. A process supervisor can handle a stuck kernel.

## Particle interaction seam

The optional `InteractiveOps` port reserves one zero-strength external force
before Context/boost construction. Live updates use stable System particle
indices, positions in nm, forces in kJ/mol/nm and positional force constants in
kJ/mol/nm². For one particle the force energy is
`-f·x + k|x-x0|²/2 + offset`. The offset preserves the exact energy when several
named harmonic tethers are combined algebraically. OpenMM updates per-particle
parameters in the existing Context; it does not reinitialize or reseed.
The fake adapter implements the same external-force formula for focused tests;
its other documented scientific approximations remain unchanged.

Targets use the same unwrapped coordinate frame as the returned coordinates.
The caller supplies real prepared-system indices, including any generated
particles; display identifiers must be mapped explicitly. Frames return the
ordered particle indices and full System particle count.

Transient drag forces replace the previous drag set. Their lifetime is bounded
by both a step deadline and a monotonic wall-clock deadline, whichever is met
first. Defaults are 1000 steps and 500 ms; maximum requests are 10000 steps and
2000 ms. The driver stops an advance at the step deadline, clears forces at the
boundary, and checks wall expiry before/after advances. Persistent named tethers
are independent of drag expiry. Recovery never revives a transient drag force.
There are at most 64 named tethers and 4096 distinct controlled particles;
commands publish these limits and numerical ranges in status.

## Recovery and acknowledgements

A successful command returns the absolute application step and monotonically
increasing `control_revision`. Checkpoint/stop commands persist the revision
returned in their acknowledgement. Validation collects field, unit, index,
shape and range problems before applying a command. The host additionally owns
command IDs, idempotency, fencing, authentication and admission.

A checkpoint descriptor lists the required artifacts explicitly, including files
read during recovery that a new sink instance has not itself written. It binds
`interactive_state.json` to the exact checkpoint SHA256 and to the manifest and
method-state companions. These hashes are verified before preparing the method
or trimming its tapes. RBFE also binds its active window metadata/checkpoint in
`interactive_ladder.json`. The service must publish the whole descriptor's files
as an atomic immutable generation; the working output directory is mutable and
must not be mistaken for such a generation.

A checkpoint is platform-, system- and OpenMM-version-bound. Binary state,
method ledgers and interactive metadata are all required. This is not a portable
cross-machine restart contract. Detailed scientific errors go to the owner
logger; public status contains the error type so filesystem paths do not leak.

## Verification scope

Public-interface tests compare bounded and batch artifacts/coordinates for all
built-in methods, cover pause/resume, calibration recovery, target replacement,
force expiry, revisions and checkpoint integrity, and exercise every built-in
method with OpenMM 8.6 on CPU. The RBFE CPU test uses the explicit mock lambda
bias; it does not qualify optional softcore toolchains. Existing golden tapes
remain the regression reference. GPU throughput, 100000-atom/10-Hz preview,
two-session admission and desktop client qualification require the service and
its deployment/client gates; small CPU tests do not establish those capacities.
