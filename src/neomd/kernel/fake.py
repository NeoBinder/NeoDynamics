"""FakeKernel — the deterministic, openmm-free textbook-Langevin adapter:
the CI workhorse whose trajectories are bit-stable for a given seed.

Physics simplifications and bit-stability contracts live at
:class:`FakeKernel`; see ``docs/architecture.md`` and
``docs/adr/0005-gamd-boost-seam.md``.
"""

from __future__ import annotations

import ast
import math
import pickle

import numpy as np

from .port import (
    CVIR,
    BiasIR,
    BoostChannelIR,
    BoostReading,
    EnergyReport,
    KernelFactory,
    KernelSpec,
    SystemData,
    cv_is_angular,
    pick_free_force_group,
    to_canonical,
)

__all__ = ["FakeKernel"]

#: molar gas constant in kJ/(mol K) — openmm unit.MOLAR_GAS_CONSTANT_R
_R_KJ_MOL_K = 8.31446261815324e-3

#: sentinel: "this parameter had no override before energy_with_params"
_UNSET = object()


# ----------------------------------------------------------------------
# synthetic default system
# ----------------------------------------------------------------------

def _default_system_data() -> SystemData:
    """4 particles of 12 dalton on a small tetrahedron, non-periodic."""
    s = 2.0 / math.sqrt(6.0)  # tetrahedron on the unit-ish sphere
    positions = np.array([
        [s, s, s], [-s, -s, s], [-s, s, -s], [s, -s, -s],
    ], dtype=np.float64)
    return SystemData(positions=positions,
                      masses=np.full(4, 12.0, dtype=np.float64),
                      box_vectors=None)


# ----------------------------------------------------------------------
# numpy geometry — mass-weighted COM / angle / dihedral
# ----------------------------------------------------------------------

def _com(masses: np.ndarray, positions: np.ndarray, idxlist) -> np.ndarray:
    """Mass-weighted center of geometry."""
    idx = np.asarray(idxlist, dtype=int)
    m = masses[idx]
    return (m[:, None] * positions[idx]).sum(axis=0) / m.sum()


def _angle_3points_rad(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle a-b-c in radians."""
    vec1 = a - b
    vec2 = c - b
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dihedral_rad(p1, p2, p3, p4) -> float:
    """Torsion p1-p2-p3-p4 in radians (praxeolitic form)."""
    p1, p2, p3, p4 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4))
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return float(-np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


# ----------------------------------------------------------------------
# kind-driven CV geometry — the MIRROR of colvars.py's evaluate track
# (same dual-track discipline as the COM/angle/dihedral helpers above; the
# tests pin the two tracks in agreement).  These are NOT OpenMM corner-case
# mimicry: they are the plain numpy evaluation of the same literature
# formulas colvars.evaluate implements.
# ----------------------------------------------------------------------

def _kabsch_rmsd(mobile, reference) -> float:
    """Unweighted optimal-rotation RMSD (Kabsch), openmm RMSDForce
    semantics — see the colvars.py twin for the algorithm notes."""
    P = np.asarray(mobile, dtype=np.float64)
    Q = np.asarray(reference, dtype=np.float64)
    P = P - P.mean(axis=0)
    Q = Q - Q.mean(axis=0)
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    diff = P @ R - Q
    return float(np.sqrt((diff * diff).sum() / len(P)))


def _coordination_pair_sum(positions, groups, r0: float, nn: float, mm: float,
                           minimum_image=None) -> float:
    """Coordination number pair sum — the colvars._coordination_sum twin
    (PLUMED-style rational switching over the grp1 x grp2 atom pairs)."""
    g1 = np.asarray(groups[0], dtype=int)
    g2 = np.asarray(groups[1], dtype=int)
    pos = np.asarray(positions, dtype=np.float64)
    delta = pos[g1][:, None, :] - pos[g2][None, :, :]
    if minimum_image is not None:
        delta = minimum_image(delta)
    r = np.sqrt((delta * delta).sum(axis=2))
    x = r / float(r0)
    values = (1.0 - x ** nn) / (1.0 - x ** mm)
    cross = g1[:, None] != g2[None, :]
    return float(values[cross].sum())


def _path_sz(mobile, images, lam: float) -> tuple[float, float]:
    """(s, z) of the Branduardi path CV — the colvars._path_values twin
    (max-shifted log-sum-exp closed forms; see colvars.py for citations)."""
    msd = np.array([_kabsch_rmsd(mobile, image) ** 2 for image in images],
                   dtype=np.float64)
    a = -msd / (lam * lam)
    shift = float(a.max())
    weights = np.exp(a - shift)
    total = float(weights.sum())
    progress = float((np.arange(1, len(msd) + 1, dtype=np.float64)
                      * weights).sum() / total)
    distance = -lam * (shift + float(np.log(total)))
    return progress, distance


def _torsion_theta_rad(p1, p2, p3, p4) -> float:
    """The openmm CustomTorsionForce ``theta`` (radians), branch included.

    Same angle as ``_dihedral_rad`` but via the IUPAC atan2 form, which
    agrees with openmm bit-wise including the +/-pi branch of a planar-trans
    torsion (the reporter form returns -pi there, openmm +pi).
    """
    p1, p2, p3, p4 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4))
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    return float(np.arctan2(
        np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3)),
        np.dot(np.cross(b1, b2), np.cross(b2, b3))))


# ----------------------------------------------------------------------
# restricted expression interpreter (the openmm expression subset)
# ----------------------------------------------------------------------

_MATH_FUNCS = {
    "max": max, "min": min, "abs": abs, "sqrt": math.sqrt, "exp": math.exp,
    "atan": math.atan, "tan": math.tan, "sin": math.sin, "cos": math.cos,
}
_GEOMETRY = {"distance", "angle", "dihedral"}


def _evaluate_expression(source: str, variables: dict[str, float],
                         coms: np.ndarray | None = None) -> float:
    """Evaluate an openmm-style custom-force expression in numpy.

    Restricted to the subset of the openmm expression language in use:
    numbers, names, unary +/-, + - * / and ``^`` (power), max/min/abs/
    sqrt/exp/atan/tan/sin/cos, distance()/angle()/dihedral() between group
    centroids g1..gN, xN/yN/zN centroid coordinates, and ``";"``-separated
    intermediate assignments (openmm's statement syntax).  Anything else
    is rejected loudly.

    ``^`` is rewritten to ``**`` BEFORE parsing: openmm treats ``^`` as power
    with power precedence (``k*x^2`` == ``k*(x^2)``), whereas Python's ``^``
    is a loose bitwise XOR (``k*x^2`` would misparse as ``(k*x)^2``).
    """
    try:
        tree = ast.parse(source.replace("^", "**"), mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"cannot parse expression {source!r}: {exc}") from None
    env: dict[str, float] = {}
    result: float | None = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                raise ValueError(f"unsupported assignment in {source!r}")
            env[stmt.targets[0].id] = _eval_node(stmt.value, variables, env, coms, source)
        elif isinstance(stmt, ast.Expr):
            result = _eval_node(stmt.value, variables, env, coms, source)
        else:
            raise ValueError(f"unsupported statement in {source!r}")
    if result is None:
        raise ValueError(f"expression {source!r} has no value statement")
    return result


def _eval_node(node, variables, env, coms, source) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in variables:
            return variables[node.id]
        raise ValueError(f"unknown name {node.id!r} in expression {source!r}")
    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, variables, env, coms, source)
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        raise ValueError(f"unsupported unary op in {source!r}")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables, env, coms, source)
        right = _eval_node(node.right, variables, env, coms, source)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):  # "^" rewritten above
            return left ** right
        raise ValueError(f"unsupported binary op in {source!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise ValueError(f"unsupported call in {source!r}")
        name = node.func.id
        if name in _GEOMETRY:
            if coms is None:
                raise ValueError(f"{name}() needs centroid groups ({source!r})")
            idx = []
            for arg in node.args:
                if not isinstance(arg, ast.Name) or len(arg.id) < 2 or arg.id[0] != "g" \
                        or not arg.id[1:].isdigit():
                    raise ValueError(f"{name}() arguments must be g1..gN ({source!r})")
                idx.append(int(arg.id[1:]) - 1)
            if any(i >= len(coms) for i in idx):
                raise ValueError(f"{name}() references a missing group ({source!r})")
            pts = [coms[i] for i in idx]
            if name == "distance":
                return float(np.linalg.norm(pts[0] - pts[1]))
            if name == "angle":
                return _angle_3points_rad(pts[0], pts[1], pts[2])
            return _torsion_theta_rad(*pts)
        if name in _MATH_FUNCS:
            args = [_eval_node(a, variables, env, coms, source) for a in node.args]
            return _MATH_FUNCS[name](*args)
        raise ValueError(f"unknown function {name!r} in expression {source!r}")
    raise ValueError(f"unsupported syntax in {source!r}: {ast.dump(node)[:80]}")


def _convert_param(value: float, unit_name: str) -> float:
    """Param -> fake canonical value: THE shared table (port.to_canonical,
    openmm's canonicalization — degrees become radians)."""
    return to_canonical(value, unit_name)


# ----------------------------------------------------------------------
# the kernel
# ----------------------------------------------------------------------

class FakeKernel:
    """Deterministic textbook-Langevin KernelPort implementation.

    Pure numpy, NO openmm import.  Dynamics: Euler-Maruyama Langevin on
    free particles (``v += (F/m)*dt - gamma*v*dt +
    sqrt(2*gamma*kT/m*dt)*N(0,1)``, ``x += v*dt``, with F = 0) with
    Maxwell-Boltzmann velocity seeding ``N(0,1)*sqrt(kT/m)`` from
    ``numpy.random.RandomState(spec.seed)`` — the same stream continues as
    the Langevin noise source, so trajectories are BIT-STABLE for a given
    seed.  The fake deliberately does NOT mimic OpenMM corner-case
    behavior (settled decision 9): the parity suite and golden tapes guard
    the real physics, not this kernel.  It also deliberately has NO
    ``write_structure`` (a PDBx writer needs a real topology, so
    fake-kernel runs skip ``last.pdbx``; the ``last.ckpt`` snapshot is
    still written) and IGNORES ``spec.ml_region`` (ADR-0004): with no MM
    forces to embed, the torch-free ML/MM pipeline tier runs its mock NNP
    through the OPENMM adapter instead.
    """

    name = "fake"

    def __init__(self, spec: KernelSpec):
        self.spec = spec
        data = spec.system_data if spec.system_data is not None else _default_system_data()
        self._positions = np.array(data.positions, dtype=np.float64)
        self._masses = np.array(data.masses, dtype=np.float64)
        if self._positions.shape != (self._masses.shape[0], 3):
            raise ValueError(
                f"SystemData positions {self._positions.shape} do not match "
                f"masses {self._masses.shape}")
        self._box = (None if data.box_vectors is None
                     else np.array(data.box_vectors, dtype=np.float64))
        self._dt = float(spec.integrator.get("dt", 0.002))
        self._gamma = float(spec.integrator.get("friction_coeff", 1.0))
        self._kT = _R_KJ_MOL_K * spec.temperature

        # textbook Maxwell-Boltzmann draw; the stream continues as Langevin noise
        self._rng = np.random.RandomState(spec.seed)
        self._velocities = self._rng.normal(size=self._positions.shape) \
            * np.sqrt(self._kT / self._masses)[:, None]
        # per-particle Euler-Maruyama noise scale, nm/ps
        self._noise_scale = np.sqrt(2.0 * self._gamma * self._kT / self._masses * self._dt)

        self._step = 0
        self._biases: list[tuple[int, BiasIR]] = []
        self._next_group = 0
        #: label -> (TableSpec, values ndarray) for CustomCVTableForce biases
        self._tables: dict[str, tuple] = {}
        #: global-parameter name -> CANONICAL float pushed by set_bias_param
        #: (steered MD); _param_variables consults this first, bypassing
        #: to_canonical so a deg override is not converted twice
        self._param_overrides: dict[str, float] = {}
        #: boost channels (port.BoostOps, ADR-0005): label -> mutable params
        self._boost: dict[str, dict] = {}
        #: label -> BoostReading of the most recent step ({} before any)
        self._boost_last: dict[str, BoostReading] = {}

    # ------------------------------------------------------------------
    # state observation
    # ------------------------------------------------------------------

    @property
    def num_particles(self) -> int:
        return self._positions.shape[0]

    @property
    def current_step(self) -> int:
        return self._step

    @property
    def masses(self) -> np.ndarray:
        """Particle masses (dalton, (N,)) — a copy, like positions()."""
        return self._masses.copy()

    def positions(self) -> np.ndarray:
        return self._positions.copy()

    def box_vectors(self) -> np.ndarray | None:
        """The synthetic system's fixed box ((3, 3) nm rows, None = vacuum)."""
        return None if self._box is None else self._box.copy()

    def energy_forces(self) -> EnergyReport:
        """Bias potential (kJ/mol, sum of installed bias energies) + ZERO
        forces (bias forces are not propagated into the dynamics); kinetic
        energy from the kernel's own velocities, temperature from
        2*KE/(dof*R) with dof = 3N (the openmm StateDataReporter convention
        minus constraints, which the fake has none)."""
        potential = self._bias_potential(self._positions)
        kinetic = 0.5 * float((self._masses
                               * (self._velocities ** 2).sum(axis=1)).sum())
        dof = 3 * self.num_particles
        temperature = (2.0 * kinetic / (dof * _R_KJ_MOL_K)) if dof > 0 else None
        volume = (float(abs(np.linalg.det(self._box))) if self._box is not None
                  else None)
        # documented: the fake reports zero forces (bias forces are not
        # propagated into its dynamics)
        return EnergyReport(potential=potential,
                            forces=np.zeros_like(self._positions),
                            kinetic=kinetic, volume=volume,
                            temperature=temperature)

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def step(self, n: int) -> None:
        for _ in range(n):
            noise = self._rng.standard_normal(self._positions.shape)
            if self._boost:
                # GaMD: F* = sum_g s(g) * F_g (own-potential forces, scaled)
                force = self._boost_force()
                self._velocities = (
                    self._velocities * (1.0 - self._gamma * self._dt)
                    + self._noise_scale[:, None] * noise
                    + (force / self._masses[:, None]) * self._dt)
            else:
                # F/m = 0 (free particles); textbook Euler-Maruyama update
                self._velocities = (self._velocities * (1.0 - self._gamma * self._dt)
                                    + self._noise_scale[:, None] * noise)
            self._positions = self._positions + self._velocities * self._dt
            self._step += 1

    # ------------------------------------------------------------------
    # boost channels (port.BoostOps, ADR-0005)
    # ------------------------------------------------------------------

    def install_boost(self, channels) -> None:
        """Install (replace) the GaMD boost channels, verbatim IR params.

        With channels installed, the fake propagates its OWN (geometric
        bias) potential's forces — per force group, via the central
        finite-difference gradient :meth:`minimize` uses — rescaled per
        channel by ``s(g) = 1 - sum_c k_c*(E_c - P_c)`` (the exact
        gradient of ``V* = sum_g V_g + sum_c dV_c(P_c)``; the F = 0
        simplification holds only when NO boost is installed).  Channel
        target energies are the summed energies of their force groups
        (``()`` = every installed bias); the applied dV/P/s of the most
        recent step is readable via :meth:`boost_potentials` (the
        reweighting trace).  Must come after every ``install_bias``:
        later bias installs are refused (see :meth:`install_bias`), so a
        stale channel set can never silently mis-scale new forces.

        Typically installed at zero strength (k=0) and calibrated later via
        :meth:`set_boost_param`.
        """
        installed = {}
        for channel in channels:
            if not isinstance(channel, BoostChannelIR):
                raise TypeError(
                    f"install_boost takes BoostChannelIR objects, got "
                    f"{type(channel).__name__}")
            if channel.label in installed:
                raise ValueError(
                    f"duplicate boost channel label {channel.label!r}")
            installed[channel.label] = {
                "groups": tuple(channel.groups),
                "threshold": float(channel.threshold),
                "k": float(channel.k),
            }
        self._boost = installed
        self._boost_last = {}

    def set_boost_param(self, label: str, name: str, value: float) -> None:
        if label not in self._boost:
            raise KeyError(
                f"no boost channel labeled {label!r} "
                f"(installed: {sorted(self._boost) or 'none'})")
        if name not in ("threshold", "k"):
            raise ValueError(
                f"boost param name must be 'threshold' or 'k', got {name!r}")
        if name == "k" and value < 0.0:
            raise ValueError(f"boost k must be >= 0, got {value}")
        self._boost[label][name] = float(value)

    def boost_potentials(self) -> dict:
        return dict(self._boost_last)

    def torsion_force_groups(self) -> tuple[int, ...]:
        """Duck-typed dual-boost discovery (mirrors the openmm adapter):
        the force groups of installed torsion biases.  The fake has no
        system forces, so its whole potential lives in the installed
        biases — a dihedral restraint IS the dihedral energy a dual-boost
        channel targets.  Both torsion spellings match: a plain
        ``CustomTorsionForce`` and the dihedral-restraint triple's
        4-group ``CustomCentroidBondForce`` whose expression calls
        ``dihedral(g1..g4)``.  Group ids stay opaque (they are only ever
        handed back to ``install_boost``)."""
        def is_torsion(bias: BiasIR) -> bool:
            if bias.kind == "CustomTorsionForce":
                return True
            return (bias.kind == "CustomCentroidBondForce"
                    and bias.groups is not None and len(bias.groups) == 4
                    and "dihedral(" in bias.energy)

        return tuple(sorted(group for group, bias in self._biases
                            if is_torsion(bias)))

    def _channel_energy(self, channel: dict, positions: np.ndarray) -> float:
        """P of one channel: summed energy of its groups (all biases if ())."""
        if not channel["groups"]:
            return self._bias_potential(positions)
        selected = set(channel["groups"])
        return sum(self._bias_energy(bias, positions)
                   for group, bias in self._biases if group in selected)

    def _boost_force(self) -> np.ndarray:
        """F* for one step + the per-channel readings (the ADR-0005 math).

        Channel energies are evaluated at the step's STARTING positions
        (the integrator convention); each force group of the potential is
        then rescaled additively by every channel targeting it.
        """
        factors: dict[str, float] = {}  # label -> k*(E-P) while P < E, else 0
        readings: dict[str, BoostReading] = {}
        for label, channel in self._boost.items():
            energy = self._channel_energy(channel, self._positions)
            depth = channel["threshold"] - energy
            if channel["k"] > 0.0 and depth > 0.0:
                boost = 0.5 * channel["k"] * depth * depth
                factors[label] = channel["k"] * depth
            else:  # no boost: above threshold or zero strength
                boost = 0.0
                factors[label] = 0.0
            readings[label] = BoostReading(
                boost=boost, energy=energy, scale=1.0 - factors[label])
        self._boost_last = readings

        # group the installed biases by their (additive) scale, one
        # finite-difference gradient per distinct scale value
        by_scale: dict[float, set[int]] = {}
        for group, _bias in self._biases:
            scale = 1.0 - sum(factor for label, factor in factors.items()
                              if not self._boost[label]["groups"]
                              or group in self._boost[label]["groups"])
            by_scale.setdefault(scale, set()).add(group)
        force = np.zeros_like(self._positions)
        eps = 1e-6  # nm (the minimize() gradient resolution)
        for scale, groups in by_scale.items():
            force -= scale * self._numerical_gradient(  # force = -grad(E)
                self._positions, eps, groups)
        return force

    def minimize(self, tolerance: float = 10.0, max_iterations: int = 10000) -> None:
        """Deterministic steepest descent on the geometric bias potentials.

        Central finite-difference gradients; the step length grows/shrinks by
        backtracking.  Velocities are zeroed (minimization convention).
        """
        self._velocities[:] = 0.0
        eps = 1e-6  # nm
        step = 1e-2  # nm
        energy = self._bias_potential(self._positions)
        for _ in range(max_iterations):
            grad = self._numerical_gradient(self._positions, eps)
            gmax = float(np.abs(grad).max())
            if gmax <= tolerance:
                break
            direction = -grad / gmax
            while step > 1e-10:
                candidate = self._positions + step * direction
                new_energy = self._bias_potential(candidate)
                if new_energy < energy:
                    self._positions = candidate
                    energy = new_energy
                    step *= 1.5
                    break
                step *= 0.5
            else:
                break  # cannot improve any further at usable resolution

    def _numerical_gradient(self, positions: np.ndarray, eps: float,
                            groups: set | None = None) -> np.ndarray:
        """Central-difference gradient of the bias potential (all biases, or
        only the force groups in ``groups`` — the boost force needs the
        per-group pieces to rescale them independently)."""
        selected = None if groups is None else set(groups)

        def energy(pos):
            return sum(
                self._bias_energy(bias, pos) for group, bias in self._biases
                if selected is None or group in selected)
        grad = np.zeros_like(positions)
        flat = positions.reshape(-1)
        gflat = grad.reshape(-1)
        for i in range(flat.size):
            xp, xm = flat.copy(), flat.copy()
            xp[i] += eps
            xm[i] -= eps
            gflat[i] = (energy(xp.reshape(-1, 3))
                        - energy(xm.reshape(-1, 3))) / (2.0 * eps)
        return grad

    # ------------------------------------------------------------------
    # biases
    # ------------------------------------------------------------------

    def install_bias(self, bias: BiasIR) -> int:
        if self._boost:
            # ADR-0005 ordering: the boost rescales force groups by explicit
            # membership — a bias installed after it would silently escape
            # the scaled update.  Refuse loudly instead.
            raise RuntimeError(
                "cannot install_bias after install_boost (boost channels "
                "target an explicit force-group set); install biases first")
        group = self._pick_force_group()
        self._biases.append((group, bias))
        self._next_group += 1  # install counter (snapshot-format field)
        return group

    def _pick_force_group(self) -> int:
        """The shared port policy (pick_free_force_group), aligned with the
        openmm adapter: max free id first — 31, 30, …
        — so fake-kernel runs exercise the same ids production would."""
        return pick_free_force_group(
            (group for group, _ in self._biases),
            {group: self._bias_label(group, bias)
             for group, bias in self._biases})

    def clear_bias(self) -> None:
        self._biases.clear()
        self._next_group = 0
        self._param_overrides.clear()

    def set_bias_param(self, name: str, value: float) -> None:
        """Live update of one installed bias's global parameter
        (port.BiasParamOps, the steered-MD ramp push).

        The override stores the CANONICAL float verbatim (no
        to_canonical round-trip: dividing then re-multiplying by the deg
        factor is not bit-stable), so _param_variables must serve it
        directly.  Unknown names raise — a typo'd parameter would
        silently no-op everywhere else.
        """
        known = {pname for _, bias in self._biases for pname in bias.params}
        if name not in known:
            raise KeyError(
                f"no installed bias declares global parameter {name!r} "
                f"(installed: {sorted(known) or 'none'})")
        self._param_overrides[name] = float(value)

    def energy_with_params(self, params) -> float:
        """Potential energy at temporarily-perturbed GLOBAL parameters
        (port.ParamEnergy — the RBFE du tape's λ-evaluation seam, ADR-0007).

        The fake's potential is the sum of its installed bias energies, so
        this evaluates ``_bias_potential`` with ``params`` temporarily
        merged into ``_param_overrides`` (canonical floats, exactly like
        ``set_bias_param``) and restores the prior override state on exit —
        no stepping, no RNG draw, bit-deterministic for a given input.
        Unknown names raise like ``set_bias_param``.  Note the multi-bond
        forces (BiasIR.bonds) take their per-bond VALUES, not overrides —
        the same documented limitation as set_bias_param.
        """
        known = {pname for _, bias in self._biases for pname in bias.params}
        unknown = sorted(set(params) - known)
        if unknown:
            raise KeyError(
                f"no installed bias declares global parameter(s) "
                f"{', '.join(repr(n) for n in unknown)} "
                f"(installed: {sorted(known) or 'none'})")
        saved = {name: self._param_overrides.get(name, _UNSET)
                 for name in params}
        try:
            self._param_overrides.update(
                {name: float(value) for name, value in params.items()})
            return self._bias_potential(self._positions)
        finally:
            for name, prior in saved.items():
                if prior is _UNSET:
                    self._param_overrides.pop(name, None)
                else:
                    self._param_overrides[name] = prior

    def bias_values(self, positions: np.ndarray | None = None) -> dict[str, float]:
        """Geometric value of each installed bias in report units.

        distance in nm, angle/dihedral in degrees (neomd.colvars
        conventions); keys are bias labels (falling back to
        ``bias{group}`` for unlabeled biases).
        """
        pos = self._positions if positions is None else np.asarray(positions, dtype=np.float64)
        return {self._bias_label(gid, bias): self._bias_quantity(bias, pos)
                for gid, bias in self._biases}

    @staticmethod
    def _bias_label(gid: int, bias: BiasIR) -> str:
        return bias.label or f"bias{gid}"

    # -- geometry of one bias ------------------------------------------

    def _centroids(self, groups: list[list[int]], positions: np.ndarray) -> np.ndarray:
        return np.array([_com(self._masses, positions, grp) for grp in groups])

    def _minimum_image(self, delta: np.ndarray) -> np.ndarray:
        if self._box is None:
            return delta
        diag = np.diag(self._box)
        if not np.allclose(self._box, np.diag(diag)):
            return delta  # orthorhombic MIC only (documented limitation)
        return delta - diag * np.round(delta / diag)

    def _distance(self, coms: np.ndarray, i: int, j: int) -> float:
        return float(np.linalg.norm(self._minimum_image(coms[i] - coms[j])))

    def _bias_quantity(self, bias: BiasIR, positions: np.ndarray) -> float:
        """Human-report value of the bias geometry (nm / degrees).

        Multi-bond forces (BiasIR.bonds, the ``distances`` restraint) report
        their FIRST bond's value — the per-pair reporting track is the
        observables spec, this is the fake's single-scalar debug view.
        """
        if bias.kind == "CustomCVForce":
            if bias.cv is None:
                raise ValueError(f"bias {bias.label!r}: CustomCVForce needs cv (CVIR)")
            return self._cv_quantity(bias.cv, positions)
        if bias.kind == "CustomTorsionForce":
            if bias.torsion is None:
                raise ValueError(f"bias {bias.label!r}: CustomTorsionForce needs torsion")
            return math.degrees(_dihedral_rad(
                *[positions[i] for i in bias.torsion]))
        if bias.bonds is not None:
            if not bias.bonds:
                raise ValueError(f"bias {bias.label!r}: empty bond list")
            groups = bias.bonds[0].groups
        else:
            groups = bias.groups
        n_groups = len(groups)
        coms = self._centroids(groups, positions)
        if n_groups == 2:
            return self._distance(coms, 0, 1)
        if n_groups == 3:
            return math.degrees(_angle_3points_rad(coms[0], coms[1], coms[2]))
        if n_groups == 4:
            return math.degrees(_dihedral_rad(*coms))
        raise NotImplementedError(
            f"bias {bias.label!r}: no scalar quantity for {n_groups} group(s)")

    def _cv_quantity(self, cv: CVIR, positions: np.ndarray) -> float:
        """Report-unit value of one CVIR (degrees for angles, else natural
        units — nm or dimensionless).  Same contract as _cv_report_units;
        the kind-driven special paths live in _cv_expression_value."""
        return self._cv_report_units(cv, positions)

    # -- energy evaluation ---------------------------------------------

    def _bias_potential(self, positions: np.ndarray) -> float:
        return sum(self._bias_energy(bias, positions) for _, bias in self._biases)

    def _bias_energy(self, bias: BiasIR, positions: np.ndarray) -> float:
        coms: np.ndarray | None = None
        if bias.kind == "CustomCentroidBondForce":
            if bias.bonds is not None:
                # multi-bond mode: one force, N bonds,
                # per-bond parameters — the same expression summed over bonds
                total = 0.0
                for bond in bias.bonds:
                    coms = self._centroids(bond.groups, positions)
                    env = {name: float(bond.params[name])
                           for name in bias.params}
                    env.update(self._com_variables(coms))
                    total += _evaluate_expression(bias.energy, env, coms)
                return total
            if not bias.groups:
                raise ValueError(
                    f"bias {bias.label!r}: CustomCentroidBondForce needs groups")
            coms = self._centroids(bias.groups, positions)
            env = self._param_variables(bias.params, coms)
        elif bias.kind == "CustomTorsionForce":
            if bias.torsion is None:
                raise ValueError(
                    f"bias {bias.label!r}: CustomTorsionForce needs torsion")
            env = {"theta": _torsion_theta_rad(*[positions[i] for i in bias.torsion])}
            env.update(self._param_variables(bias.params, None))
        elif bias.kind == "CustomCVForce":
            if bias.cv is None:
                raise ValueError(f"bias {bias.label!r}: CustomCVForce needs cv (CVIR)")
            name = bias.cv.label or "cv"
            cv_value = self._cv_expression_value(bias.cv, positions)
            env = {name: cv_value, "cv": cv_value}
            env.update(self._param_variables(bias.params, None))
        elif bias.kind == "CustomCVTableForce":
            return self._table_lookup(bias, positions)
        else:
            raise NotImplementedError(
                f"bias kind {bias.kind!r} is not defined (bias {bias.label!r})")
        return _evaluate_expression(bias.energy, env, coms)

    # -- table biases (metadynamics) ------------------------------------

    def _table_state(self, bias: BiasIR) -> tuple:
        """(TableSpec, current values ndarray in the reversed-axis layout)
        for one CustomCVTableForce bias."""
        table = bias.table
        key = bias.label or "metadynamics"
        state = self._tables.get(key)
        if state is None:
            shape = tuple(grid.bins for grid in reversed(table.grids))
            values = np.asarray(table.initial, dtype=np.float64).reshape(shape)
            state = (table, values)
            self._tables[key] = state
        return state

    def _cv_report_units(self, cv: CVIR, positions: np.ndarray) -> float:
        """CV value in the grid's natural units (degree for torsion/angle
        CVs; nm or dimensionless otherwise) — matching how colvars grids are
        declared."""
        value = self._cv_expression_value(cv, positions)  # nm / rad / raw
        if cv_is_angular(cv):
            return math.degrees(value)
        return value

    def _table_lookup(self, bias: BiasIR, positions: np.ndarray) -> float:
        """Multilinear table lookup at the current CV point (the fake's own
        interpolation — NOT openmm's; physics parity comes from the openmm
        adapter and the golden tapes, not from the fake)."""
        table, values = self._table_state(bias)
        point = [self._cv_report_units(cv, positions) for cv in table.cvs]
        return _interp_multilinear(values, table.grids, point)

    def bias_ops(self):
        return _FakeBiasOps(self)

    def group_energy(self, groups) -> float:
        """Potential energy (kJ/mol) of the biases on a SET of force groups.

        The per-restraint bias-energy read the restraint reporter needs
        (duck-typed public method, mirroring the openmm adapter's
        ``getState(groups=...)``): the fake evaluates its installed biases
        geometrically, so it can honor the request exactly — the sum of
        ``_bias_energy`` over the biases whose assigned group is in
        ``groups`` (0.0 for an empty selection).
        """
        selected = {int(g) for g in groups}
        return sum(self._bias_energy(bias, self._positions)
                   for group, bias in self._biases if group in selected)

    def _param_variables(self, params: dict, coms: np.ndarray | None) -> dict[str, float]:
        env = {name: self._param_overrides.get(name,
                                               _convert_param(p.value, p.unit))
               for name, p in params.items()}
        if coms is not None:
            env.update(self._com_variables(coms))
        return env

    @staticmethod
    def _com_variables(coms: np.ndarray) -> dict[str, float]:
        # centroid coordinates (x1/y1/z1 variables)
        env: dict[str, float] = {}
        for i, com in enumerate(coms, start=1):
            env[f"x{i}"] = float(com[0])
            env[f"y{i}"] = float(com[1])
            env[f"z{i}"] = float(com[2])
        return env

    def _cv_variables(self, cv: CVIR) -> dict[str, float]:
        return {name: _convert_param(p.value, p.unit)
                for name, p in cv.bond_params.items()}

    def _cv_expression_value(self, cv: CVIR, positions: np.ndarray) -> float:
        """CV value in openmm canonical units (nm / radians / raw
        dimensionless).

        Kind-driven CVs (RMSDForce / coordination / PathCV) take numpy
        special paths — Kabsch RMSD, the coordination pair sum with the
        orthorhombic minimum image on periodic systems, and the path-CV
        log-sum-exp closed forms: the MIRROR of colvars.py's evaluate
        track, PINNED BIT-EXACT against it (and against the openmm
        adapter) by tests; this also makes the rmsd RESTRAINT runnable on
        the fake kernel, whose CustomCVForce("...RMSD...") needs the CV
        value.  Expression-driven CVs go through the restricted
        interpreter.  Units follow openmm's canonicalization: nm, kJ/mol,
        dimensionless, radians — Param(unit="deg") converts to radians.
        """
        if cv.kind == "RMSDForce":
            if cv.ref_positions is None or cv.indices is None:
                raise ValueError(
                    f"cv {cv.label!r}: RMSDForce needs ref_positions and indices")
            sel = list(cv.indices)
            reference = np.asarray(cv.ref_positions, dtype=np.float64)[sel]
            return _kabsch_rmsd(np.asarray(positions, dtype=np.float64)[sel],
                                reference)
        if cv.kind == "CustomNonbondedForce":
            if len(cv.groups) != 2:
                raise ValueError(
                    f"cv {cv.label!r}: CustomNonbondedForce needs 2 groups")
            params = {name: _convert_param(p.value, p.unit)
                      for name, p in cv.bond_params.items()}
            return _coordination_pair_sum(
                positions, cv.groups, params["r0"], params["nn"],
                params["mm"], minimum_image=self._minimum_image)
        if cv.kind == "PathCV":
            if (cv.ref_positions is None or np.asarray(cv.ref_positions).ndim != 3
                    or cv.indices is None or cv.expression not in ("s", "z")):
                raise ValueError(
                    f"cv {cv.label!r}: PathCV needs stacked ref_positions "
                    f"(P, N, 3), indices and expression 's'|'z'")
            sel = list(cv.indices)
            lam = _convert_param(cv.bond_params["lambda"].value,
                                 cv.bond_params["lambda"].unit)
            images = np.asarray(cv.ref_positions, dtype=np.float64)[:, sel, :]
            progress, distance = _path_sz(
                np.asarray(positions, dtype=np.float64)[sel], images, lam)
            return progress if cv.expression == "s" else distance
        if cv.kind == "CustomTorsionForce":
            torsion = cv.torsion
            if torsion is None and len(cv.groups) == 4:
                torsion = tuple(g[0] for g in cv.groups)
            if torsion is None:
                raise ValueError(f"cv {cv.label!r}: CustomTorsionForce needs torsion")
            env = {"theta": _torsion_theta_rad(*[positions[i] for i in torsion])}
            env.update(self._cv_variables(cv))
            return _evaluate_expression(cv.expression, env, None)
        coms = self._centroids(cv.groups, positions) if cv.groups else None
        env = self._cv_variables(cv)
        if coms is not None:
            for i, com in enumerate(coms, start=1):
                env[f"x{i}"] = float(com[0])
                env[f"y{i}"] = float(com[1])
                env[f"z{i}"] = float(com[2])
        return _evaluate_expression(cv.expression, env, coms)

    # ------------------------------------------------------------------
    # snapshots
    # ------------------------------------------------------------------

    def snapshot(self) -> bytes:
        """Pickle of (positions, velocities, step, installed biases, group
        counter, tables, the RandomState state, steered-MD parameter
        overrides, boost channels + last readings) — restoring mid-run
        reproduces the subsequent trajectory BIT-FOR-BIT."""
        payload = {
            "format": "neomd-fake-kernel-v2",
            "positions": self._positions,
            "velocities": self._velocities,
            "step": self._step,
            "biases": list(self._biases),
            "next_group": self._next_group,
            "tables": {k: (t, v.copy()) for k, (t, v) in self._tables.items()},
            "rng_state": self._rng.get_state(),
            "param_overrides": dict(self._param_overrides),
            "boost": {
                "channels": [(label, params["groups"], params["threshold"],
                              params["k"])
                             for label, params in self._boost.items()],
                "last": {label: (r.boost, r.energy, r.scale)
                         for label, r in self._boost_last.items()},
            },
        }
        return pickle.dumps(payload, protocol=4)

    def restore(self, data: bytes) -> None:
        """Restore a :meth:`snapshot` blob (accepts the older v1 payload
        format too); mid-run restores continue the trajectory bit-for-bit."""
        payload = pickle.loads(data)
        if payload.get("format") not in ("neomd-fake-kernel-v1",
                                         "neomd-fake-kernel-v2"):
            raise ValueError("not a FakeKernel snapshot")
        self._positions = np.array(payload["positions"], dtype=np.float64)
        self._velocities = np.array(payload["velocities"], dtype=np.float64)
        self._step = int(payload["step"])
        self._biases = list(payload["biases"])
        self._next_group = int(payload["next_group"])
        self._tables = {k: (t, v.copy())
                        for k, (t, v) in payload.get("tables", {}).items()}
        self._rng.set_state(payload["rng_state"])
        # older "neomd-fake-kernel-v1" payloads carry no overrides
        self._param_overrides = dict(payload.get("param_overrides", {}))
        # payloads without a "boost" section predate GaMD (ADR-0005)
        boost = payload.get("boost") or {}
        self._boost = {
            label: {"groups": tuple(groups), "threshold": float(threshold),
                    "k": float(k)}
            for label, groups, threshold, k in boost.get("channels", [])}
        self._boost_last = {
            label: BoostReading(boost=float(b), energy=float(e), scale=float(s))
            for label, (b, e, s) in boost.get("last", {}).items()}


KernelFactory.register_adapter("fake", FakeKernel)


def _interp_multilinear(values: np.ndarray, grids, point) -> float:
    """Multilinear interpolation on the (reversed-axis) table grid.

    ``values`` is shaped ``tuple(grid.bins for grid in reversed(grids))``.
    Periodic axes wrap; non-periodic axes clamp.
    """
    frac = []
    for grid in grids:  # iterate in logical (cvs) order
        span = grid.maximum - grid.minimum
        x = (point[len(frac)] - grid.minimum) / span
        if grid.periodic:
            x = x % 1.0
        else:
            x = min(max(x, 0.0), 1.0 - 1e-12)
        frac.append(x * (grid.bins - 1))
    # values' axes are reversed(grids): the LAST axis is cv_0, so processing
    # cvs in order always collapses the last axis of the current result
    result = values
    for f in frac:
        i0 = int(np.floor(f))
        i1 = min(i0 + 1, result.shape[-1] - 1)
        w = f - i0
        result = result[..., i0] * (1.0 - w) + result[..., i1] * w
    return float(result)


class _FakeBiasOps:
    """BiasOps over the fake kernel's table biases (see port.BiasOps)."""

    def __init__(self, kernel: "FakeKernel"):
        self._kernel = kernel

    def _bias(self, label: str) -> BiasIR:
        for _, bias in self._kernel._biases:
            if bias.kind == "CustomCVTableForce" and (bias.label or "metadynamics") == label:
                return bias
        raise KeyError(f"no table bias labeled {label!r}")

    def cv_values(self, label: str) -> list[float]:
        bias = self._bias(label)
        pos = self._kernel.positions()
        return [self._kernel._cv_report_units(cv, pos) for cv in bias.table.cvs]

    def bias_energy(self, label: str) -> float:
        bias = self._bias(label)
        return self._kernel._table_lookup(bias, self._kernel.positions())

    def update_table(self, label: str, values) -> None:
        bias = self._bias(label)
        table, _ = self._kernel._table_state(bias)
        shape = tuple(grid.bins for grid in reversed(table.grids))
        self._kernel._tables[label] = (
            table, np.asarray(values, dtype=np.float64).reshape(shape).copy())
