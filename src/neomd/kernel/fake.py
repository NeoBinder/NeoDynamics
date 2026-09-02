"""FakeKernel — the deterministic CI workhorse on the KernelPort seam.

NO openmm import: pure numpy textbook physics.  Per the plan's risk table
(§7, "Fake kernel drift"), the fake deliberately does NOT mimic OpenMM
corner-case behavior — it exists so driver/probe/method tests run fast and
bit-reproducibly; the parity suite and golden tapes guard the real physics.

Physics (documented simplifications):
* dynamics: Euler-Maruyama Langevin on free particles,
  ``v += (F/m)*dt - gamma*v*dt + sqrt(2*gamma*kT/m*dt)*N(0,1)``,
  ``x += v*dt``, with F = 0 (a "resting" force field).  Velocities are
  initialized by textbook Maxwell-Boltzmann: ``N(0,1) * sqrt(kT/m)`` from
  ``numpy.random.RandomState(spec.seed)`` — the same stream continues as the
  Langevin noise source, so trajectories are bit-stable for a given seed
  (float64 arithmetic is deterministic).
* installed biases are evaluated geometrically (the v1 cores:
  calculate_com / angle_3points_rad / calculate_dihedral from
  src/neomd/restraints/reporter.py, ported to numpy below) and their energy
  expression is evaluated by a restricted arithmetic interpreter (the subset
  of the openmm expression language v1 uses: + - * / ^ (=power) ,
  max/min/abs/sqrt/exp/atan/tan/sin/cos, distance()/angle()/dihedral()
  between group centroids g1..g4, xN/yN/zN centroid coordinates,
  ``";"``-separated intermediate assignments).
* forces are always reported as ZERO (the fake does not propagate bias
  forces into the dynamics); kinetic energy comes from its own velocities
  and temperature from 2*KE/(dof*R) with dof = 3N (the openmm
  StateDataReporter convention minus constraints, which the fake has none).
* energy_report().potential = sum of installed bias energies, kJ/mol.
* units inside expressions follow openmm's canonicalization: nm, kJ/mol,
  dimensionless, and radians — Param(unit="deg") is converted to radians
  (verified: openmm converts unit.degree global parameters to radians).
* minimize(): deterministic steepest descent on the geometric bias
  potentials (central finite-difference gradients), and velocities are
  zeroed, as minimization conventionally does.
* periodic boxes: orthorhombic minimum-image convention only (the default
  synthetic system is non-periodic anyway).

Snapshot/restore pickles (positions, velocities, step, installed biases,
group counter, the RandomState state, and the steered-MD parameter
overrides) — restoring mid-run reproduces the subsequent trajectory
bit-for-bit.

Public helpers beyond the port operations (used by driver/probe tests):
``bias_values()`` — geometric value of each installed bias in report units
(distance in nm, angle/dihedral in degrees), matching the v1 reporter and
neomd.colvars evaluate conventions; ``group_energy(groups)`` — per-force-
group bias-energy sum for the restraint reporter.  The fake deliberately
has NO ``write_structure`` (the driver's final-positions artifact seam):
a PDBx writer needs a real topology, so fake-kernel runs skip ``last.pdbx``
(the ``last.ckpt`` snapshot is still written).
"""

from __future__ import annotations

import ast
import math
import pickle

import numpy as np

from .port import (
    BiasIR,
    CVIR,
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
# numpy geometry — ported from v1 restraints/reporter.py
# ----------------------------------------------------------------------

def _com(masses: np.ndarray, positions: np.ndarray, idxlist) -> np.ndarray:
    """v1 reporter.calculate_com (mass-weighted)."""
    idx = np.asarray(idxlist, dtype=int)
    m = masses[idx]
    return (m[:, None] * positions[idx]).sum(axis=0) / m.sum()


def _angle_3points_rad(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """v1 reporter.angle_3points_rad."""
    vec1 = a - b
    vec2 = c - b
    cos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def _dihedral_rad(p1, p2, p3, p4) -> float:
    """v1 reporter.calculate_dihedral, kept in radians."""
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


def _torsion_theta_rad(p1, p2, p3, p4) -> float:
    """The openmm CustomTorsionForce ``theta`` (radians), branch included.

    Same angle as ``_dihedral_rad`` but via the IUPAC atan2 form, which
    agrees with openmm bit-wise including the +/-pi branch of a planar-trans
    torsion (the v1-reporter form returns -pi there, openmm +pi).
    """
    p1, p2, p3, p4 = (np.asarray(p, dtype=np.float64) for p in (p1, p2, p3, p4))
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    return float(np.arctan2(
        np.linalg.norm(b2) * np.dot(b1, np.cross(b2, b3)),
        np.dot(np.cross(b1, b2), np.cross(b2, b3))))


# ----------------------------------------------------------------------
# restricted expression interpreter (the openmm subset v1 emits)
# ----------------------------------------------------------------------

_MATH_FUNCS = {
    "max": max, "min": min, "abs": abs, "sqrt": math.sqrt, "exp": math.exp,
    "atan": math.atan, "tan": math.tan, "sin": math.sin, "cos": math.cos,
}
_GEOMETRY = {"distance", "angle", "dihedral"}


def _evaluate_expression(source: str, variables: dict[str, float],
                         coms: np.ndarray | None = None) -> float:
    """Evaluate an openmm-style custom-force expression in numpy.

    Supports the v1 subset: numbers, names, unary +/-, + - * / and ``^``
    (power), the math functions above, distance()/angle()/dihedral() over
    group centroids g1..gN, and ``";"``-separated intermediate assignments
    (openmm's statement syntax).  Anything else is rejected loudly.

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
    """Deterministic textbook-Langevin KernelPort implementation."""

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
            # F/m = 0 (free particles); textbook Euler-Maruyama update
            self._velocities = (self._velocities * (1.0 - self._gamma * self._dt)
                                + self._noise_scale[:, None] * noise)
            self._positions = self._positions + self._velocities * self._dt
            self._step += 1

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

    def _numerical_gradient(self, positions: np.ndarray, eps: float) -> np.ndarray:
        grad = np.zeros_like(positions)
        flat = positions.reshape(-1)
        gflat = grad.reshape(-1)
        for i in range(flat.size):
            xp, xm = flat.copy(), flat.copy()
            xp[i] += eps
            xm[i] -= eps
            gflat[i] = (self._bias_potential(xp.reshape(-1, 3))
                        - self._bias_potential(xm.reshape(-1, 3))) / (2.0 * eps)
        return grad

    # ------------------------------------------------------------------
    # biases
    # ------------------------------------------------------------------

    def install_bias(self, bias: BiasIR) -> int:
        group = self._pick_force_group()
        self._biases.append((group, bias))
        self._next_group += 1  # install counter (snapshot-format field)
        return group

    def _pick_force_group(self) -> int:
        """The shared port policy (pick_free_force_group), aligned with the
        openmm adapter (improvements item 5): max free id first — 31, 30, …
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

    def bias_values(self, positions: np.ndarray | None = None) -> dict[str, float]:
        """Geometric value of each installed bias in report units.

        distance in nm, angle/dihedral in degrees (v1 reporter and
        neomd.colvars conventions); keys are bias labels (falling back to
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
        """Report-unit value of one CVIR (degrees for angles, else nm)."""
        if cv.kind == "CustomTorsionForce":
            torsion = cv.torsion
            if torsion is None and len(cv.groups) == 4:
                torsion = tuple(g[0] for g in cv.groups)
            if torsion is None:
                raise ValueError(f"cv {cv.label!r}: CustomTorsionForce needs torsion")
            return math.degrees(_dihedral_rad(*[positions[i] for i in torsion]))
        coms = self._centroids(cv.groups, positions) if cv.groups else None
        env = self._cv_variables(cv)
        value = _evaluate_expression(cv.expression, env, coms)
        # distance-type CVs are nm; angular CVs come back in radians
        if cv_is_angular(cv):
            return math.degrees(value)
        return value

    # -- energy evaluation ---------------------------------------------

    def _bias_potential(self, positions: np.ndarray) -> float:
        return sum(self._bias_energy(bias, positions) for _, bias in self._biases)

    def _bias_energy(self, bias: BiasIR, positions: np.ndarray) -> float:
        coms: np.ndarray | None = None
        if bias.kind == "CustomCentroidBondForce":
            if bias.bonds is not None:
                # multi-bond mode (v1 179ae35 distances): one force, N bonds,
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
        """(TableSpec, current values ndarray shaped like v1's reversed-axis
        convention) for one CustomCVTableForce bias."""
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
        CVs, nm otherwise) — matching how colvars grids are declared."""
        value = self._cv_expression_value(cv, positions)  # nm / radians
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
        # centroid coordinates (v1 xyz_box-style x1/y1/z1)
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
        """CV value in openmm canonical units (nm / radians) for expressions."""
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
        }
        return pickle.dumps(payload, protocol=4)

    def restore(self, data: bytes) -> None:
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
        # v1 payloads predate steered MD: no overrides is the correct state
        self._param_overrides = dict(payload.get("param_overrides", {}))


KernelFactory.register_adapter("fake", FakeKernel)


def _interp_multilinear(values: np.ndarray, grids, point) -> float:
    """Multilinear interpolation on the (reversed-axis) table grid.

    ``values`` is shaped ``tuple(grid.bins for grid in reversed(grids))``
    (v1's convention).  Periodic axes wrap; non-periodic axes clamp.
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
