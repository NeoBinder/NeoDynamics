#!/usr/bin/env python
"""Inspect prepared force-field parameters for user-selected atoms.

Reads a prepared topology (``pdb``/``pdbx``) and the serialized OpenMM
System (``system.xml`` written by ``neomd prepare``), selects atoms by
keyword combinations, and prints the resulting force-field parameters as
tables.  Usable both as a shell command and as a Python function.

Shell example::

    python bin/inspect_ff_params.py \
        --topology solv.pdbx --system system.xml \
        --select "chain=A;resname=ALA;resid=10-20" \
        --select "distance=5:(chain=B;resname=JZ4)" \
        --params bonds,angles,dihedrals,nonbonded,mass \
        --mode any

Python example::

    from inspect_ff_params import inspect_params
    tables = inspect_params(
        topology_file="solv.pdbx",
        system_file="system.xml",
        selects=["chain=A;resname=ALA", "distance=5:(resname=JZ4)"],
        params=["bonds", "nonbonded"],
        mode="any",          # "any": term shown if it touches >=1 selected
                             # atom; "all": only when every atom of the term
                             # is selected
    )

Selection language (inside one ``--select`` group; several groups are OR-ed):

    ``;``            separates keyword terms, all terms are AND-ed
    ``,``            lists several values of ONE keyword
    ``not (...)``    negates a parenthesised sub-selection
    ``distance=C:(...)``
                     atoms within C angstrom (suffix "nm" for nanometers)
                     of every atom matched by the parenthesised selection;
                     the reference atoms themselves are NOT included unless
                     they also match the other terms of the group — OR them
                     in with a second ``--select`` group

    chain=A;element=C,N,O;distance=5:(resname=JZ4);not (name=H,HA)
        = chain A AND element in {C, N, O} AND within 5 A of JZ4
          AND NOT named H or HA

    com0;distance=5:(com1);com2   = com0 AND (within 5 of com1) AND com2
    com1;not (com2;com3);com4     = com1 AND !(com2 AND com3) AND com4

Keywords:

    chain     chain id list:                chain=A,B
    resid     residue id (PDB id):          resid=100-110,215
    resname   residue name list:            resname=ALA,GLY
    name      atom name list:               name=CA,CB
    index     atom index (0-based):         index=0-50,128
    resindex  residue index (0-based):      resindex=5-9
    element   element symbol list:          element=C,N,O

Parameter types are discovered from the forces actually present in the
system file, so different ``system.xml`` files may offer different types.
Known types: bonds, angles, dihedrals, nonbonded, mass.  Requesting a type
that is not available prints "<type> is not in the available parameter
list, skipped." and continues.
"""

import argparse
import csv
import sys

import numpy as np
import pandas as pd
from openmm import XmlSerializer, unit
from openmm.app import PDBFile, PDBxFile

# ---------------------------------------------------------------------------
# atom table
# ---------------------------------------------------------------------------


def _element_symbol(atom):
    return atom.element.symbol if atom.element is not None else ""


def build_atom_table(topology):
    """One row per atom: index, name, element, resname, resid, resindex, chain."""
    rows = []
    for atom in topology.atoms():
        res = atom.residue
        rows.append(
            {
                "index": atom.index,
                "name": atom.name,
                "element": _element_symbol(atom),
                "resname": res.name,
                "resid": str(res.id),
                "resindex": res.index,
                "chain": res.chain.id,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

_SELECT_KEYS = ("chain", "resid", "resname", "name", "index", "resindex",
                "element")


def _parse_tokens(value):
    return [tok.strip() for tok in value.split(",") if tok.strip()]


def _match_numeric_tokens(tokens, values):
    """tokens like ['10-20', '35']; values numeric. Returns boolean mask."""
    mask = np.zeros(len(values), dtype=bool)
    for tok in tokens:
        if "-" in tok:
            lo_s, hi_s = tok.split("-", 1)
            lo, hi = float(lo_s), float(hi_s)
            mask |= (values >= lo) & (values <= hi)
        else:
            mask |= values == float(tok)
    return mask


def _match_string_tokens(tokens, values):
    mask = np.zeros(len(values), dtype=bool)
    for tok in tokens:
        mask |= values == tok
    return mask


class _SelectParser:
    """Recursive-descent parser for the selection mini-language.

    Grammar::

        group    := term (";" term)*            # ";" = AND
        term     := "not" "(" group ")"         # negation
                  | "distance" "=" CUTOFF ":(" group ")"
                  | KEY "=" value ("," value)*  # "," lists values of one KEY
        CUTOFF   := number, angstrom by default, "nm" suffix for nanometers

    Example:  ``chain=A;element=C,N,O;distance=5:(resname=JZ4);not (name=H)``
    """

    def __init__(self, text):
        self.text = text
        self.i = 0

    def _ws(self):
        while self.i < len(self.text) and self.text[self.i] in " \t":
            self.i += 1

    def _expect(self, char):
        self._ws()
        if self.i >= len(self.text) or self.text[self.i] != char:
            raise ValueError(
                f"expected {char!r} at position {self.i} in {self.text!r}"
            )
        self.i += 1

    def parse_group(self):
        terms = [self.parse_term()]
        while True:
            self._ws()
            if self.i < len(self.text) and self.text[self.i] == ";":
                self.i += 1
                terms.append(self.parse_term())
            else:
                break
        return ("and", terms)

    def parse_term(self):
        self._ws()
        if self.text.startswith("not", self.i) and (
            self.i + 3 >= len(self.text) or self.text[self.i + 3] in " \t("
        ):
            self.i += 3
            self._expect("(")
            node = self.parse_group()
            self._expect(")")
            return ("not", node)

        eq = self.text.find("=", self.i)
        if eq == -1:
            raise ValueError(
                f"expected key=value at position {self.i} in {self.text!r}"
            )
        key = self.text[self.i:eq].strip().lower()
        self.i = eq + 1
        if key == "atname":
            key = "name"

        if key == "distance":
            self._ws()
            colon = self.text.find(":(", self.i)
            if colon == -1:
                raise ValueError(
                    "distance needs a parenthesised reference selection: "
                    "'distance=5:(resname=JZ4)'"
                )
            cutoff_s = self.text[self.i:colon].strip()
            self.i = colon + 1  # at '('
            if cutoff_s.endswith("nm"):
                cutoff_nm = float(cutoff_s[:-2])
            else:  # angstrom by default
                cutoff_nm = float(cutoff_s.rstrip("Aa")) / 10.0
            self._expect("(")
            ref = self.parse_group()
            self._expect(")")
            return ("distance", cutoff_nm, ref)

        if key not in _SELECT_KEYS:
            raise ValueError(
                f"unknown selection keyword {key!r}; "
                f"supported: {', '.join(_SELECT_KEYS)}, distance, not"
            )
        start = self.i
        while self.i < len(self.text) and self.text[self.i] not in ";)":
            self.i += 1
        tokens = _parse_tokens(self.text[start:self.i])
        if not tokens:
            raise ValueError(f"keyword {key!r} needs at least one value")
        for tok in tokens:
            if "=" in tok:
                raise ValueError(
                    f"value {tok!r} contains '='; use ';' to separate "
                    f"keywords (',' only lists values of one keyword)"
                )
        return ("key", key, tokens)


def parse_select_string(spec):
    """Parse a selection string into an AST (see :class:`_SelectParser`)."""
    parser = _SelectParser(spec)
    node = parser.parse_group()
    parser._ws()
    if parser.i != len(parser.text):
        raise ValueError(
            f"unexpected trailing text {parser.text[parser.i:]!r} "
            f"in selection {spec!r}"
        )
    return node


def _key_mask(atom_df, key, tokens):
    if key in ("index", "resindex", "resid"):
        values = atom_df[key].to_numpy()
        try:
            return _match_numeric_tokens(tokens, values.astype(float))
        except (ValueError, TypeError):
            return _match_string_tokens(tokens, values.astype(str))
    return _match_string_tokens(tokens, atom_df[key].astype(str).to_numpy())


def eval_select(node, atom_df, positions_nm):
    """Evaluate a parsed selection AST to a boolean mask over atoms."""
    kind = node[0]
    if kind == "and":
        mask = np.ones(len(atom_df), dtype=bool)
        for term in node[1]:
            mask &= eval_select(term, atom_df, positions_nm)
        return mask
    if kind == "not":
        return ~eval_select(node[1], atom_df, positions_nm)
    if kind == "distance":
        _, cutoff_nm, ref = node
        ref_mask = eval_select(ref, atom_df, positions_nm)
        if not ref_mask.any():
            raise ValueError("distance reference selection matched no atoms")
        ref_pos = positions_nm[ref_mask]
        # distance from every atom to its nearest reference atom
        d2 = ((positions_nm[:, None, :] - ref_pos[None, :, :]) ** 2).sum(-1)
        return d2.min(axis=1) <= cutoff_nm ** 2
    if kind == "key":
        return _key_mask(atom_df, node[1], node[2])
    raise ValueError(f"unknown selection node {kind!r}")


def selection_mask(atom_df, positions_nm, spec):
    """Boolean mask over atoms for one selection (string or parsed AST)."""
    if isinstance(spec, str):
        spec = parse_select_string(spec)
    return eval_select(spec, atom_df, positions_nm)


def combined_mask(atom_df, positions_nm, selects):
    """Union (OR) of the per-group AND masks."""
    mask = np.zeros(len(atom_df), dtype=bool)
    for spec in selects:
        mask |= selection_mask(atom_df, positions_nm, spec)
    return mask


# ---------------------------------------------------------------------------
# parameter tables
# ---------------------------------------------------------------------------


def _atom_label(atom_df, index):
    row = atom_df.iloc[index]
    return f"{row['chain']}:{row['resname']}:{row['resid']}:{row['name']}"


def _term_mask(idxs_per_term, selected_set, mode):
    out = []
    for idxs in idxs_per_term:
        hits = sum(i in selected_set for i in idxs)
        out.append(hits == len(idxs) if mode == "all" else hits > 0)
    return np.array(out, dtype=bool)


def _get_force(system, class_name):
    for force in system.getForces():
        if force.__class__.__name__ == class_name:
            return force
    return None


def parse_bonds(atom_df, system, selected, mode):
    force = _get_force(system, "HarmonicBondForce")
    terms, params = [], []
    for i in range(force.getNumBonds()):
        p1, p2, length, k = force.getBondParameters(i)
        terms.append((p1, p2))
        params.append(
            {
                "length (nm)": length.value_in_unit(unit.nanometer),
                "k (kJ/mol/nm^2)": k.value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer**2
                ),
            }
        )
    keep = _term_mask(terms, selected, mode)
    rows = []
    for (p1, p2), prm, ok in zip(terms, params, keep):
        if not ok:
            continue
        rows.append(
            {
                "index1": p1,
                "atom1": _atom_label(atom_df, p1),
                "index2": p2,
                "atom2": _atom_label(atom_df, p2),
                **prm,
            }
        )
    return pd.DataFrame(rows)


def parse_angles(atom_df, system, selected, mode):
    force = _get_force(system, "HarmonicAngleForce")
    terms, params = [], []
    for i in range(force.getNumAngles()):
        p1, p2, p3, angle, k = force.getAngleParameters(i)
        terms.append((p1, p2, p3))
        params.append(
            {
                "angle (deg)": angle.value_in_unit(unit.degree),
                "k (kJ/mol/rad^2)": k.value_in_unit(
                    unit.kilojoule_per_mole / unit.radian**2
                ),
            }
        )
    keep = _term_mask(terms, selected, mode)
    rows = []
    for (p1, p2, p3), prm, ok in zip(terms, params, keep):
        if not ok:
            continue
        rows.append(
            {
                "index1": p1,
                "atom1": _atom_label(atom_df, p1),
                "index2": p2,
                "atom2": _atom_label(atom_df, p2),
                "index3": p3,
                "atom3": _atom_label(atom_df, p3),
                **prm,
            }
        )
    return pd.DataFrame(rows)


def parse_dihedrals(atom_df, system, selected, mode):
    force = _get_force(system, "PeriodicTorsionForce")
    terms, params = [], []
    for i in range(force.getNumTorsions()):
        p1, p2, p3, p4, periodicity, phase, k = force.getTorsionParameters(i)
        terms.append((p1, p2, p3, p4))
        params.append(
            {
                "periodicity": periodicity,
                "phase (deg)": phase.value_in_unit(unit.degree),
                "k (kJ/mol)": k.value_in_unit(unit.kilojoule_per_mole),
            }
        )
    keep = _term_mask(terms, selected, mode)
    rows = []
    for (p1, p2, p3, p4), prm, ok in zip(terms, params, keep):
        if not ok:
            continue
        rows.append(
            {
                "index1": p1,
                "atom1": _atom_label(atom_df, p1),
                "index2": p2,
                "atom2": _atom_label(atom_df, p2),
                "index3": p3,
                "atom3": _atom_label(atom_df, p3),
                "index4": p4,
                "atom4": _atom_label(atom_df, p4),
                **prm,
            }
        )
    return pd.DataFrame(rows)


def parse_nonbonded(atom_df, system, selected, mode):
    force = _get_force(system, "NonbondedForce")
    rows = []
    for i in sorted(selected):
        chg, sigma, eps = force.getParticleParameters(i)
        row = atom_df.iloc[i]
        rows.append(
            {
                "index": i,
                "atom": _atom_label(atom_df, i),
                "element": row["element"],
                "charge (e)": chg.value_in_unit(unit.elementary_charge),
                "sigma (nm)": sigma.value_in_unit(unit.nanometer),
                "epsilon (kJ/mol)": eps.value_in_unit(unit.kilojoule_per_mole),
            }
        )
    return pd.DataFrame(rows)


def parse_mass(atom_df, system, selected, mode):
    rows = []
    for i in sorted(selected):
        mass = system.getParticleMass(i)
        row = atom_df.iloc[i]
        rows.append(
            {
                "index": i,
                "atom": _atom_label(atom_df, i),
                "element": row["element"],
                "mass (amu)": mass.value_in_unit(unit.dalton),
            }
        )
    return pd.DataFrame(rows)


# param type -> (force class providing it or None, parser)
PARAM_PARSERS = {
    "bonds": ("HarmonicBondForce", parse_bonds),
    "angles": ("HarmonicAngleForce", parse_angles),
    "dihedrals": ("PeriodicTorsionForce", parse_dihedrals),
    "nonbonded": ("NonbondedForce", parse_nonbonded),
    "mass": (None, parse_mass),
}


def available_param_types(system):
    out = []
    for name, (force_class, _) in PARAM_PARSERS.items():
        if force_class is None or _get_force(system, force_class) is not None:
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------


def load_topology(topology_file):
    if topology_file.endswith(".pdb"):
        pdb = PDBFile(topology_file)
    elif topology_file.endswith((".pdbx", ".cif")):
        pdb = PDBxFile(topology_file)
    else:
        raise ValueError(f"unsupported topology format: {topology_file}")
    return pdb.topology, pdb.getPositions(asNumpy=True).value_in_unit(
        unit.nanometer
    )


def load_system(system_file):
    """Deserialize an OpenMM System XML (content may use any file extension)."""
    with open(system_file) as f:
        text = f.read()
    if not text.lstrip().startswith("<"):
        raise ValueError(
            f"{system_file} is not a serialized OpenMM System XML "
            "(expected the system.xml written by `neomd prepare`)"
        )
    return XmlSerializer.deserialize(text)


def inspect_params(
    topology_file,
    system_file,
    selects=None,
    params=None,
    mode="any",
    out_csv=None,
):
    """Select atoms and print their force-field parameter tables.

    Parameters
    ----------
    topology_file : str
        pdb/pdbx of the prepared system.
    system_file : str
        serialized OpenMM System (system.xml from ``neomd prepare``).
    selects : list[str]
        Selection groups; terms are AND-ed with ``;`` inside a group,
        groups are OR-ed.  None or empty selects ALL atoms.
    params : list[str]
        Parameter types; None or ['all'] means every available type.
    mode : str
        "any": a bonded term is shown when it contains >=1 selected atom;
        "all": only when every atom of the term is selected.
    out_csv : str | None
        Write each table to CSV (``<stem>_<type><suffix>`` when several
        types are requested, else exactly this path).

    Returns
    -------
    dict[str, pandas.DataFrame]
    """
    if mode not in ("any", "all"):
        raise ValueError("mode must be 'any' or 'all'")
    topology, positions_nm = load_topology(topology_file)
    atom_df = build_atom_table(topology)
    system = load_system(system_file)

    if selects:
        mask = combined_mask(atom_df, positions_nm, selects)
    else:
        mask = np.ones(len(atom_df), dtype=bool)
    selected = set(atom_df["index"].to_numpy()[mask].tolist())

    n_chains = atom_df.loc[mask, "chain"].nunique()
    n_res = atom_df.loc[mask, ["chain", "resid", "resindex"]].drop_duplicates().shape[0]
    print(
        f"Selected {n_chains} chain(s), {n_res} residue(s), "
        f"{int(mask.sum())} atom(s)."
    )
    if not selected:
        print("WARNING: selection is empty; nothing to show.", file=sys.stderr)
        return {}

    avail = available_param_types(system)
    print(f"Available parameter types: {', '.join(avail)}")

    if not params or params == ["all"]:
        wanted = list(avail)
    else:
        wanted = []
        for p in params:
            p = p.strip().lower()
            if p == "all":
                wanted.extend(t for t in avail if t not in wanted)
            elif p in avail:
                if p not in wanted:
                    wanted.append(p)
            else:
                print(f"{p} is not in the available parameter list, skipped.")

    tables = {}
    for p in wanted:
        _, parser = PARAM_PARSERS[p]
        df = parser(atom_df, system, selected, mode)
        tables[p] = df
        print(f"\n=== {p} ({len(df)} row(s), mode={mode}) ===")
        print(df.to_string(index=False) if len(df) else "(none)")

    if out_csv:
        for p, df in tables.items():
            path = out_csv
            if len(tables) > 1:
                if "." in out_csv.rsplit("/", 1)[-1]:
                    stem, _, suffix = out_csv.rpartition(".")
                    path = f"{stem}_{p}.{suffix}"
                else:
                    path = f"{out_csv}_{p}.csv"
            df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
            print(f"wrote {path}")
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect prepared force-field parameters for selected atoms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--topology", "-t", required=True,
                        help="prepared topology file (pdb/pdbx)")
    parser.add_argument("--system", "-s", required=True,
                        help="serialized OpenMM System (system.xml from `neomd prepare`)")
    parser.add_argument("--select", action="append", default=[],
                        help="selection group, e.g. 'chain=A;resname=ALA;resid=10-20' "
                             "or 'distance=5:(resname=JZ4)'; repeatable (groups are OR-ed)")
    parser.add_argument("--params", "-p", default="all",
                        help="comma-separated parameter types or 'all' (default)")
    parser.add_argument("--mode", "-m", choices=["any", "all"], default="any",
                        help="'any': show terms touching >=1 selected atom; "
                             "'all': only terms fully inside the selection")
    parser.add_argument("--out-csv", default=None,
                        help="also write tables to CSV")
    args = parser.parse_args(argv)

    inspect_params(
        topology_file=args.topology,
        system_file=args.system,
        selects=args.select or None,
        params=[p for p in args.params.split(",") if p.strip()],
        mode=args.mode,
        out_csv=args.out_csv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
