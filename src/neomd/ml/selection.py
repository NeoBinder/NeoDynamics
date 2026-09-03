"""selection — the ml_region residue-selector grammar (ADR-0004, W3-c).

Active-site ML regions are spelled as RESIDUE SELECTORS instead of raw
particle indices, so a plan can say "the ligand plus residue 29 of chain A"
without a topology-derived index list that rots on every re-preparation::

    ml_region:
      residues: ["A:JZ4", "A:29", "A:31"]   # ligand + two pocket residues
      model: {...}

THE GRAMMAR (deliberately minimal — one spelling per intent):

* ``"CHAIN:RESID"``  — the tail is NUMERIC: every atom of every residue whose
  ``id`` equals the tail in that chain (e.g. ``"A:29"``).  Residue ids are
  the topology's own (PDB author numbering, kept by ``keepIds=True``).
* ``"CHAIN:NAME"``   — the tail is NON-numeric: every atom of every residue
  NAMED tail in that chain (e.g. ``"A:JZ4"``, the ligand-by-resname
  spelling; ``"A:HOH"`` would take every water of the chain — the
  per-residue id spelling is the surgical one).
* chain id, residue id and residue name compare CASE-INSENSITIVELY (PDB
  resnames are upper-case; ``"a:jz4"`` == ``"A:JZ4"``).

Selection is resolved against the system's TOPOLOGY (the plan's
``input_files.complex`` — the same file the openmm adapter loads): at the
``neomd validate --check-files`` tier for early collect-all feedback, and at
adapter assembly time as the definitive resolution (a hand-built
``KernelSpec`` gets the same treatment — the defensive-second-gate rule of
``ml.spec``).  A selector matching NOTHING is an error with a did-you-mean
over the topology's chains / that chain's residue names.

This module is openmm-free: it ducks-types the topology
(``topology.atoms()`` with ``atom.index`` / ``atom.residue.name`` /
``atom.residue.id`` / ``atom.residue.chain.id`` — exactly the
``openmm.app.Topology`` surface), so validation, adapters and tests all
consume one definition.
"""

from __future__ import annotations

import re

from ..errors import ConfigValueError, suggest

__all__ = ["RESIDUE_SELECTOR_RE", "is_residue_selector", "parse_residue_selector",
           "match_residue_selector", "selector_suggestions",
           "unmatched_selector_error", "resolve_residues"]

#: one "CHAIN:tail" token: exactly one colon, no whitespace on either side
RESIDUE_SELECTOR_RE = re.compile(r"^[^:\s]+:[^:\s]+$")


def is_residue_selector(value) -> bool:
    """Structural check only (no topology): does ``value`` read like one?"""
    return isinstance(value, str) and bool(RESIDUE_SELECTOR_RE.match(value))


def parse_residue_selector(selector: str) -> tuple[str, str, str]:
    """``"A:29"`` -> ``("A", "29", "id")``; ``"A:JZ4"`` -> ``("A", "JZ4", "name")``.

    Raises :class:`~neomd.errors.ConfigValueError` (key path ``residues``)
    on anything that is not one colon-separated token — the same error the
    structural plan-validation pass reports.
    """
    if not isinstance(selector, str) or not RESIDUE_SELECTOR_RE.match(selector):
        raise ConfigValueError(
            f"ml_region.residues entries must be 'CHAIN:RESID' (numeric tail, "
            f"e.g. 'A:29') or 'CHAIN:NAME' (e.g. 'A:JZ4') selectors, got "
            f"{selector!r}",
            key="residues", value=selector)
    chain, tail = selector.split(":", 1)
    kind = "id" if tail.isdigit() else "name"
    return chain, tail, kind


def _residue_matches(residue, chain: str, tail: str, kind: str) -> bool:
    attribute = residue.id if kind == "id" else residue.name
    return (str(residue.chain.id).upper() == chain.upper()
            and str(attribute).upper() == tail.upper())


def match_residue_selector(selector: str, topology) -> list[int]:
    """One well-formed selector -> the atom indices it selects (may be [])."""
    chain, tail, kind = parse_residue_selector(selector)
    return [atom.index for atom in topology.atoms()
            if _residue_matches(atom.residue, chain, tail, kind)]


def _topology_chains(topology) -> list[str]:
    """Distinct chain ids, first-seen order (the residue() iteration order)."""
    seen: dict[str, None] = {}
    for atom in topology.atoms():
        seen.setdefault(str(atom.residue.chain.id).upper(), None)
    return list(seen)


def _chain_residue_names(topology, chain: str) -> list[str]:
    """Distinct residue names of one chain, first-seen order."""
    seen: dict[str, None] = {}
    for atom in topology.atoms():
        residue = atom.residue
        if str(residue.chain.id).upper() == chain.upper():
            seen.setdefault(str(residue.name).upper(), None)
    return list(seen)


def selector_suggestions(selector: str, topology) -> list[str]:
    """Did-you-mean entries for an unmatched selector, topology-aware.

    Unknown chain -> the topology's chain ids; known chain, bad tail -> that
    chain's residue names (bounded: amino acids + ligand + solvent names —
    never per-residue ids, which would be thousands of entries for water).
    """
    chain, tail, _kind = parse_residue_selector(selector)
    chains = _topology_chains(topology)
    if chain.upper() not in chains:
        return suggest(chain, chains) or chains[:4]
    names = _chain_residue_names(topology, chain)
    return suggest(tail, names, cutoff=0.5) or names[:4]


def _chain_residue_ids(topology, chain: str) -> list[str]:
    """Every residue id of one chain, first-seen order (may be long)."""
    ids: list[str] = []
    for atom in topology.atoms():
        residue = atom.residue
        if str(residue.chain.id).upper() == chain.upper():
            residue_id = str(residue.id)
            if not ids or ids[-1] != residue_id:
                ids.append(residue_id)
    return ids


def unmatched_selector_error(selector: str, topology,
                             source: str | None = None) -> ConfigValueError:
    """The error an unmatched selector raises/reports (one definition point).

    Used by :func:`resolve_residues` (first problem raises — the defensive
    boundary gate) and by the ``--check-files`` tier (every unmatched
    selector appended to the collect-all list, with ``source`` provenance).
    """
    chain, tail, kind = parse_residue_selector(selector)
    chains = _topology_chains(topology)
    if chain.upper() not in chains:
        return ConfigValueError(
            f"ml_region.residues selector {selector!r} matches nothing: the "
            f"topology has no chain {chain!r} (chains: "
            f"{', '.join(chains)})",
            key="residues", value=selector, source=source,
            candidates=selector_suggestions(selector, topology))
    if kind == "id":
        ids = _chain_residue_ids(topology, chain)
        numeric = sorted(int(i) for i in ids if i.strip().lstrip("-").isdigit())
        span = (f"{numeric[0]}..{numeric[-1]}" if len(numeric) > 8
                else ", ".join(ids))
        return ConfigValueError(
            f"ml_region.residues selector {selector!r} matches nothing: chain "
            f"{chain!r} has no residue id {tail!r} (residue ids: {span})",
            key="residues", value=selector, source=source)
    names = _chain_residue_names(topology, chain)
    return ConfigValueError(
        f"ml_region.residues selector {selector!r} matches nothing: chain "
        f"{chain!r} has no residue named {tail!r} (residue names: "
        f"{', '.join(names)})",
        key="residues", value=selector, source=source,
        candidates=selector_suggestions(selector, topology))


def resolve_residues(selectors, topology) -> list[int]:
    """Selector list -> the sorted unique atom-index list of the ML region.

    Raises on the FIRST unmatched selector (this is the adapter-boundary
    defensive gate — the collect-all reporting of every unmatched selector
    belongs to the ``--check-files`` tier, which walks selectors itself).
    """
    indices: set[int] = set()
    for selector in selectors:
        matched = match_residue_selector(selector, topology)
        if not matched:
            raise unmatched_selector_error(selector, topology)
        indices.update(matched)
    if not indices:  # pragma: no cover - an empty selectors list is refused earlier
        raise ConfigValueError(
            "ml_region.residues must be a non-empty list of selectors",
            key="residues", value=selectors)
    return sorted(indices)
