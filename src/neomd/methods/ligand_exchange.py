"""Deterministic ligand-slot exchange, with a checkpoint-trimmable identity ledger.

This is the earlier LiGaMD selection procedure extended to configured GaMD
partitions, not a Metropolis move. It preserves the validated base force-field
and kinetic energies but can change user restraints and boosts. Equilibrium CE2
qualification is a separate requirement.
"""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np

from neomd.kernel.port import LigandExchangeOps, provides

LEDGER_FILENAME = "ligand_exchange.json"
EVENTS_FILENAME = "ligand_exchange_events.jsonl"


def minimum_distance(positions, ligand_atoms, reference_atoms, box=None):
    """Minimum atom-pair distance (nm), using reduced periodic-cell images."""
    positions = np.asarray(positions, dtype=float)
    delta = positions[np.asarray(ligand_atoms)][:, None] - positions[np.asarray(reference_atoms)][None, :]
    if box is not None:
        box = np.asarray(box, dtype=float)
        if box.shape != (3, 3) or not np.all(np.isfinite(box)) or np.linalg.det(box) <= 0:
            raise ValueError("exchange requires finite right-handed periodic box vectors")
        fractional = delta @ np.linalg.inv(box)
        fractional -= np.rint(fractional)
        # The extra images matter for skewed reduced triclinic cells.
        images = np.asarray(list(itertools.product((-1, 0, 1), repeat=3)))
        delta = (fractional[..., None, :] + images) @ box
    squared = np.sum(delta * delta, axis=-1)
    return float(np.sqrt(np.min(squared)))


class LigandExchange:
    """Method-owned selection and ledger; the kernel owns atomic permutation.

    With a cutoff, retain the designated copy while it is at or below threshold.
    Only when it is outside and another copy is inside, swap the closest eligible
    other copy. The default check cadence is every integration step. With no
    cutoff, all slots enter the original argmin. Equal distances choose the first
    configured eligible slot. No energy or Metropolis criterion vetoes a swap.

    Hooks execute before reporters and checkpoint writers. A completed event
    with step <= a restored checkpoint is already applied; later events are
    discarded. A pending event at the checkpoint step is rejected, since it
    contradicts that ordering rather than guessed to be applied or not.
    """

    def __init__(self, kernel, settings, sink=None, *, resumed=False, clock=None):
        self.kernel, self.sink = kernel, sink
        #: journal clock: maps a leg-relative integration step onto the
        #: monotone axis the event ledger is ordered by.  GaMD runs LEGS whose
        #: step axes restart at 0 (ADR-0011), so the default identity would
        #: make event order non-monotone; gamd passes its virtual clock.
        self.clock = (lambda step: int(step)) if clock is None else clock
        self.settings = dict(settings)
        allowed = {"copies", "designated_slot", "reference_atoms", "ligand_reference_atoms",
                   "interval", "cutoff_nm", "periodic"}
        if set(settings) - allowed:
            raise ValueError(f"unknown gamd.exchange keys: {sorted(set(settings) - allowed)}")
        self.copies = tuple(tuple(c) for c in settings["copies"])
        self.slot = settings.get("designated_slot", 0)
        self.interval = settings.get("interval", 1)
        self.reference_atoms = tuple(settings["reference_atoms"])
        self.offsets = tuple(settings.get("ligand_reference_atoms", range(len(self.copies[0])))) if self.copies else ()
        self.periodic = settings.get("periodic", True)
        self.cutoff = settings.get("cutoff_nm")
        self._validate()
        if not provides(kernel, LigandExchangeOps):
            raise NotImplementedError("ligand exchange requires LigandExchangeOps")
        kernel.validate_ligand_exchange(self.copies)
        self.slot_to_identity = list(range(len(self.copies)))
        self.event_count = 0
        self.last_event_step = None
        self.last_check = None
        self._last_checked_step = -1
        self.pending = None
        self.selection_policy = ("all_slot_argmin" if self.cutoff is None else
                                 "retain_inside_else_nearest_eligible")
        self.fingerprint = hashlib.sha256(json.dumps({
            "settings": self.settings, "selection_policy": self.selection_policy,
        }, sort_keys=True).encode()).hexdigest()
        if resumed:
            if sink is None or not sink.exists(LEDGER_FILENAME):
                raise FileNotFoundError("exchange resume requires ligand_exchange.json")
            saved = json.loads(sink.read_bytes(LEDGER_FILENAME))
            if saved.get("schema_version") != 2 or saved["fingerprint"] != self.fingerprint:
                raise ValueError("ligand exchange configuration changed since checkpoint")
            pending = saved.get("pending")
            if pending is not None and pending["step"] <= self.clock(kernel.current_step):
                raise ValueError("checkpoint intersects an incomplete ligand exchange transaction")
            from neomd.resume import trim_exchange_events
            trim_exchange_events(sink, self.clock(kernel.current_step))
            for event in self.events:
                permutation = event["permutation"]
                if (not event.get("changed") or sorted(permutation) != list(range(len(self.copies)))
                        or permutation == list(range(len(self.copies)))):
                    raise ValueError("invalid completed ligand exchange")
                self.slot_to_identity = [self.slot_to_identity[i] for i in permutation]
                if event["slot_to_identity"] != self.slot_to_identity:
                    raise ValueError("ligand exchange identity disagrees with journal")
                self.event_count += 1
                self.last_event_step = event["step"]
            saved_last = saved.get("last_event_step")
            if saved_last is None or saved_last <= self.clock(kernel.current_step):
                if (saved.get("event_count") != self.event_count
                        or saved_last != self.last_event_step
                        or saved.get("slot_to_identity") != self.slot_to_identity):
                    raise ValueError("ligand exchange journal is missing committed history")
            self._last_checked_step = self.clock(kernel.current_step)
        elif sink is not None:
            sink.write_bytes(EVENTS_FILENAME, b"")
        self.persist()

    @property
    def events(self):
        """Persisted mutations loaded on demand; runs without a sink keep no history."""
        if self.sink is None:
            return []
        return [json.loads(line) for line in self.sink.read_bytes(EVENTS_FILENAME).splitlines()]

    def _validate(self):
        def integer(value):
            return isinstance(value, int) and not isinstance(value, bool)

        if not integer(self.interval) or self.interval < 1:
            raise ValueError("gamd.exchange.interval must be a positive integer")
        if len(self.copies) < 2 or not self.copies[0] or len({len(c) for c in self.copies}) != 1:
            raise ValueError("exchange needs at least two equal-sized nonempty aligned copies")
        atoms = [i for c in self.copies for i in c]
        if any(not integer(i) or i < 0 or i >= self.kernel.num_particles for i in atoms):
            raise ValueError("invalid ligand atom index")
        if len(set(atoms)) != len(atoms):
            raise ValueError("ligand copies must be disjoint without duplicate atoms")
        if not integer(self.slot) or not 0 <= self.slot < len(self.copies):
            raise ValueError("invalid designated ligand slot")
        if not self.reference_atoms or any(not integer(i) or i < 0 or i >= self.kernel.num_particles for i in self.reference_atoms):
            raise ValueError("invalid exchange reference_atoms")
        if set(atoms) & set(self.reference_atoms):
            raise ValueError("exchange reference_atoms must be outside the exchanging copies")
        if not self.offsets or any(not integer(i) or not 0 <= i < len(self.copies[0]) for i in self.offsets):
            raise ValueError("ligand_reference_atoms are offsets within each aligned copy")
        if not isinstance(self.periodic, bool):
            raise ValueError("gamd.exchange.periodic must be boolean")
        if self.cutoff is not None and (not np.isfinite(self.cutoff) or self.cutoff <= 0):
            raise ValueError("gamd.exchange.cutoff_nm must be finite and positive")

    def persist(self):
        if self.sink is not None:
            self.sink.write_bytes(LEDGER_FILENAME, json.dumps({
                "schema_version": 2, "fingerprint": self.fingerprint,
                "settings": self.settings, "selection_policy": self.selection_policy,
                "frame_semantics": "post_exchange",
                "qualification": "experimental_deterministic_selection_not_metropolis",
                "equilibrium_reweighting_qualified": False,
                "energy_convention": "base_force_field_and_kinetic_preserved;restraints_and_boosts_may_change",
                "slot_to_identity": self.slot_to_identity,
                "event_count": self.event_count, "last_event_step": self.last_event_step,
                "pending": self.pending,
            }, indent=2).encode())

    def on_step(self, step, view=None):
        if step != self.kernel.current_step or step % self.interval:
            raise ValueError("exchange hook must run on the current configured step boundary")
        journal_step = self.clock(step)
        if journal_step <= self._last_checked_step:
            return
        box = self.kernel.box_vectors() if self.periodic else None
        if self.periodic and box is None:
            raise ValueError("periodic ligand exchange requires a periodic box")
        positions = self.kernel.positions()
        distances = [minimum_distance(positions, [copy[i] for i in self.offsets],
                                      self.reference_atoms, box) for copy in self.copies]
        if not np.all(np.isfinite(distances)):
            raise ValueError("ligand exchange encountered nonfinite coordinates")
        # Retain a designated copy inside the threshold, even when another is
        # closer. When it leaves, select the nearest other eligible copy.
        # Without a threshold, retain the original all-slot nearest procedure.
        candidates = [slot for slot in range(len(self.copies))
                      if self.cutoff is None or slot != self.slot]
        winner = min(candidates, key=lambda slot: (distances[slot], slot))
        permutation = list(range(len(self.copies)))
        eligible = (self.cutoff is None or
                    (distances[self.slot] > self.cutoff and distances[winner] <= self.cutoff))
        if eligible:
            permutation[self.slot], permutation[winner] = permutation[winner], permutation[self.slot]
        identities = [self.slot_to_identity[i] for i in permutation]
        event = {"step": int(journal_step), "selected_slot": winner, "distances_nm": distances,
                 "permutation": permutation, "slot_to_identity": identities,
                 "changed": permutation != list(range(len(self.copies)))}
        self.last_check = event
        if not event["changed"]:
            self._last_checked_step = journal_step
            return False
        self.pending = event
        self.persist()  # intent precedes mutation; no checkpoint may be emitted inside this hook
        self.kernel.permute_ligands(self.copies, permutation)
        if self.sink is not None:
            self.sink.append_bytes(EVENTS_FILENAME, json.dumps(event, separators=(",", ":")).encode() + b"\n")
        self.slot_to_identity = identities
        self.event_count += 1
        self.last_event_step = journal_step
        self._last_checked_step = journal_step
        self.pending = None
        self.persist()  # completion precedes every reporter and checkpoint at this step
        return True
