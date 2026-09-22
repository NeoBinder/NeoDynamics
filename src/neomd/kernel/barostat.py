"""Persistent isotropic Monte Carlo proposal state; no OpenMM dependency.

The molecular-center proposal and adaptation follow OpenMM's isotropic
barostat. Its private ForceImpl counters are not checkpointed in OpenMM 8.6;
keeping this small state here makes interrupted runs reproducible.
"""
from __future__ import annotations

import math

import numpy as np

R_KJ_MOL_K = 0.00831446261815324
BAR_NM3_TO_KJ_MOL = 0.0602214076


def metropolis_work(delta_energy, pressure_bar, old_volume, new_volume,
                    temperature, molecules):
    """ΔU + PΔV − Nmol RT log(Vnew/Vold), in kJ/mol."""
    return (delta_energy + pressure_bar * BAR_NM3_TO_KJ_MOL * (new_volume - old_volume)
            - molecules * R_KJ_MOL_K * temperature * math.log(new_volume / old_volume))


class IsotropicBarostat:
    """Step phase, proposal amplitude, adaptive counts and independent RNG."""

    def __init__(self, frequency, seed):
        self.frequency = int(frequency)
        self.since = 0
        self.volume_scale = None
        self.attempted = 0
        self.accepted = 0
        self.rng = np.random.default_rng(seed if seed != 0 else None)

    def initialize_volume(self, volume):
        if self.volume_scale is None:
            self.volume_scale = .01 * volume

    def record(self, accepted, old_volume):
        self.attempted += 1
        self.accepted += int(accepted)
        if self.attempted >= 10:
            if self.accepted < .25 * self.attempted:
                self.volume_scale /= 1.1
            elif self.accepted > .75 * self.attempted:
                self.volume_scale = min(self.volume_scale * 1.1, old_volume * .3)
            else:
                return
            self.attempted = self.accepted = 0

    def state(self):
        return dict(frequency=self.frequency, since=self.since, volume_scale=self.volume_scale,
                    attempted=self.attempted, accepted=self.accepted,
                    rng=self.rng.bit_generator.state)

    def restore(self, state):
        if state['frequency'] != self.frequency:
            raise ValueError('NPT checkpoint barostat frequency differs from the configured frequency')
        since, attempted, accepted = state['since'], state['attempted'], state['accepted']
        scale = state['volume_scale']
        if (not 0 <= since < self.frequency or not 0 <= accepted <= attempted
                or not math.isfinite(scale) or scale <= 0):
            raise ValueError('invalid managed NPT checkpoint state')
        self.since, self.attempted, self.accepted = since, attempted, accepted
        self.volume_scale = scale
        self.rng.bit_generator.state = state['rng']
