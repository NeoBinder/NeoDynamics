"""Native preparation contract, frozen-input transport and classic science.

This subpackage is the reusable seam for a resident preparation service: the
versioned native contract (:mod:`.native_preparation`), the trusted
frozen-input transport (:mod:`.batch_prepare`) and the classic executor
(:mod:`.batch_prepare_science`).  Callers outside this distribution import
only these names; the modules stay importable directly for tests.
"""

from __future__ import annotations

from .batch_prepare import (
    BatchPreparationError,
    canonical_json,
    materialize_preparation,
    run_batch_preparation,
    sha256,
)
from .batch_prepare_science import execute_classic_preparation, project_qc
from .native_preparation import (
    CONTRACT,
    PROFILE,
    contract_directory,
    inspect_native_preparation,
    load_native_preparation,
)

__all__ = [
    "CONTRACT",
    "PROFILE",
    "BatchPreparationError",
    "canonical_json",
    "contract_directory",
    "execute_classic_preparation",
    "inspect_native_preparation",
    "load_native_preparation",
    "materialize_preparation",
    "project_qc",
    "run_batch_preparation",
    "sha256",
]
