"""Stable engine facade for prepared-bundle preparation verification.

The implementation moved to the ``neomd_preparation`` package with the
preparation subsystem. This module keeps the historical
``neomd.prepared_pipeline`` import path (the verifier identity is part of the
prepared-bundle contract), and the import is function-level so importing the
engine never requires the preparation package at module import.
"""

from __future__ import annotations


def verify_protein_preparation(*args, **kwargs):
    """Verify the exact LOAD->GROUP->repair->H->solvent path and actual System.

    Delegates to :func:`neomd_preparation.verify.pipeline.verify_protein_preparation`;
    see that function for the verification contract.
    """
    from neomd_preparation.verify.pipeline import (
        verify_protein_preparation as impl,
    )

    return impl(*args, **kwargs)
