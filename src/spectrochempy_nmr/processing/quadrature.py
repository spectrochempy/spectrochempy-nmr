"""Validated mathematical quadrature reconstructions."""

from __future__ import annotations

import numpy as np


NOESY_STATES_TPPI = "noesy-states-tppi"


def reconstruct_f1_interferograms(data, *, profile):
    """Reconstruct ``C_R`` and ``C_I`` for one validated profile.

    ``data`` contains the two complex branches after the direct-dimension
    transform.  This function deliberately returns arrays only; dataset
    construction and Fourier orchestration belong to the common engine.
    """
    if profile != NOESY_STATES_TPPI:
        raise ValueError(f"unsupported quadrature reconstruction: {profile}")
    data = np.asarray(data)
    if data.ndim != 3 or data.shape[0] != 2 or not np.iscomplexobj(data):
        raise ValueError("quadrature reconstruction requires two complex channels")
    c_r = data[0].real - 1j * data[1].real
    c_i = data[0].imag - 1j * data[1].imag
    return np.stack([c_r, c_i], axis=0)
