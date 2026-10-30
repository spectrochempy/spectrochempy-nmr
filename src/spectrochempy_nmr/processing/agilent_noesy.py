"""Experimental Agilent NOESY States-TPPI processing prototype."""

from __future__ import annotations

import numpy as np


def _meta_value(meta, key, default=None):
    """Return a metadata value without depending on a concrete Meta class."""
    try:
        return meta.get(key, default)
    except AttributeError:
        return default


def _complex_fft(data, axis):
    """Apply the public SpectroChemPy complex FFT to a plain complex array."""
    import spectrochempy as scp  # noqa: PLC0415

    ndim = data.ndim
    dataset = scp.NDDataset(
        data,
        coordset=[scp.Coord.arange(size) for size in data.shape],
    )
    dataset.meta.isfreq = [False] * ndim
    dataset.meta.td = list(data.shape)
    return np.asarray(scp.fft(dataset, dim=axis).data)


def _validate_profile(dataset, ni, np_raw, arraydim, ntraces):
    """Validate the deliberately narrow experimental profile."""
    data = np.asarray(dataset.data)
    if data.ndim != 2 or not np.iscomplexobj(data):
        raise ValueError("Agilent NOESY processing requires a complex 2D array")
    if _meta_value(dataset.meta, "seqfil") != "NOESY":
        raise ValueError("Agilent NOESY processing requires seqfil='NOESY'")
    if ntraces is None:
        raise ValueError("Agilent NOESY processing requires ntraces=1 metadata")
    if isinstance(ntraces, bool) or not isinstance(ntraces, (int, np.integer)):
        raise ValueError("Agilent NOESY ntraces must be an integer")
    if ntraces != 1:
        raise ValueError("Agilent NOESY processing requires ntraces=1")

    phase = _meta_value(dataset.meta, "phase")
    if (
        not isinstance(phase, (tuple, list))
        or len(phase) != 2
        or any(isinstance(value, bool) for value in phase)
        or not all(isinstance(value, (int, np.integer)) for value in phase)
        or tuple(int(value) for value in phase) != (1, 2)
    ):
        raise ValueError("Agilent NOESY processing requires phase=(1, 2)")
    if any(value is None for value in (ni, np_raw, arraydim)):
        raise ValueError("ni, np_raw and arraydim metadata are required")
    if any(isinstance(value, bool) for value in (ni, np_raw, arraydim)):
        raise ValueError("ni, np_raw and arraydim must be integer values")
    if not all(
        isinstance(value, (int, np.integer)) for value in (ni, np_raw, arraydim)
    ):
        raise ValueError("ni, np_raw and arraydim must be integers")
    if ni <= 0 or np_raw <= 0 or np_raw % 2 or arraydim != 2 * ni:
        raise ValueError("inconsistent Agilent NOESY States dimensions")
    expected = (arraydim, np_raw // 2)
    if data.shape != expected:
        raise ValueError(f"data shape {data.shape} does not match {expected}")
    return data


def process_agilent_noesy_states_tppi(
    dataset,
    *,
    transport="none",
    ni=None,
    np_raw=None,
    arraydim=None,
    ntraces=None,
    inplace=False,
):
    """Process an explicitly annotated Agilent NOESY States-TPPI dataset.

    This is an experimental, opt-in prototype. The input must be the raw
    complex matrix with ``seqfil='NOESY'`` and ``phase=(1, 2)`` metadata.
    ``transport='none'`` expects NMRPipe-convention samples; the explicit
    ``transport='var2pipe'`` option applies the validated D2473 transport
    operator before the two complex FFTs.

    No quaternion FFT, apodization, phasing, LP, baseline correction or
    canonical physical quadrature interpretation is performed.
    """
    if transport not in ("none", "var2pipe"):
        raise ValueError("transport must be 'none' or 'var2pipe'")
    meta = dataset.meta
    ni = _meta_value(meta, "ni", ni)
    np_raw = _meta_value(meta, "np_raw", _meta_value(meta, "np", np_raw))
    arraydim = _meta_value(meta, "arraydim", arraydim)
    ntraces = _meta_value(meta, "ntraces", ntraces)
    data = _validate_profile(dataset, ni, np_raw, arraydim, ntraces)

    if transport == "var2pipe":
        data = ((-1) ** np.arange(data.shape[0]))[:, None] * np.conjugate(data)

    # N * fftshift(ifft(data)) equals conjugate(fftshift(fft(conjugate(data))))
    # for the core FFT convention; do not apply N a second time here.
    f2 = np.conjugate(_complex_fft(np.conjugate(data), axis=1))
    e1 = f2[0::2]
    e2 = f2[1::2]
    c_r = e1.real - 1j * e2.real
    c_i = e1.imag - 1j * e2.imag
    s_r = _complex_fft(c_r, axis=0)
    s_i = _complex_fft(c_i, axis=0)

    rr, ir = s_r.real, s_r.imag
    ri, ii = s_i.real, s_i.imag
    try:
        from spectrochempy_hypercomplex import as_quaternion  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Agilent NOESY processing requires spectrochempy-hypercomplex"
        ) from exc

    quaternion = as_quaternion(rr + 1j * ri, ir + 1j * ii)
    import spectrochempy as scp  # noqa: PLC0415

    result = dataset if inplace else dataset.copy()
    result.data = quaternion
    result.set_coordset(scp.Coord.arange(ni), scp.Coord.arange(np_raw // 2))
    result.meta.update(
        {
            "processing_profile": "agilent-noesy-states-tppi-experimental",
            "transport": transport,
            "fft_dimensions": ("F2", "F1"),
            "states_pairing": "trace[2a]=phase=1; trace[2a+1]=phase=2",
            "mathematical_component_order": ("RR", "RI", "IR", "II"),
            "normalization": "none",
            "apodization": "none",
            "linear_prediction": "none",
            "raw_shape": tuple(data.shape),
            "raw_ni": int(ni),
            "raw_np": int(np_raw),
            "raw_arraydim": int(arraydim),
            "raw_ntraces": int(ntraces),
            "returned_shape": (int(ni), int(np_raw // 2)),
            "data_domain": "frequency",
            "axis_roles": ("F1", "F2"),
            "coordinates_status": "provisional_index",
            "raw_acquisition": {
                "seqfil": "NOESY",
                "phase": (1, 2),
                "ni": int(ni),
                "np": int(np_raw),
                "arraydim": int(arraydim),
                "ntraces": int(ntraces),
            },
        }
    )
    result.meta.td = [int(ni), int(np_raw // 2)]
    result.meta.isfreq = [True, True]
    if inplace:
        return None
    return result
