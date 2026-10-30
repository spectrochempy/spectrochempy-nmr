"""Experimental Agilent NOESY States-TPPI processing prototype."""

from __future__ import annotations

import numpy as np


_RECONSTRUCTION_PROFILE = "noesy-states-tppi"


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


def _has_mask(dataset):
    """Return whether any input point is masked."""
    mask = np.asarray(dataset.mask, dtype=bool)
    return bool(np.any(mask))


def _compute_legacy_noesy_arrays(
    dataset,
    *,
    transport,
    ni,
    np_raw,
    arraydim,
    ntraces,
    reject_masked=False,
):
    """Compute the validated NOESY channel arrays without a storage choice."""
    data = _validate_profile(dataset, ni, np_raw, arraydim, ntraces)
    if reject_masked and _has_mask(dataset):
        raise ValueError("Agilent NOESY FFT processing requires unmasked input")

    if transport == "var2pipe":
        data = ((-1) ** np.arange(data.shape[0]))[:, None] * np.conjugate(data)

    # N * fftshift(ifft(data)) equals conjugate(fftshift(fft(conjugate(data))))
    # for the core FFT convention; do not apply N a second time here.
    f2 = np.conjugate(_complex_fft(np.conjugate(data), axis=1))
    e1 = f2[0::2]
    e2 = f2[1::2]
    c_r = e1.real - 1j * e2.real
    c_i = e1.imag - 1j * e2.imag
    return _complex_fft(c_r, axis=0), _complex_fft(c_i, axis=0), data


def _validate_complex_channels(dataset, reconstruction):
    """Validate the vendor-neutral input to the common channel engine."""
    if reconstruction != _RECONSTRUCTION_PROFILE:
        raise ValueError(f"unsupported complex-channel reconstruction: {reconstruction}")
    if tuple(dataset.dims) != ("c", "y", "x"):
        raise ValueError("complex-channel input must have dimensions ('c', 'y', 'x')")
    data = np.asarray(dataset.data)
    if data.ndim != 3 or data.shape[0] != 2 or not np.iscomplexobj(data):
        raise ValueError("complex-channel input must have shape (2, nF1, nF2)")
    if _meta_value(dataset.meta, "component_axis") != "c":
        raise ValueError("complex-channel input must declare component_axis='c'")
    axis_roles = _meta_value(dataset.meta, "axis_roles")
    if (tuple(axis_roles) if axis_roles is not None else ()) != (
        "component",
        "F1",
        "F2",
    ):
        raise ValueError("complex-channel input must declare component, F1 and F2 axis roles")
    if _meta_value(dataset.meta, "stage") != "acquisition":
        raise ValueError("complex-channel input must have stage='acquisition'")
    labels = _meta_value(dataset.meta, "component_labels")
    if (tuple(labels) if labels is not None else ()) != ("branch_0", "branch_1"):
        raise ValueError("complex-channel input must label branches branch_0 and branch_1")
    channel_labels = dataset.coordset["c"].labels
    if channel_labels is None or tuple(str(label) for label in channel_labels) != (
        "branch_0",
        "branch_1",
    ):
        raise ValueError("complex-channel coordinate labels must identify both branches")
    isfreq = _meta_value(dataset.meta, "isfreq")
    if list(isfreq or ()) != [False, False, False]:
        raise ValueError("complex-channel input must be in the time domain")
    td = _meta_value(dataset.meta, "td")
    if list(td or ()) != list(data.shape):
        raise ValueError("complex-channel td metadata must match the dataset shape")
    if _meta_value(dataset.meta, "encoding") not in (None, ""):
        raise ValueError("complex-channel input must not carry an active FFT encoding")
    if _has_mask(dataset):
        raise ValueError("complex-channel FFT processing requires unmasked input")
    return data


def _carried_metadata(dataset):
    """Return descriptive normalized-input metadata without interpreting it."""
    carried = {}
    for key in ("provenance", "acquisition", "normalization"):
        value = _meta_value(dataset.meta, key)
        if value is not None:
            carried[key] = value
    return carried


def _make_channel_dataset(data, *, labels, y_coord, x_coord, stage, meta=None):
    """Construct a channel dataset with explicit stage and axis metadata."""
    import spectrochempy as scp  # noqa: PLC0415

    result = scp.NDDataset(
        np.asarray(data),
        coordset=[
            scp.Coord.arange(2, dims="c", labels=list(labels)),
            y_coord.copy(),
            x_coord.copy(),
        ],
        dims=["c", "y", "x"],
    )
    result.meta.update(
        {
            "component_axis": "c",
            "axis_roles": ("component", "F1", "F2"),
            "stage": stage,
            "component_labels": tuple(labels),
            "reconstruction_profile": _RECONSTRUCTION_PROFILE,
        }
    )
    if meta is not None:
        result.meta.update(meta)
    return result


def _process_complex_channels(dataset, *, reconstruction):
    """Process normalized complex channels using the NDDataset 3D container."""
    _validate_complex_channels(dataset, reconstruction)
    import spectrochempy as scp  # noqa: PLC0415

    working = dataset.copy()
    working.meta.td = list(working.shape)
    carried = _carried_metadata(dataset)
    # The validated NOESY transport uses N * fftshift(ifft). Express that
    # convention through the core's ordinary complex FFT identity, while
    # retaining the complete 3D channel dataset throughout the operation.
    f2_input = working.copy()
    f2_input.data = np.conjugate(f2_input.data)
    f2 = scp.fft(f2_input, dim="x")
    f2_data = np.conjugate(np.asarray(f2.data))
    c_r = f2_data[0].real - 1j * f2_data[1].real
    c_i = f2_data[0].imag - 1j * f2_data[1].imag
    interferograms = _make_channel_dataset(
        np.stack([c_r, c_i], axis=0),
        labels=("C_R", "C_I"),
        y_coord=f2.coordset["y"],
        x_coord=f2.coordset["x"],
        stage="f1_interferograms",
        meta={
            "processing_profile": _RECONSTRUCTION_PROFILE,
            "fft_f2": True,
            "reconstruction_f1": True,
            **carried,
        },
    )
    interferograms.meta.isfreq = [False, False, True]
    interferograms.meta.td = list(interferograms.shape)
    spectrum = scp.fft(interferograms, dim="y")
    spectrum = _make_channel_dataset(
        np.asarray(spectrum.data),
        labels=("S_R", "S_I"),
        y_coord=spectrum.coordset["y"],
        x_coord=spectrum.coordset["x"],
        stage="spectrum",
        meta={
            "processing_profile": _RECONSTRUCTION_PROFILE,
            "fft_f2": True,
            "reconstruction_f1": True,
            "fft_f1": True,
            "mathematical_component_order": ("RR", "RI", "IR", "II"),
            "data_domain": "frequency",
            "coordinates_status": "provisional_index",
            **carried,
        },
    )
    spectrum.meta.update(
        {
            "fft_dimensions": ("x", "y"),
        }
    )
    spectrum.meta.td = list(spectrum.shape)
    spectrum.meta.isfreq = [False, True, True]
    return spectrum


def _agilent_channels_adapter(dataset, *, transport, ni, np_raw, arraydim, ntraces):
    """Validate Agilent metadata and produce normalized branch channels."""
    data = _validate_profile(dataset, ni, np_raw, arraydim, ntraces)
    if _has_mask(dataset):
        raise ValueError("Agilent NOESY FFT processing requires unmasked input")
    if transport == "var2pipe":
        data = ((-1) ** np.arange(data.shape[0]))[:, None] * np.conjugate(data)

    branches = np.stack([data[0::2], data[1::2]], axis=0)
    import spectrochempy as scp  # noqa: PLC0415

    channels = scp.NDDataset(
        branches,
        coordset=[
            scp.Coord.arange(2, dims="c", labels=["branch_0", "branch_1"]),
            scp.Coord.arange(ni, dims="y"),
            scp.Coord.arange(np_raw // 2, dims="x"),
        ],
        dims=["c", "y", "x"],
    )
    channels.meta.update(
        {
            "stage": "acquisition",
            "component_axis": "c",
            "component_labels": ("branch_0", "branch_1"),
            "axis_roles": ("component", "F1", "F2"),
            "processing_profile": _RECONSTRUCTION_PROFILE,
            "transport": transport,
            "coordinates_status": "provisional_index",
            "raw_shape": tuple(data.shape),
            "provenance": _meta_value(dataset.meta, "provenance"),
            "acquisition": _meta_value(dataset.meta, "acquisition"),
            "normalization": _meta_value(dataset.meta, "normalization"),
        }
    )
    channels.meta.isfreq = [False, False, False]
    channels.meta.td = list(channels.shape)
    return channels


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
    s_r, s_i, data = _compute_legacy_noesy_arrays(
        dataset,
        transport=transport,
        ni=ni,
        np_raw=np_raw,
        arraydim=arraydim,
        ntraces=ntraces,
    )

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


def process_agilent_noesy_states_tppi_channels(
    dataset,
    *,
    transport="none",
    ni=None,
    np_raw=None,
    arraydim=None,
    ntraces=None,
):
    """Return the experimental NOESY spectrum as a complex-channel dataset.

    The result has shape ``(2, ni, np_raw // 2)`` and dimensions ``("c", "y",
    "x")``. Its channels are ``S_R`` and ``S_I``. The input is an explicitly
    annotated, unmasked NOESY profile; no quaternion storage or quaternion FFT
    is used.
    """
    if transport not in ("none", "var2pipe"):
        raise ValueError("transport must be 'none' or 'var2pipe'")
    meta = dataset.meta
    ni = _meta_value(meta, "ni", ni)
    np_raw = _meta_value(meta, "np_raw", _meta_value(meta, "np", np_raw))
    arraydim = _meta_value(meta, "arraydim", arraydim)
    ntraces = _meta_value(meta, "ntraces", ntraces)
    channels = _agilent_channels_adapter(
        dataset,
        transport=transport,
        ni=ni,
        np_raw=np_raw,
        arraydim=arraydim,
        ntraces=ntraces,
    )
    result = _process_complex_channels(
        channels,
        reconstruction=_RECONSTRUCTION_PROFILE,
    )
    result.meta.update(
        {
            "processing_profile": "agilent-noesy-states-tppi-experimental",
            "stage": "spectrum",
            "transport": transport,
            "component_axis": "c",
            "component_labels": ("S_R", "S_I"),
            "axis_roles": ("component", "F1", "F2"),
            "mathematical_component_order": ("RR", "RI", "IR", "II"),
            "data_domain": "frequency",
            "coordinates_status": "provisional_index",
            "normalization": _meta_value(result.meta, "normalization", "none"),
            "provenance": _meta_value(meta, "provenance"),
            "raw_shape": tuple(dataset.data.shape),
            "raw_ni": int(ni),
            "raw_np": int(np_raw),
            "raw_arraydim": int(arraydim),
            "raw_ntraces": int(ntraces),
            "returned_shape": (2, int(ni), int(np_raw // 2)),
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
    result.meta.td = [2, int(ni), int(np_raw // 2)]
    result.meta.isfreq = [False, True, True]
    return result


def extract_noesy_components(dataset):
    """Return ``RR``, ``RI``, ``IR`` and ``II`` as independent 2D datasets."""
    if tuple(dataset.dims) != ("c", "y", "x"):
        raise ValueError("NOESY channel dataset must have dimensions ('c', 'y', 'x')")
    if _meta_value(dataset.meta, "stage") != "spectrum":
        raise ValueError("NOESY components require a spectrum-stage dataset")
    if tuple(_meta_value(dataset.meta, "component_labels", ())) != ("S_R", "S_I"):
        raise ValueError("NOESY spectrum channels must be labelled S_R and S_I")

    import spectrochempy as scp  # noqa: PLC0415

    source = np.ma.array(dataset.data, mask=np.asarray(dataset.mask, dtype=bool))
    sr, si = source[0], source[1]
    arrays = {"RR": sr.real, "RI": si.real, "IR": sr.imag, "II": si.imag}
    masks = {
        "RR": np.ma.getmaskarray(sr),
        "IR": np.ma.getmaskarray(sr),
        "RI": np.ma.getmaskarray(si),
        "II": np.ma.getmaskarray(si),
    }
    coords = [dataset.coordset["y"].copy(), dataset.coordset["x"].copy()]
    result = {}
    for name, values in arrays.items():
        item = scp.NDDataset(
            np.ma.array(
                np.asarray(values).copy(),
                mask=np.asarray(masks[name], dtype=bool).copy(),
            ),
            coordset=coords,
            dims=["y", "x"],
        )
        item.meta.update(
            {
                "stage": "spectrum_component",
                "component": name,
                "component_source": "S_R" if name in ("RR", "IR") else "S_I",
                "profile": _meta_value(dataset.meta, "processing_profile"),
            }
        )
        result[name] = item
    return result
