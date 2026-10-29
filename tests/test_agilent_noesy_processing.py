"""Synthetic tests for the opt-in Agilent NOESY prototype."""

import numpy as np
import pytest
import spectrochempy as scp
from spectrochempy_hypercomplex import as_float_array

from spectrochempy_nmr import process_agilent_noesy_states_tppi


def _dataset():
    rows, points = 6, 8
    row = np.arange(rows, dtype=float)[:, None]
    col = np.arange(points, dtype=float)[None, :]
    data = (
        np.exp(1j * (0.31 * row + 0.17 * col)) * (1.0 + 0.07 * row + 0.013 * col)
    ).astype(np.complex64)
    ds = scp.NDDataset(
        data,
        coordset=[scp.Coord.arange(rows), scp.Coord.arange(points)],
    )
    ds.meta.update(
        {
            "seqfil": "NOESY",
            "phase": (1, 2),
            "ni": 3,
            "np": 16,
            "arraydim": 6,
        }
    )
    return ds


def _reference(raw):
    f2 = raw.shape[1] * np.fft.fftshift(np.fft.ifft(raw, axis=1), axes=1)
    e1, e2 = f2[0::2], f2[1::2]
    sr = np.fft.fftshift(np.fft.fft(e1.real - 1j * e2.real, axis=0), axes=0)
    si = np.fft.fftshift(np.fft.fft(e1.imag - 1j * e2.imag, axis=0), axes=0)
    return np.stack([sr.real, si.real, sr.imag, si.imag], axis=-1)


def _var2pipe_reference(raw):
    return (-1) ** np.arange(raw.shape[0])[:, None] * np.conjugate(raw)


def test_noesy_pipeline_matches_independent_complex_reference():
    source = _dataset()
    expected = _reference(source.data)

    result = process_agilent_noesy_states_tppi(source)

    assert result.data.shape == (3, 8)
    np.testing.assert_allclose(
        as_float_array(result.data), expected, rtol=2e-6, atol=2e-6
    )
    assert result.meta.mathematical_component_order == ("RR", "RI", "IR", "II")
    assert result.meta.raw_shape == (6, 8)
    assert result.meta.raw_ni == 3
    assert result.meta.raw_np == 16
    assert result.meta.returned_shape == (3, 8)
    assert result.meta.td == [3, 8]
    assert result.coordset[0].size == 8
    assert result.coordset[1].size == 3
    assert source.data.shape == (6, 8)


def test_noesy_pipeline_accepts_explicit_var2pipe_transport():
    source = _dataset()
    result = process_agilent_noesy_states_tppi(source, transport="var2pipe")
    expected = _reference(_var2pipe_reference(source.data))

    assert result.meta.transport == "var2pipe"
    assert result.data.shape == (3, 8)
    np.testing.assert_allclose(
        as_float_array(result.data), expected, rtol=2e-6, atol=2e-6
    )


def test_noesy_pipeline_rejects_unknown_transport():
    with pytest.raises(ValueError, match="transport"):
        process_agilent_noesy_states_tppi(_dataset(), transport="guess")


@pytest.mark.parametrize(
    "changes",
    [
        {"seqfil": "gHSQCAD"},
        {"phase": (1,)},
        {"phase": (True, 2)},
        {"phase": (1.0, 2.0)},
        {"arraydim": 4},
        {"np": 15},
        {"ni": True},
    ],
)
def test_noesy_pipeline_rejects_non_noesy_profiles(changes):
    source = _dataset()
    source.meta.update(changes)

    with pytest.raises(ValueError):
        process_agilent_noesy_states_tppi(source)


def test_noesy_pipeline_inplace_returns_none():
    source = _dataset()

    result = process_agilent_noesy_states_tppi(source, inplace=True)

    assert result is None
    assert source.data.shape == (3, 8)
