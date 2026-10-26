"""Analytical conventions used by the external NMRPipe oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import spectrochempy as scp
import spectrochempy_nmr as scp_nmr

ORACLE = Path(__file__).parent / "data" / "nmrpipe"


def _dataset(data):
    return scp.NDDataset(
        data,
        coordset=[
            scp.Coord.arange(data.shape[0], units="s"),
            scp.Coord.arange(data.shape[1], units="s"),
        ]
        if data.ndim == 2
        else [scp.Coord.arange(data.shape[0], units="s")],
        meta={"td": list(data.shape)},
    )


def test_nmrpipe_f2_order_is_an_explicit_dft_convention():
    """NMRPipe's centered positive-sign DFT maps to SCP by reverse+roll."""
    n = 64
    signal = np.exp(2j * np.pi * 5 * np.arange(n) / n)

    spectrochempy = _dataset(signal).fft().data
    nmrpipe = np.fft.fftshift(np.fft.ifft(signal)) * n
    expected = np.roll(spectrochempy[::-1], 1)

    np.testing.assert_allclose(spectrochempy, np.fft.fftshift(np.fft.fft(signal)))
    np.testing.assert_allclose(nmrpipe, expected, atol=1e-12)
    assert np.argmax(abs(spectrochempy)) == 37
    assert np.argmax(abs(nmrpipe)) == 27


def test_nmrpipe_f1_neg_is_an_explicit_conjugation_convention():
    """The external F1 ``FT -neg`` convention is not the SCP FFT itself."""
    n = 64
    signal = np.exp(2j * np.pi * 5 * np.arange(n) / n)

    spectrochempy = np.asarray(_dataset(signal).fft().data)
    numpy_reference = np.fft.fftshift(np.fft.fft(signal))
    expected = np.conjugate(numpy_reference)

    np.testing.assert_allclose(spectrochempy, numpy_reference)
    np.testing.assert_allclose(expected, np.conjugate(spectrochempy))
    assert np.argmax(abs(expected)) == 37


def test_qf_transpose_and_magnitude_preserve_explicit_shapes():
    """The QF checkpoint operations preserve product shapes and components."""
    data = (
        np.arange(32 * 48, dtype=float).reshape(32, 48)
        + 1j * np.arange(32 * 48, dtype=float).reshape(32, 48) / 10
    )
    transformed = _dataset(data).fft(dim="x", size=64)
    transposed = transformed.T
    magnitude = transposed.mc()

    assert transposed.shape == (64, 32)
    assert magnitude.shape == transposed.shape
    np.testing.assert_allclose(magnitude.data, np.abs(transposed.data))


def test_qf_checkpoint_matches_nmrpipe_oracle():
    """Run the product primitives against compact NMRPipe checkpoints."""
    manifest = json.loads((ORACLE / "qf_oracle_manifest.json").read_text())
    fixture = ORACLE / "qf_oracle.npz"
    assert (
        hashlib.sha256(fixture.read_bytes()).hexdigest() == manifest["fixture_sha256"]
    )

    with np.load(fixture) as oracle:
        for name, details in manifest["files"].items():
            assert oracle[name].shape == tuple(details["shape"])
            assert str(oracle[name].dtype) == details["dtype"]

        raw = oracle["input"]
        f2 = oracle["f2"]
        f2tp = oracle["f2tp"]
        f1 = oracle["f1"]
        magnitude = oracle["magnitude"]

    def zero_fill(data, size):
        result = np.zeros((*data.shape[:-1], size), dtype=data.dtype)
        result[..., : data.shape[-1]] = data
        return result

    # Exercise the plugin's registered QF callback.  NMRPipe's positive-sign
    # centered convention is the explicit reverse+roll mapping below.
    fft_encoding = scp_nmr.NMRPlugin().register_handlers()["fft.encoding"]
    f2_handler = fft_encoding(zero_fill(raw, f2.shape[1]), "QF", original_axis=-1)
    np.testing.assert_allclose(
        f2,
        np.roll(f2_handler[:, ::-1], 1, axis=1),
        rtol=5e-6,
        atol=1e-3,
    )

    f2_dataset = _dataset(f2)
    np.testing.assert_allclose(f2tp, f2_dataset.T.data, rtol=0, atol=0)

    f1_handler = fft_encoding(zero_fill(f2tp, f1.shape[1]), "QF", original_axis=0)
    np.testing.assert_allclose(
        f1,
        np.roll(f1_handler[:, ::-1], 1, axis=1),
        rtol=5e-6,
        atol=1e-3,
    )

    magnitude_scp = scp.NDDataset(f1).mc().T.data
    np.testing.assert_allclose(magnitude, magnitude_scp, rtol=5e-6, atol=1e-3)
