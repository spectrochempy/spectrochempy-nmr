# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: S101, F841

"""Tests for Agilent/Varian NMR reader."""

import numpy as np
import pytest
from spectrochempy_nmr.nmr_metadata import NMRMetadata
from spectrochempy_nmr.nmr_metadata import extract_agilent_metadata
from spectrochempy_nmr.readers.read_agilent import _reconstruct_agilent_states

import spectrochempy as scp

EXTRA_DATADIR = scp.preferences.datadir.parent / "testdata-extra"
AGILENT_DIR = EXTRA_DATADIR / "testdata" / "nmrdata" / "agilent"


def _has_agilent_data():
    return (AGILENT_DIR / "agilent_1d" / "fid").exists()


def _read_agilent_or_skip(*args, **kwargs):
    try:
        result = scp.nmr.read_agilent(*args, **kwargs)
    except FileNotFoundError as exc:
        pytest.skip(f"Agilent test data incomplete: {exc}")
    if result is None:
        pytest.skip("Agilent test data could not be read in this environment")
    return result


# ---------------------------------------------------------------------------
# nmrglue vendored layer
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestAgilentNMRGlue:
    """Tests for the vendored nmrglue Agilent reading functions."""

    def test_read_varian_procpar(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_1d" / "procpar"))
        assert isinstance(pdic, dict)
        assert "np" in pdic
        assert "sw" in pdic
        assert "sfrq" in pdic
        assert pdic["np"]["values"] == ["3000"]
        assert pdic["sw"]["values"] == ["50000"]

    def test_find_varian_shape_1d(self):
        from spectrochempy_nmr.extern.nmrglue import find_varian_shape
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_1d" / "procpar"))
        shape = find_varian_shape(pdic)
        assert shape == (1500,)

    def test_find_varian_shape_2d(self):
        from spectrochempy_nmr.extern.nmrglue import find_varian_shape
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_2d" / "procpar"))
        shape = find_varian_shape(pdic)
        assert shape[1] == 1500  # direct dim np/2
        assert shape[0] == 332  # ni * 2 phase values

    def test_read_varian_fid_1d(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian

        dic, data = read_varian(
            str(AGILENT_DIR / "agilent_1d"),
            fid_file="fid",
            procpar_file="procpar",
        )
        assert data.ndim == 1
        assert data.shape == (1500,)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_read_varian_fid_2d(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian

        dic, data = read_varian(
            str(AGILENT_DIR / "agilent_2d"),
            fid_file="fid",
            procpar_file="procpar",
        )
        assert data.ndim == 2
        assert data.shape == (332, 1500)
        assert np.issubdtype(data.dtype, np.complexfloating)

    def test_find_varian_torder(self):
        from spectrochempy_nmr.extern.nmrglue import find_varian_shape
        from spectrochempy_nmr.extern.nmrglue import find_varian_torder
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_2d" / "procpar"))
        shape = find_varian_shape(pdic)
        torder = find_varian_torder(pdic, shape)
        assert torder in ("f", "r", "o")

    def test_procpar_key_count(self):
        from spectrochempy_nmr.extern.nmrglue import read_varian_procpar

        pdic = read_varian_procpar(str(AGILENT_DIR / "agilent_1d" / "procpar"))
        assert len(pdic) > 100  # typical procpar has 100+ parameters


# ---------------------------------------------------------------------------
# Public reader API
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestReadAgilent:
    """Tests for the public read_agilent() function."""

    def test_read_1d_fid(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.shape == (1500,)
        assert ds.origin == "agilent"
        assert np.issubdtype(ds.data.dtype, np.complexfloating)
        assert ds.history_entries[-1]["operation"] is None
        assert (
            ds.history_entries[-1]["message"]
            == "Imported Agilent/Varian dataset agilent_1d (FID)"
        )

    def test_read_1d_directory(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d"))
        assert ds.shape == (1500,)
        assert ds.origin == "agilent"

    def test_read_2d_fid(self):
        pytest.importorskip("spectrochempy_hypercomplex")
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        assert ds.ndim == 2
        assert ds.origin == "agilent"
        assert ds.data.shape == (166, 1500)
        assert ds.meta.td == [166, 1500]
        assert ds.x.size == 1500
        assert ds.y.size == 166

    def test_read_2d_tppi(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d_tppi" / "fid"))
        assert ds.ndim == 2
        assert ds.origin == "agilent"
        enc = ds.meta.encoding
        assert "TPPI" in enc

    def test_1d_metadata_sw(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        sw = ds.meta.sw
        assert sw[0] == 50000.0

    def test_1d_metadata_sfrq(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        sfrq = ds.meta.sfrq
        assert sfrq[0] is not None
        assert 100 < sfrq[0] < 200  # C13 frequency around 125 MHz

    def test_1d_metadata_nucleus(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        nuc = ds.meta.nucleus
        assert nuc[0] is not None

    def test_1d_complex_data(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert np.issubdtype(ds.data.dtype, np.complexfloating)

    def test_1d_normalised(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.meta.ns_norm == 1.0

    def test_1d_has_coords(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.x is not None

    def test_2d_encoding(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        enc = ds.meta.encoding
        assert len(enc) == 2
        assert enc[1] in ("STATES", "TPPI")

    def test_2d_iscomplex(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        ic = ds.meta.iscomplex
        assert len(ic) == 2
        assert ic[-1] is True  # direct dim is complex

    def test_2d_has_coords(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        assert ds.x is not None
        assert ds.y is not None

    def test_2d_origin(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        assert ds.origin == "agilent"

    def test_meta_readonly(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        assert ds.meta.readonly is True


class TestAgilentStatesReconstruction:
    """Synthetic tests for the documented Agilent States trace layout."""

    def test_pairs_traces_and_preserves_direct_complex_axis(self):
        as_float_array = pytest.importorskip(
            "spectrochempy_hypercomplex"
        ).as_float_array

        data = np.empty((4, 3), dtype=np.complex128)
        data[0] = [1 + 2j, 3 + 4j, 5 + 6j]
        data[1] = [10 + 20j, 30 + 40j, 50 + 60j]
        data[2] = [100 + 200j, 300 + 400j, 500 + 600j]
        data[3] = [1000 + 2000j, 3000 + 4000j, 5000 + 6000j]

        result = _reconstruct_agilent_states(data)

        assert result.shape == (2, 3)
        np.testing.assert_array_equal(
            as_float_array(result),
            np.array(
                [
                    [[1, 2, 10, 20], [3, 4, 30, 40], [5, 6, 50, 60]],
                    [
                        [100, 200, 1000, 2000],
                        [300, 400, 3000, 4000],
                        [500, 600, 5000, 6000],
                    ],
                ]
            ),
        )

    def test_odd_trace_count_is_rejected_without_pairing(self):
        with pytest.raises(ValueError, match="paired traces"):
            _reconstruct_agilent_states(np.zeros((3, 4), dtype=np.complex128))


# ---------------------------------------------------------------------------
# extract_agilent_metadata
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_agilent_data(), reason="Agilent test data not available")
class TestExtractAgilentMetadata:
    """Tests for extract_agilent_metadata()."""

    def test_1d_metadata(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_1d" / "fid"))
        nmr_meta = extract_agilent_metadata(ds.meta)
        assert isinstance(nmr_meta, NMRMetadata)
        assert nmr_meta.ndim == 1
        assert nmr_meta.domains == ("time",)
        assert nmr_meta.source_kind == "fid"
        assert nmr_meta.iscomplex == (True,)
        assert nmr_meta.spectral_width_hz is not None
        assert nmr_meta.spectral_width_hz[0] == 50000.0
        assert nmr_meta.spectrometer_freq_mhz is not None
        assert 100 < nmr_meta.spectrometer_freq_mhz[0] < 200

    def test_2d_metadata(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d" / "fid"))
        nmr_meta = extract_agilent_metadata(ds.meta)
        assert nmr_meta.ndim == 2
        assert nmr_meta.domains == ("time", "time")
        assert nmr_meta.source_kind == "ser"
        assert nmr_meta.nuclei is not None
        assert len(nmr_meta.nuclei) == 2

    def test_none_input(self):
        result = extract_agilent_metadata(None)
        assert result.ndim == 0
        assert result.domains == ()

    def test_encoding_strings(self):
        ds = _read_agilent_or_skip(str(AGILENT_DIR / "agilent_2d_tppi" / "fid"))
        nmr_meta = extract_agilent_metadata(ds.meta)
        assert nmr_meta.encoding is not None
        assert "TPPI" in nmr_meta.encoding
