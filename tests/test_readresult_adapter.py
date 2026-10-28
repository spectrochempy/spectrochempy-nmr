"""Tests for the internal ReadResult-to-NDDataset adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spectrochempy_nmr.readresult_adapter import ReadResultAdapterError
from spectrochempy_nmr.readresult_adapter import read_result_to_nddataset


def _result(*, components=None, fmt="bruker-pdata", diagnostics=()):
    real = np.arange(8, dtype=float)
    imaginary = np.linspace(1.0, 2.0, 8)
    if components is None:
        components = [
            SimpleNamespace(name="1r", data=real),
            SimpleNamespace(name="1i", data=imaginary),
        ]
    metadata = {
        "acqus": {"RG": 40.3, "NS": 16, "TD": 16},
        "procs": {"OFFSET": 10.0, "SF": 400.0, "SW_p": 8000.0, "SI": 8},
    }
    axis = SimpleNamespace(
        values=np.linspace(4000.0, -4000.0, 8),
        unit="Hz",
        label="1H",
        direction="decreasing",
        reference=SimpleNamespace(
            observation_frequency=400.0,
            observation_frequency_unit="MHz",
            offset=10.0,
            offset_unit="ppm",
            source="processing",
        ),
    )
    return SimpleNamespace(
        data=real,
        axes=(axis,),
        metadata=metadata,
        components=tuple(components),
        applied_corrections={"scale_data": True, "NC_proc": -5},
        diagnostics=diagnostics,
        format=fmt,
        source="synthetic/pdata/1",
    )


def test_decoded_policy_reconstructs_complex_data_and_preserves_metadata():
    result = _result()
    original = result.components[0].data.copy()
    adapted = read_result_to_nddataset(result, normalization="decoded")

    np.testing.assert_array_equal(adapted.data.real, result.components[0].data)
    np.testing.assert_array_equal(adapted.data.imag, result.components[1].data)
    assert adapted.data.dtype == np.complex128
    assert str(adapted.x.units) == "ppm"
    assert adapted.x.data[0] == pytest.approx(10.0)
    assert adapted.x.data[-1] < adapted.x.data[0]
    assert adapted.meta.nmrglue_read_result["applied_corrections"]["NC_proc"] == -5
    np.testing.assert_array_equal(result.components[0].data, original)
    np.testing.assert_allclose(
        adapted.x.data,
        np.asarray([10.0 - 8000.0 / 400.0 / 7 * i for i in range(8)]),
        atol=1e-3,
    )


def test_historical_policy_uses_acquisition_rg_only():
    result = _result()
    adapted = read_result_to_nddataset(
        result,
        normalization="spectrochempy-historical",
        axis_unit="Hz",
    )

    np.testing.assert_allclose(
        adapted.data,
        (result.components[0].data + 1j * result.components[1].data) / 40.3,
    )
    assert str(adapted.x.units) == "Hz"
    assert adapted.meta.nmrglue_read_result["normalization"] == {
        "policy": "spectrochempy-historical",
        "RG": 40.3,
        "NS": 16,
    }


def test_adapter_rejects_non_bruker_or_non_1d_profiles():
    with pytest.raises(ReadResultAdapterError, match="Bruker processed 1D"):
        read_result_to_nddataset(_result(fmt="bruker-raw"))

    result = _result()
    result.data = np.zeros((2, 8))
    with pytest.raises(ReadResultAdapterError, match="Only 1D"):
        read_result_to_nddataset(result)


def test_adapter_rejects_invalid_historical_rg_and_component_shapes():
    missing_rg = _result()
    missing_rg.metadata["acqus"]["RG"] = None
    with pytest.raises(ReadResultAdapterError, match="acquisition RG"):
        read_result_to_nddataset(missing_rg, normalization="spectrochempy-historical")

    bad = _result(
        components=[
            SimpleNamespace(name="1r", data=np.zeros(8)),
            SimpleNamespace(name="1i", data=np.zeros(7)),
        ]
    )
    with pytest.raises(ReadResultAdapterError, match="different shapes"):
        read_result_to_nddataset(bad)


def test_adapter_rejects_missing_primary_component():
    result = _result(components=[SimpleNamespace(name="1i", data=np.zeros(8))])

    with pytest.raises(ReadResultAdapterError, match="1r component"):
        read_result_to_nddataset(result)


def test_missing_imaginary_component_is_explicitly_diagnostic():
    result = _result(components=[SimpleNamespace(name="1r", data=np.arange(8.0))])
    adapted = read_result_to_nddataset(result)

    assert not np.iscomplexobj(adapted.data)
    diagnostics = adapted.meta.nmrglue_read_result["diagnostics"]
    assert diagnostics[-1]["code"] == "component.missing"


@pytest.mark.parametrize("name", ["OFFSET", "SF", "SW_p"])
def test_adapter_rejects_missing_processed_reference_parameter(name):
    result = _result()
    result.metadata["procs"].pop(name)

    with pytest.raises(ReadResultAdapterError, match=name):
        read_result_to_nddataset(result)


@pytest.mark.parametrize("name", ["OFFSET", "SF", "SW_p"])
def test_adapter_rejects_malformed_processed_reference_parameter(name):
    result = _result()
    result.metadata["procs"][name] = float("nan") if name == "OFFSET" else "bad"

    with pytest.raises(ReadResultAdapterError, match=name):
        read_result_to_nddataset(result)


def test_adapter_rejects_unsupported_axis_units():
    result = _result()
    result.axes[0].unit = "points"

    with pytest.raises(ReadResultAdapterError, match="Unsupported"):
        read_result_to_nddataset(result)

    with pytest.raises(ReadResultAdapterError, match="Unsupported"):
        read_result_to_nddataset(_result(), axis_unit="kHz")

    result.axes[0].unit = "points"
    with pytest.raises(ReadResultAdapterError, match="Unsupported"):
        read_result_to_nddataset(result, axis_unit="points")


def test_axis_conversion_respects_observation_frequency_units():
    result = _result()
    result.axes[0].reference.observation_frequency = 0.4
    result.axes[0].reference.observation_frequency_unit = "GHz"
    adapted = read_result_to_nddataset(result, axis_unit="ppm")

    assert adapted.x.data[0] == pytest.approx(10.0)
    frequency = adapted.x.meta["acquisition_frequency"]
    assert float(frequency.magnitude) == pytest.approx(400.0)
    assert str(frequency.units) == "MHz"


@pytest.mark.parametrize("section", ["procs", "acqus"])
def test_malformed_metadata_containers_raise_adapter_error(section):
    result = _result()
    result.metadata[section] = None

    with pytest.raises(ReadResultAdapterError, match="metadata"):
        read_result_to_nddataset(
            result,
            normalization="spectrochempy-historical"
            if section == "acqus"
            else "decoded",
        )


@pytest.mark.parametrize("rg", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_historical_policy_rejects_invalid_rg(rg):
    result = _result()
    result.metadata["acqus"]["RG"] = rg

    with pytest.raises(ReadResultAdapterError, match="acquisition RG"):
        read_result_to_nddataset(result, normalization="spectrochempy-historical")


def test_decoded_policy_does_not_require_rg():
    result = _result()
    result.metadata["acqus"].pop("RG")

    adapted = read_result_to_nddataset(result, normalization="decoded")

    np.testing.assert_array_equal(adapted.data.real, result.components[0].data)
