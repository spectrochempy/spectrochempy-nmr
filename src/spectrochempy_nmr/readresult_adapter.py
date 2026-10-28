"""Internal adapter from nmrglue-ng ``ReadResult`` to ``NDDataset``.

The adapter is deliberately structural: importing the plugin does not require
nmrglue-ng. A compatible ReadResult-like object is accepted at call time.
"""

from __future__ import annotations

from copy import deepcopy
import math
from numbers import Real
from typing import Any

import numpy as np


class ReadResultAdapterError(ValueError):
    """Raised when a ReadResult cannot be represented by this profile."""


def _diagnostic_dict(diagnostic: Any) -> dict[str, Any]:
    return {
        "code": diagnostic.code,
        "severity": diagnostic.severity,
        "message": diagnostic.message,
        "context": deepcopy(dict(diagnostic.context)),
    }


def _component_map(result: Any) -> dict[str, np.ndarray]:
    components = getattr(result, "components", None)
    if components is None:
        raise ReadResultAdapterError("ReadResult does not expose named components")
    mapped = {component.name: np.asarray(component.data) for component in components}
    if "1r" not in mapped:
        raise ReadResultAdapterError("Bruker processed 1D requires a 1r component")
    return mapped


def _finite_positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ReadResultAdapterError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ReadResultAdapterError(f"{name} must be a finite positive number")
    return value


def _frequency_mhz(value: Any, unit: Any) -> float:
    value = _finite_positive(value, "reference observation frequency")
    factors = {"hz": 1.0e-6, "khz": 1.0e-3, "mhz": 1.0, "ghz": 1.0e3}
    key = str(unit or "").lower()
    if key not in factors:
        raise ReadResultAdapterError(
            "Unsupported observation frequency unit for Hz-to-ppm conversion"
        )
    return value * factors[key]


def _axis_metadata(result: Any, *, axis_unit: str) -> tuple[np.ndarray, dict[str, Any]]:
    axes = getattr(result, "axes", ())
    if len(axes) != 1:
        raise ReadResultAdapterError("Bruker processed 1D requires exactly one axis")
    axis = axes[0]
    if axis.values is None:
        raise ReadResultAdapterError("ReadResult axis has no coordinate values")
    values = np.asarray(axis.values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ReadResultAdapterError("ReadResult axis must be a finite 1D array")

    reference = axis.reference
    metadata = {
        "source_unit": axis.unit,
        "direction": axis.direction,
        "label": axis.label,
        "reference": deepcopy(reference.__dict__) if reference is not None else None,
    }
    supported_units = {"Hz", "ppm"}
    if axis.unit not in supported_units or axis_unit not in supported_units:
        raise ReadResultAdapterError(
            f"Unsupported axis unit conversion from {axis.unit!r} to {axis_unit!r}"
        )
    if axis_unit == axis.unit:
        return values, metadata
    if axis.unit != "Hz" or axis_unit != "ppm" or reference is None:
        raise ReadResultAdapterError(
            f"Cannot convert ReadResult axis from {axis.unit!r} to {axis_unit!r}"
        )
    sf_mhz = _frequency_mhz(
        reference.observation_frequency, reference.observation_frequency_unit
    )
    metadata["observation_frequency_mhz"] = sf_mhz
    return values / sf_mhz, metadata


def _processed_reference(metadata: dict[str, Any]) -> dict[str, float]:
    procs = metadata.get("procs", {})
    if not isinstance(procs, dict):
        raise ReadResultAdapterError("processed metadata procs must be a mapping")
    values = {name: procs.get(name) for name in ("OFFSET", "SF", "SW_p")}
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ReadResultAdapterError(
                f"processed metadata {name} must be a finite numeric value"
            )
        if not math.isfinite(float(value)):
            raise ReadResultAdapterError(
                f"processed metadata {name} must be a finite numeric value"
            )
    if values["SF"] <= 0 or values["SW_p"] <= 0:
        raise ReadResultAdapterError("processed metadata SF and SW_p must be positive")
    return {name: float(value) for name, value in values.items()}


def read_result_to_nddataset(
    result: Any,
    *,
    normalization: str = "decoded",
    axis_unit: str = "ppm",
):
    """Adapt a Bruker processed 1D ReadResult to an NDDataset.

    Parameters
    ----------
    result : ReadResult-like
        Object exposing ``data``, named ``components``, ``axes``, ``metadata``,
        ``diagnostics``, ``applied_corrections``, ``format`` and ``source``.
    normalization : {"decoded", "spectrochempy-historical"}
        ``decoded`` preserves nmrglue-ng values. The historical mode divides by
        acquisition ``RG`` explicitly; it never applies ``NS`` implicitly.
    axis_unit : {"ppm", "Hz"}
        Output coordinate unit for the processed axis.
    """
    if getattr(result, "format", None) != "bruker-pdata":
        raise ReadResultAdapterError(
            "Only the Bruker processed 1D profile is supported"
        )
    if not isinstance(getattr(result, "data", None), np.ndarray):
        raise ReadResultAdapterError("ReadResult data must be an ndarray")
    if result.data.ndim != 1:
        raise ReadResultAdapterError("Only 1D processed data are supported")
    if normalization not in {"decoded", "spectrochempy-historical"}:
        raise ReadResultAdapterError(f"Unknown normalization policy: {normalization!r}")

    components = _component_map(result)
    real = np.asarray(components["1r"])
    if real.ndim != 1:
        raise ReadResultAdapterError("1r component must be one-dimensional")
    data = real.astype(np.result_type(real.dtype, np.float64), copy=True)
    adapter_diagnostics = []
    if "1i" in components:
        imaginary = np.asarray(components["1i"])
        if imaginary.shape != real.shape:
            raise ReadResultAdapterError("1r and 1i components have different shapes")
        data = data + 1j * imaginary.astype(data.dtype, copy=False)
    else:
        adapter_diagnostics.append(
            {
                "code": "component.missing",
                "severity": "warning",
                "message": "1i is absent; the adapted dataset remains real-valued.",
                "context": {"expected": ["1r", "1i"], "missing": ["1i"]},
            }
        )

    source_metadata = deepcopy(dict(getattr(result, "metadata", {})))
    processing_reference = _processed_reference(source_metadata)
    applied = deepcopy(dict(getattr(result, "applied_corrections", {})))
    normalization_record = {"policy": normalization}
    if normalization == "spectrochempy-historical":
        acqus = source_metadata.get("acqus", {})
        if not isinstance(acqus, dict):
            raise ReadResultAdapterError("acquisition metadata acqus must be a mapping")
        rg = _finite_positive(acqus.get("RG"), "acquisition RG")
        data = data / rg
        normalization_record["RG"] = rg
        normalization_record["NS"] = acqus.get("NS")

    values, axis_record = _axis_metadata(result, axis_unit=axis_unit)
    axis_record["processing_reference"] = processing_reference
    if values.size != data.size:
        raise ReadResultAdapterError("ReadResult axis and data lengths differ")

    from spectrochempy import Coord, NDDataset  # noqa: PLC0415
    from spectrochempy import ur  # noqa: PLC0415

    coord = Coord(values, units=axis_unit, title=axis_record.get("label"))
    if axis_record.get("observation_frequency_mhz") is not None:
        coord.meta["acquisition_frequency"] = (
            float(axis_record["observation_frequency_mhz"]) * ur.MHz
        )
    coord.meta["read_result_axis"] = axis_record

    diagnostics = [
        _diagnostic_dict(item) for item in getattr(result, "diagnostics", ())
    ] + adapter_diagnostics
    metadata = {
        "nmrglue_read_result": {
            "format": result.format,
            "source": str(result.source),
            "applied_corrections": applied,
            "diagnostics": diagnostics,
            "normalization": normalization_record,
        },
        "vendor_metadata": source_metadata,
    }
    return NDDataset(data, coordset=[coord], meta=metadata)
