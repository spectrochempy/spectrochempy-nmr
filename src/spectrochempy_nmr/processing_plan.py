"""Declarative vendor-processing plans.

The first contract is deliberately independent of :class:`NDDataset`: it
describes what a vendor profile means without executing any operation.
"""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
import math
from numbers import Real
from typing import Any
from typing import Mapping


def _plain(value: Any) -> Any:
    """Return a JSON-friendly copy without depending on a units package."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "magnitude") and hasattr(value, "units"):
        return {
            "value": _plain(value.magnitude),
            "units": str(value.units),
        }
    if hasattr(value, "item"):
        try:
            return _plain(value.item())
        except ValueError:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass(frozen=True)
class PlanOperation:
    """One proposed SpectroChemPy operation, without execution."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    source_parameters: tuple[str, ...] = ()
    convention: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _plain(asdict(self))


@dataclass(frozen=True)
class NMRProcessingPlan:
    """Inspectable, non-executable interpretation of a vendor recipe."""

    source: dict[str, Any]
    acquisition: dict[str, Any]
    processing: dict[str, Any]
    initial_state: dict[str, Any]
    operations: tuple[PlanOperation, ...] = ()
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    unsupported: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    status: str = "blocked"

    def __post_init__(self) -> None:
        if self.status not in {"validated", "partial", "blocked"}:
            raise ValueError(f"Unknown processing-plan status: {self.status!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize the plan to plain JSON-compatible values."""
        result = {
            "source": self.source,
            "acquisition": self.acquisition,
            "processing": self.processing,
            "initial_state": self.initial_state,
            "operations": [operation.to_dict() for operation in self.operations],
            "assumptions": self.assumptions,
            "limitations": self.limitations,
            "unsupported": self.unsupported,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "status": self.status,
        }
        return _plain(result)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the plan deterministically as JSON."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True) + "\n"

    def __str__(self) -> str:
        operations = " -> ".join(operation.name for operation in self.operations)
        return (
            f"NMRProcessingPlan(status={self.status}, "
            f"vendor={self.source.get('vendor', 'unknown')}, "
            f"operations=[{operations or 'none'}])"
        )


def _parameter(parameters: Mapping[str, Any], name: str) -> Any:
    """Read a vendor parameter without inventing a missing value."""
    return parameters.get(name)


def _integral_parameter(value: Any, name: str, blockers: list[str]) -> int | None:
    """Validate an integer vendor field without silently coercing it."""
    if isinstance(value, bool) or not isinstance(value, Real):
        blockers.append(f"TopSpin processing parameter {name} must be an integer.")
        return None
    if not math.isfinite(float(value)) or int(value) != value:
        blockers.append(f"TopSpin processing parameter {name} must be an integer.")
        return None
    return int(value)


def build_topspin_1d_plan(
    *,
    processing: Mapping[str, Any] | None,
    acquisition: Mapping[str, Any] | None = None,
    initial_state: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> NMRProcessingPlan:
    """Interpret a TopSpin 1D profile without executing it.

    Parameters are intentionally mappings so the function can consume the
    reader's ``vendor_profile`` while remaining independent of ``NDDataset``.
    """
    source_data = {"vendor": "bruker", "format": "topspin", **(source or {})}
    acquisition_data = dict(acquisition or {})
    profile = dict(processing or {})
    if "parameters" in profile:
        profile_parameters = profile.pop("parameters")
        processing_data = dict(profile_parameters or {})
        source_data.update(
            {
                "vendor": profile.pop("vendor", source_data["vendor"]),
                "processing_source_files": profile.pop("source_files", []),
            }
        )
    else:
        processing_data = profile
    state_data = dict(initial_state or {})
    operations: list[PlanOperation] = []
    assumptions: list[str] = []
    limitations: list[str] = []
    unsupported: list[str] = []
    warnings: list[str] = []
    blockers: list[str] = []
    known_parameters = {
        "WDW",
        "LB",
        "GB",
        "SSB",
        "SI",
        "FT_mod",
        "PHC0",
        "PHC1",
        "NC_proc",
    }

    if source_data.get("vendor") not in {"bruker", "topspin"}:
        blockers.append(
            "This first plan builder only accepts a TopSpin/Bruker profile."
        )
    if not processing:
        blockers.append("TopSpin processing parameters are missing.")
    if state_data.get("domain") != "time":
        blockers.append("The initial dataset state must explicitly be time-domain.")
    if not (
        source_data.get("path")
        or source_data.get("provenance")
        or source_data.get("processing_source_files")
    ):
        warnings.append(
            "Source provenance is incomplete; no path or source files were supplied."
        )

    for name in sorted(set(processing_data) - known_parameters):
        unsupported.append(f"TopSpin processing parameter {name} is not interpreted.")

    wdw = _parameter(processing_data, "WDW")
    si = _parameter(processing_data, "SI")
    if wdw is None:
        blockers.append("TopSpin processing parameter WDW is missing.")
    else:
        wdw = _integral_parameter(wdw, "WDW", blockers)
    if si is None:
        blockers.append("TopSpin processing parameter SI is missing.")
    else:
        si = _integral_parameter(si, "SI", blockers)
        if si is not None and si <= 0:
            blockers.append("TopSpin processing parameter SI must be positive.")

    if not blockers:
        if wdw == 1:
            lb = _parameter(processing_data, "LB")
            if lb is None:
                blockers.append("EM interpretation requires TopSpin parameter LB.")
            else:
                operations.append(
                    PlanOperation(
                        "em",
                        {"lb": {"value": lb, "units": "Hz"}},
                        ("WDW", "LB"),
                        "TopSpin WDW=1 interpreted as exponential multiplication.",
                    )
                )
        elif wdw == 2:
            lb = _parameter(processing_data, "LB")
            gb = _parameter(processing_data, "GB")
            if lb is None or gb is None:
                blockers.append("GM interpretation requires both LB and GB.")
            else:
                operations.append(
                    PlanOperation(
                        "gm",
                        {
                            "lb": {"value": lb, "units": "Hz"},
                            "gb": {"value": gb, "units": "Hz"},
                        },
                        ("WDW", "LB", "GB"),
                        "TopSpin WDW=2 interpreted as Lorentz-to-Gauss multiplication.",
                    )
                )
        elif wdw == 3:
            ssb = _parameter(processing_data, "SSB")
            if ssb is None:
                blockers.append("SP interpretation requires TopSpin parameter SSB.")
            else:
                operations.append(
                    PlanOperation(
                        "sp",
                        {"ssb": ssb},
                        ("WDW", "SSB"),
                        "TopSpin WDW=3 interpreted as sine-bell multiplication.",
                    )
                )
                unsupported.append(
                    "TopSpin sine-bell power is not independently sourced."
                )
                warnings.append(
                    "SP plan is partial until the TopSpin power convention is explicit."
                )
        else:
            unsupported.append(f"TopSpin WDW={wdw} has no validated core mapping.")
            blockers.append(f"Unsupported TopSpin window code WDW={wdw}.")

        if not blockers:
            operations.append(
                PlanOperation(
                    "zf_size",
                    {"size": si},
                    ("SI",),
                    "TopSpin SI is interpreted as the requested processed size.",
                )
            )
            fft_sources = ["SI"]
            if _parameter(processing_data, "FT_mod") is not None:
                fft_sources.append("FT_mod")
            operations.append(
                PlanOperation(
                    "fft",
                    {"size": si},
                    tuple(fft_sources),
                    "FFT direction is the time-to-frequency operation; FT_mod is retained, not replayed.",
                )
            )

    phc0 = _parameter(processing_data, "PHC0")
    phc1 = _parameter(processing_data, "PHC1")
    if phc0 is not None or phc1 is not None:
        unsupported.extend(
            name
            for name, value in (("PHC0", phc0), ("PHC1", phc1))
            if value is not None
        )
        warnings.append(
            "TopSpin phase parameters are preserved but not replayed by this first plan."
        )

    if _parameter(processing_data, "FT_mod") is not None:
        limitations.append(
            "FT_mod is recorded as provenance; it is not silently converted to an FFT variant."
        )
    if _parameter(processing_data, "NC_proc") is not None:
        unsupported.append(
            "NC_proc amplitude scaling is not included in the first plan."
        )

    assumptions.extend(
        (
            "The input is interpreted as a 1D time-domain FID.",
            "TopSpin WDW codes are mapped only through independently validated core windows.",
            "SI is interpreted as the requested processed size, not as a data mutation.",
        )
    )

    if blockers:
        status = "blocked"
    elif warnings or unsupported:
        status = "partial"
    else:
        status = "validated"
    return NMRProcessingPlan(
        source=source_data,
        acquisition=acquisition_data,
        processing=processing_data,
        initial_state=state_data,
        operations=tuple(operations),
        assumptions=tuple(assumptions),
        limitations=tuple(limitations),
        unsupported=tuple(unsupported),
        warnings=tuple(warnings),
        blockers=tuple(blockers),
        status=status,
    )
