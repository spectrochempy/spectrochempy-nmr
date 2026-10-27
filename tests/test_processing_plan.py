"""Tests for the declarative vendor-processing plan contract."""

from __future__ import annotations

import json

from spectrochempy_nmr.processing_plan import build_topspin_1d_plan


def _profile(**overrides):
    profile = {
        "WDW": 1,
        "LB": 5.0,
        "SI": 2048,
        "FT_mod": 6,
    }
    profile.update(overrides)
    return profile


def test_topspin_1d_plan_is_declarative_and_ordered():
    processing = _profile()
    plan = build_topspin_1d_plan(
        processing=processing,
        acquisition={"TD": 1024, "SW_h": 8000.0, "SFO1": 400.0},
        initial_state={"domain": "time", "source_kind": "fid"},
        source={"vendor": "bruker", "path": "synthetic/fid"},
    )

    assert plan.status == "validated"
    assert [operation.name for operation in plan.operations] == ["em", "zf_size", "fft"]
    assert plan.operations[0].parameters == {"lb": {"value": 5.0, "units": "Hz"}}
    assert plan.operations[-1].parameters == {"size": 2048}
    assert plan.processing == processing
    assert plan.blockers == ()
    assert "FT_mod" in plan.limitations[0]

    payload = plan.to_dict()
    restored = json.loads(plan.to_json())
    assert restored == payload
    assert "operations" in payload
    assert "execute" not in payload
    assert "NMRProcessingPlan(status=validated" in str(plan)


def test_topspin_vendor_profile_parameters_and_provenance_are_preserved():
    profile = {
        "vendor": "bruker",
        "source_files": ["procs"],
        "parameters": _profile(),
    }
    plan = build_topspin_1d_plan(
        processing=profile,
        initial_state={"domain": "time", "source_kind": "fid"},
    )

    assert plan.status == "validated"
    assert plan.source["processing_source_files"] == ["procs"]
    assert plan.processing["WDW"] == 1
    assert plan.processing["SI"] == 2048


def test_unknown_topspin_window_blocks_without_fallback():
    plan = build_topspin_1d_plan(
        processing=_profile(WDW=99),
        initial_state={"domain": "time", "source_kind": "fid"},
    )

    assert plan.status == "blocked"
    assert any("WDW=99" in item for item in plan.blockers)
    assert plan.operations == ()
    assert any("WDW=99" in item for item in plan.unsupported)


def test_missing_required_parameter_blocks_without_inventing_defaults():
    profile = _profile()
    del profile["LB"]
    plan = build_topspin_1d_plan(
        processing=profile,
        initial_state={"domain": "time", "source_kind": "fid"},
    )

    assert plan.status == "blocked"
    assert any("LB" in item for item in plan.blockers)
    assert not any(operation.name == "em" for operation in plan.operations)


def test_non_integral_vendor_sizes_are_blocked_not_truncated():
    plan = build_topspin_1d_plan(
        processing=_profile(WDW=1.5, SI=2048.5),
        initial_state={"domain": "time", "source_kind": "fid"},
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "blocked"
    assert any("WDW" in item for item in plan.blockers)
    assert any("SI" in item for item in plan.blockers)


def test_unknown_vendor_processing_parameters_are_reported():
    plan = build_topspin_1d_plan(
        processing=_profile(FOO=1),
        initial_state={"domain": "time", "source_kind": "fid"},
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "partial"
    assert any("FOO" in item for item in plan.unsupported)


def test_frequency_domain_input_is_blocked_before_fft_plan():
    plan = build_topspin_1d_plan(
        processing=_profile(),
        initial_state={"domain": "frequency", "source_kind": "processed_1d"},
    )

    assert plan.status == "blocked"
    assert any("time-domain" in item for item in plan.blockers)
    assert plan.operations == ()


def test_topspin_phase_parameters_are_preserved_but_not_replayed():
    profile = _profile(PHC0=12.5, PHC1=-3.0)
    plan = build_topspin_1d_plan(
        processing=profile,
        initial_state={"domain": "time", "source_kind": "fid"},
    )

    assert plan.status == "partial"
    assert "PHC0" in plan.unsupported
    assert "PHC1" in plan.unsupported
    assert not any(operation.name == "pk" for operation in plan.operations)
    assert plan.processing["PHC0"] == 12.5
    assert plan.processing["PHC1"] == -3.0


def test_gm_and_sp_mappings_report_unvalidated_conventions():
    gm = build_topspin_1d_plan(
        processing=_profile(WDW=2, GB=3.0),
        initial_state={"domain": "time", "source_kind": "fid"},
    )
    sp = build_topspin_1d_plan(
        processing=_profile(WDW=3, SSB=2.0),
        initial_state={"domain": "time", "source_kind": "fid"},
    )

    assert [operation.name for operation in gm.operations] == ["gm", "zf_size", "fft"]
    assert gm.operations[0].parameters["gb"] == {"value": 3.0, "units": "Hz"}
    assert sp.status == "partial"
    assert [operation.name for operation in sp.operations] == ["sp", "zf_size", "fft"]
    assert any("power" in item for item in sp.unsupported)


def test_plan_serialization_does_not_mutate_source_mappings():
    processing = _profile()
    acquisition = {"TD": 1024}
    initial = {"domain": "time", "source_kind": "fid"}
    plan = build_topspin_1d_plan(
        processing=processing,
        acquisition=acquisition,
        initial_state=initial,
    )

    _ = plan.to_json()
    assert processing == _profile()
    assert acquisition == {"TD": 1024}
    assert initial == {"domain": "time", "source_kind": "fid"}
