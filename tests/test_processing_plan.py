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


def _state(**overrides):
    state = {
        "domain": "time",
        "source_kind": "fid",
        "current_size": 2048,
        "iscomplex": True,
    }
    state.update(overrides)
    return state


def test_topspin_1d_plan_is_declarative_and_ordered():
    processing = _profile()
    plan = build_topspin_1d_plan(
        processing=processing,
        acquisition={"TD": 1024, "SW_h": 8000.0, "SFO1": 400.0},
        initial_state=_state(),
        source={"vendor": "bruker", "path": "synthetic/fid"},
    )

    assert plan.status == "partial"
    assert [operation.name for operation in plan.operations] == ["em", "zf_size", "fft"]
    assert plan.operations[0].parameters == {"lb": {"value": 5.0, "units": "Hz"}}
    assert plan.operations[-1].parameters == {"size": 2048}
    assert plan.processing == processing
    assert plan.blockers == ()
    assert any("FT_mod" in item for item in plan.unsupported)

    payload = plan.to_dict()
    restored = json.loads(plan.to_json())
    assert restored == payload
    assert "operations" in payload
    assert "execute" not in payload
    assert "NMRProcessingPlan(status=partial" in str(plan)


def test_topspin_vendor_profile_parameters_and_provenance_are_preserved():
    profile = {
        "vendor": "bruker",
        "source_files": ["procs"],
        "parameters": _profile(),
    }
    plan = build_topspin_1d_plan(
        processing=profile,
        initial_state=_state(),
    )

    assert plan.status == "partial"
    assert plan.source["processing_source_files"] == ["procs"]
    assert plan.processing["WDW"] == 1
    assert plan.processing["SI"] == 2048


def test_unknown_topspin_window_blocks_without_fallback():
    plan = build_topspin_1d_plan(
        processing=_profile(WDW=99),
        initial_state=_state(),
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
        initial_state=_state(),
    )

    assert plan.status == "blocked"
    assert any("LB" in item for item in plan.blockers)
    assert not any(operation.name == "em" for operation in plan.operations)


def test_non_integral_vendor_sizes_are_blocked_not_truncated():
    plan = build_topspin_1d_plan(
        processing=_profile(WDW=1.5, SI=2048.5),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "blocked"
    assert any("WDW" in item for item in plan.blockers)
    assert any("SI" in item for item in plan.blockers)


def test_unknown_vendor_processing_parameters_are_reported():
    plan = build_topspin_1d_plan(
        processing=_profile(FOO=1),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "partial"
    assert any("FOO" in item for item in plan.unsupported)


def test_frequency_domain_input_is_blocked_before_fft_plan():
    plan = build_topspin_1d_plan(
        processing=_profile(),
        initial_state=_state(domain="frequency", source_kind="processed_1d"),
    )

    assert plan.status == "blocked"
    assert any("time-domain" in item for item in plan.blockers)
    assert plan.operations == ()


def test_missing_initial_state_returns_blocked_plan_not_exception():
    plan = build_topspin_1d_plan(processing=_profile(), initial_state=None)

    assert plan.status == "blocked"
    assert any("time-domain" in item for item in plan.blockers)
    assert any("Effective input size" in item for item in plan.blockers)


def test_effective_size_can_come_from_tdeff_and_td_complexity():
    from_tdeff = build_topspin_1d_plan(
        processing=_profile(),
        acquisition={"TDeff": 2048},
        initial_state={"domain": "time", "source_kind": "fid", "iscomplex": True},
        source={"path": "synthetic/fid"},
    )
    from_td = build_topspin_1d_plan(
        processing=_profile(),
        acquisition={"TD": 4096},
        initial_state={"domain": "time", "source_kind": "fid", "iscomplex": True},
        source={"path": "synthetic/fid"},
    )

    assert from_tdeff.status == "partial"
    assert from_td.status == "partial"
    assert [operation.name for operation in from_tdeff.operations] == [
        "em",
        "zf_size",
        "fft",
    ]
    assert [operation.name for operation in from_td.operations] == [
        "em",
        "zf_size",
        "fft",
    ]


def test_malformed_td_is_blocked_without_raising():
    plan = build_topspin_1d_plan(
        processing=_profile(),
        acquisition={"TD": "not-a-number"},
        initial_state={"domain": "time", "source_kind": "fid", "iscomplex": True},
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "blocked"
    assert any("TD" in item for item in plan.blockers)


def test_nonfinite_window_parameters_are_blocked():
    plan = build_topspin_1d_plan(
        processing=_profile(LB=float("nan")),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "blocked"
    assert any("LB" in item and "finite" in item for item in plan.blockers)


def test_topspin_phase_parameters_are_preserved_but_not_replayed():
    profile = _profile(PHC0=12.5, PHC1=-3.0)
    plan = build_topspin_1d_plan(
        processing=profile,
        initial_state=_state(),
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
        initial_state=_state(),
    )
    sp = build_topspin_1d_plan(
        processing=_profile(WDW=3, SSB=2.0),
        initial_state=_state(),
    )

    assert gm.status == "blocked"
    assert gm.operations == ()
    assert any("GB" in item for item in gm.unsupported)
    assert sp.status == "partial"
    assert [operation.name for operation in sp.operations] == ["zf_size", "fft"]
    assert any("power" in item for item in sp.unsupported)


def test_gm_valid_gb_remains_partial_without_unit_conversion():
    plan = build_topspin_1d_plan(
        processing=_profile(WDW=2, GB=0.5),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "partial"
    assert not any(operation.name == "gm" for operation in plan.operations)
    assert any("dimensionless" in item for item in plan.unsupported)


def test_qsine_uses_squared_power_and_ssb_zero_normalization():
    plan = build_topspin_1d_plan(
        processing=_profile(WDW="qsine", SSB=0.0),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "partial"
    assert plan.operations[0].parameters == {"ssb": 1.0, "pow": 2}
    assert plan.processing["SSB"] == 0.0


def test_nonfinite_gb_and_ssb_are_blocked():
    gb_plan = build_topspin_1d_plan(
        processing=_profile(WDW=2, GB=float("inf")),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )
    ssb_plan = build_topspin_1d_plan(
        processing=_profile(WDW="sine", SSB=float("nan")),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert gb_plan.status == "blocked"
    assert ssb_plan.status == "blocked"
    assert '"nan"' in ssb_plan.to_json().lower()
    assert '"inf"' in gb_plan.to_json().lower()


def test_si_smaller_than_effective_input_blocks_truncation():
    plan = build_topspin_1d_plan(
        processing=_profile(SI=1024),
        acquisition={"TDeff": 2048},
        initial_state={"domain": "time", "source_kind": "fid", "iscomplex": True},
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "blocked"
    assert any("truncation" in item for item in plan.blockers)
    assert not any(operation.name == "zf_size" for operation in plan.operations)


def test_topspin_sine_values_below_two_normalize_without_forwarding_zero():
    sine_zero = build_topspin_1d_plan(
        processing=_profile(WDW="sine", SSB=0.0),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )
    sine_one = build_topspin_1d_plan(
        processing=_profile(WDW="sine", SSB=1.0),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )
    sine_two = build_topspin_1d_plan(
        processing=_profile(WDW="sine", SSB=2.0),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert sine_zero.status == "partial"
    assert sine_zero.operations[0].parameters["ssb"] == 1.0
    assert sine_one.operations[0].parameters["ssb"] == 1.0
    assert sine_two.operations[0].parameters["ssb"] == 2.0
    assert sine_zero.processing["SSB"] == 0.0
    assert any("normalized" in item for item in sine_zero.assumptions)


def test_negative_sine_shift_is_blocked():
    plan = build_topspin_1d_plan(
        processing=_profile(WDW="sine", SSB=-1.0),
        initial_state=_state(),
        source={"path": "synthetic/fid"},
    )

    assert plan.status == "blocked"
    assert any("negative" in item for item in plan.blockers)


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
