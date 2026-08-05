import pytest

import all_agent


def fresh_snapshot():
    all_agent._reset_embedded_twin()
    return all_agent._embedded_snapshot()


def run_pipeline(snapshot, extra_log=None):
    return all_agent.pipeline.invoke(all_agent._initial_state(snapshot, extra_log=extra_log))


def test_baseline_run_produces_three_scored_plans():
    final = run_pipeline(fresh_snapshot())
    resp = all_agent._format_response(final, "embedded")
    assert {p["id"] for p in resp["candidate_plans"]} == {"A", "B", "C"}
    for p in resp["candidate_plans"]:
        assert 0 <= p["score"] <= 100
    assert resp["recommended_action"]["score"] == max(p["score"] for p in resp["candidate_plans"])


def test_track_failure_triggers_reroutes_and_penalizes_inaction():
    snapshot = fresh_snapshot()
    snapshot["tracks"]["T23"]["health"] = 0.0
    snapshot["tracks"]["T23"]["status"] = "CLOSED"
    final = run_pipeline(snapshot)
    resp = all_agent._format_response(final, "embedded")

    assert "T23" in resp["failed_tracks"]
    # Trains crossing Nagpur-Raipur must be rerouted or held
    assert resp["rerouted_trains"] or resp["held_trains"]
    # Do-nothing plan C must be heavily penalized (stranded trains)
    by_id = {p["id"]: p for p in resp["candidate_plans"]}
    assert by_id["C"]["score"] < by_id["A"]["score"]
    assert by_id["C"]["score"] < by_id["B"]["score"]
    assert resp["recommended_action"]["id"] != "C"


def test_reroutes_avoid_the_failed_edge():
    snapshot = fresh_snapshot()
    snapshot["tracks"]["T23"]["health"] = 0.0
    snapshot["tracks"]["T23"]["status"] = "CLOSED"
    final = run_pipeline(snapshot)
    for s in final["routing_strategies"]:
        if not s["strategy_id"].startswith("R_REROUTE_"):
            continue
        route = s["new_route"]
        for i in range(len(route) - 1):
            assert {route[i], route[i + 1]} != {"NAGPUR_JUNCT", "RAIPUR_JUNCT"}


def test_routing_never_emits_degenerate_single_station_reroute():
    # A train sitting at its terminus while its route crosses a closed track
    # must not produce a 1-station reroute (which would freeze it forever).
    snapshot = fresh_snapshot()
    snapshot["tracks"]["T23"]["health"] = 0.0
    snapshot["tracks"]["T23"]["status"] = "CLOSED"
    for train in snapshot["trains"].values():
        route = train["route"]
        crosses = any(
            {route[i], route[i + 1]} == {"NAGPUR_JUNCT", "RAIPUR_JUNCT"}
            for i in range(len(route) - 1)
        )
        if crosses:
            train["current_station"] = route[-1]  # parked at the terminus
    final = run_pipeline(snapshot)
    for s in final["routing_strategies"]:
        if s["strategy_id"].startswith("R_REROUTE_"):
            assert len(s["new_route"]) >= 2


def test_storm_activates_emergency_weather_protocol():
    snapshot = fresh_snapshot()
    snapshot["weather"] = {"T05": "STORM"}
    final = run_pipeline(snapshot)
    assert final["weather_risk"]["T05"] == 85.0
    assert final["weather_strategies"][0]["strategy_id"] == "W1"


def test_mcdm_normalisation_orders_by_weighted_cost():
    values = [10.0, 20.0, 30.0]
    normed = all_agent._minmax(values)
    assert normed == [0.0, 0.5, 1.0]
    assert all_agent._minmax([5.0, 5.0]) == [0.0, 0.0]


def test_plan_payload_carries_preview_data():
    snapshot = fresh_snapshot()
    snapshot["tracks"]["T23"]["health"] = 0.0
    snapshot["tracks"]["T23"]["status"] = "CLOSED"
    final = run_pipeline(snapshot)
    resp = all_agent._format_response(final, "embedded")
    plan_a = next(p for p in resp["candidate_plans"] if p["id"] == "A")
    assert plan_a["failed_tracks"] == ["T23"]
    plan_c = next(p for p in resp["candidate_plans"] if p["id"] == "C")
    assert plan_c["rerouted_trains"] == []


def test_template_explanation_mentions_selected_plan():
    snapshot = fresh_snapshot()
    snapshot["tracks"]["T23"]["health"] = 0.0
    snapshot["tracks"]["T23"]["status"] = "CLOSED"
    final = run_pipeline(snapshot)
    text = all_agent._template_explanation(final)
    assert final["best_plan"]["plan_name"] in text
    assert "/100" in text


def test_execution_log_has_structured_entries():
    final = run_pipeline(fresh_snapshot())
    resp = all_agent._format_response(final, "embedded")
    assert len(resp["execution_log"]) >= 5
    for entry in resp["execution_log"]:
        assert set(entry) == {"t", "source", "message"}
    # Newest first: Master's selection is the first entry
    assert resp["execution_log"][0]["source"] == "Master"


def test_embedded_twin_mutations_persist_across_snapshots():
    all_agent._reset_embedded_twin()
    all_agent._twin_base_cache["base"] = None
    all_agent._twin_base_cache["checked_at"] = float("inf")  # force embedded
    try:
        all_agent._twin_close_track("T23")
        snap = all_agent._embedded_snapshot()
        assert snap["tracks"]["T23"]["status"] == "CLOSED"
        all_agent._twin_reset()
        snap = all_agent._embedded_snapshot()
        assert snap["tracks"]["T23"]["status"] != "CLOSED"
    finally:
        all_agent._twin_base_cache["checked_at"] = 0.0
        all_agent._reset_embedded_twin()
