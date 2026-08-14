import pytest
from fastapi.testclient import TestClient

import all_agent


@pytest.fixture()
def client():
    all_agent._reset_embedded_twin()
    all_agent._twin_base_cache.update(
        {"base": None, "snapshot": None, "checked_at": float("inf")}  # force embedded
    )
    all_agent._last_run.update({"final": None, "twin_source": None, "ts": 0.0})
    try:
        yield TestClient(all_agent.app)
    finally:
        all_agent._twin_base_cache.update({"base": None, "snapshot": None, "checked_at": 0.0})
        all_agent._last_run.update({"final": None, "twin_source": None, "ts": 0.0})
        all_agent._reset_embedded_twin()


def test_run_returns_three_plans(client):
    resp = client.post("/run")
    assert resp.status_code == 200
    body = resp.json()
    assert {p["id"] for p in body["candidate_plans"]} == {"A", "B", "C"}
    assert body["twin_source"] == "embedded"
    assert body["explanation"]["source"] in {"rules", "claude"}


def test_state_serves_trains(client):
    resp = client.get("/state")
    assert resp.status_code == 200
    assert resp.json()["trains"]


def test_run_stream_rejects_unknown_condition(client):
    resp = client.get("/run-stream", params={"weather_track": "T05", "condition": "SNOW"})
    assert resp.status_code == 400


def test_inject_unknown_track_404s(client):
    resp = client.post("/simulate-track-failure/NOPE")
    assert resp.status_code == 404


def test_apply_plan_executes_the_approved_plan(client):
    ran = client.post("/run").json()
    approved = ran["recommended_action"]["id"]
    resp = client.post("/apply-plan", params={"plan_id": approved})
    assert resp.status_code == 200
    body = resp.json()
    assert body["executed_plan"] == approved
    # Apply must answer from the template, never a slow LLM call
    assert body["explanation"]["source"] == "rules"


def test_apply_unknown_plan_404s(client):
    client.post("/run")
    resp = client.post("/apply-plan", params={"plan_id": "Z"})
    assert resp.status_code in {400, 404, 422}


def test_reset_succeeds(client):
    assert client.post("/reset").status_code == 200
