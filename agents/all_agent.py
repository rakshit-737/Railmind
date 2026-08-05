"""
all_agent.py  —  RailMind agent orchestrator (FastAPI + LangGraph)
────────────────────────────────────────────────────────────────
Flow:
  Digital Twin snapshot
    → WeatherAgent   (risk scores per track)
    → TrackAgent     (strategies: close / restrict / monitor)
    → SignalAgent    (strategies: RED / YELLOW / GREEN per section)
    → RoutingAgent   (strategies: reroute options per affected train)
    → PlannerAgent   (combines strategies → Plan A / Plan B / Plan C)
    → SimulationNode (clones twin, applies each plan, scores locally)
    → MasterAgent    (MCDM normalised ranking → best plan)

Twin sources (first reachable wins):
  1. TWIN_BASE_URL env var
  2. Local Django twin  (http://127.0.0.1:8000)
  3. Hosted twin        (https://railmind-backend-4pep.onrender.com)
  4. Embedded twin      (railmind package, fully offline)

Agent contract (every specialist):
  Input  : RailState (read-only)
  Output : strategies: [{ strategy_id, ... }] appended to state

Master contract:
  Input  : simulation_results[]
  Output : { selected_plan, score, ranking } via min-max normalised MCDM
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, TypedDict

import networkx as nx
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.graph import END, StateGraph

try:
    import anthropic
except ImportError:
    anthropic = None

# Make the repo root importable so the embedded twin fallback works offline.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

app = FastAPI(
    title="RailMind Agents",
    description="Multi-agent decision pipeline for railway operations",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

TWIN_CANDIDATES = [
    os.environ.get("TWIN_BASE_URL"),
    "http://127.0.0.1:8000",
    "https://railmind-backend-4pep.onrender.com",
]

PLAN_NAMES = {
    "A": "Plan A — Safety First",
    "B": "Plan B — Balanced Response",
    "C": "Plan C — Minimal Intervention",
}

WEATHER_CONDITION_RISK = {"CLEAR": 5.0, "RAIN": 55.0, "FOG": 65.0, "STORM": 85.0}

MCDM_WEIGHTS = {
    "delay":            0.35,
    "risk":             0.40,
    "passenger_impact": 0.15,
    "congestion":       0.10,
}


# ─────────────────────────────────────────────────────────────────────────────
# TWIN ACCESS
# The orchestrator talks to the first reachable twin API; when none is
# reachable it falls back to an embedded in-process twin, so every endpoint
# (including injections, apply-plan and reset) works fully offline.
# ─────────────────────────────────────────────────────────────────────────────

_embedded_twin = None
_embedded_lock = threading.Lock()

# Cache of the reachable remote base (None = use embedded), refreshed periodically
_twin_base_cache: dict = {"base": None, "checked_at": 0.0}
_TWIN_BASE_TTL = 20.0


def _get_embedded_twin():
    global _embedded_twin
    from railmind.data_loader import load_railway
    from railmind.graph import build_network_from_data
    from railmind.twin import DigitalTwin

    with _embedded_lock:
        if _embedded_twin is None:
            stations, tracks = load_railway("India")
            graph = build_network_from_data(stations, tracks)
            twin = DigitalTwin(graph)
            twin.seed_trains(5)
            _embedded_twin = twin
        return _embedded_twin


def _reset_embedded_twin():
    global _embedded_twin
    with _embedded_lock:
        _embedded_twin = None


def _embedded_snapshot() -> dict:
    """Snapshot from the in-process twin — no network needed. Ticks the twin
    so trains move between reads even with no Django server running."""
    twin = _get_embedded_twin()
    twin.maybe_tick()
    dump = twin.get_state()
    dump["graph"] = {
        "nodes": list(twin.graph.graph.nodes()),
        "edges": list(twin.graph.graph.edges()),
    }
    return dump


def _twin_base() -> str | None:
    """Return the first reachable remote twin base URL, or None for embedded."""
    now = time.time()
    if now - _twin_base_cache["checked_at"] < _TWIN_BASE_TTL:
        return _twin_base_cache["base"]

    base_found = None
    for base in TWIN_CANDIDATES:
        if not base:
            continue
        try:
            resp = requests.get(f"{base}/api/state/", timeout=3)
            resp.raise_for_status()
            base_found = base
            break
        except requests.RequestException:
            continue

    _twin_base_cache["base"] = base_found
    _twin_base_cache["checked_at"] = now
    return base_found


def fetch_snapshot() -> tuple[dict, str]:
    """Return (snapshot, source_label) from the live twin (remote or embedded)."""
    base = _twin_base()
    if base:
        try:
            resp = requests.get(f"{base}/api/state/", timeout=4)
            resp.raise_for_status()
            return resp.json(), base
        except requests.RequestException:
            _twin_base_cache["base"] = None
    return _embedded_snapshot(), "embedded"


def _twin_close_track(track_id: str) -> None:
    base = _twin_base()
    if base:
        requests.post(f"{base}/api/track/close/", json={"track_id": track_id}, timeout=4).raise_for_status()
    else:
        _get_embedded_twin().close_track(track_id)


def _twin_set_weather(track_id: str, condition: str) -> None:
    base = _twin_base()
    if base:
        requests.post(f"{base}/api/weather/set/", json={"track_id": track_id, "condition": condition}, timeout=4).raise_for_status()
    else:
        _get_embedded_twin().set_weather(track_id, condition)


def _twin_reroute_train(train_id: str, route: List[str]) -> None:
    base = _twin_base()
    if base:
        requests.post(f"{base}/api/train/reroute/", json={"train_id": train_id, "route": route}, timeout=4).raise_for_status()
    else:
        _get_embedded_twin().reroute_train(train_id, route)


def _twin_reset() -> None:
    base = _twin_base()
    if base:
        requests.post(f"{base}/api/reset/", timeout=6).raise_for_status()
    else:
        _reset_embedded_twin()


# ─────────────────────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────────────────────

class RailState(TypedDict):
    snapshot:               dict            # /api/state/ response (or embedded)
    graph:                  nx.Graph        # station network

    # ── specialist agent outputs (read-only after each agent) ──
    weather_risk:           Dict[str, float]        # track_id → 0-100
    weather_strategies:     List[dict]
    track_strategies:       List[dict]
    signal_strategies:      List[dict]
    routing_strategies:     List[dict]

    # ── orchestration ──
    plans:                  List[dict]              # Planner output
    simulation_results:     List[dict]              # Simulation output
    best_plan:              dict                    # Master output

    log:                    List[dict]              # {t, source, message}


def _log(state: RailState, source: str, message: str) -> None:
    state["log"].append({"t": time.strftime("%H:%M:%S"), "source": source, "message": message})


def _strategy(sid: str, name: str, confidence: float, actions: List[str]) -> dict:
    """Standard strategy object used by every specialist agent."""
    return {
        "strategy_id": sid,
        "strategy_name": name,
        "confidence": confidence,
        "actions": actions,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 1 — WEATHER
# Reads  : snapshot.weather  (dict track_id→condition, or list of sensor dicts)
# Outputs: weather_risk, weather_strategies[]
# Rule   : weighted formula — no ML, no API
# ─────────────────────────────────────────────────────────────────────────────

def weather_node(state: RailState) -> RailState:
    weather = state["snapshot"].get("weather", {})
    risk_map: Dict[str, float] = {}

    if isinstance(weather, dict):
        # Twin format: {track_id: "STORM" | "RAIN" | "FOG" | "CLEAR"}
        for track_id, condition in weather.items():
            cond = condition.value if hasattr(condition, "value") else str(condition)
            risk_map[track_id] = WEATHER_CONDITION_RISK.get(cond.upper(), 0.0)
    else:
        # Sensor format: [{track_id, rainfall, temperature}, ...]
        for w in weather:
            track_id    = w["track_id"]
            rainfall    = w.get("rainfall", 0)
            temperature = w.get("temperature", 25)
            risk = 100 * (
                0.7 * min(rainfall / 50, 1.0) +
                0.3 * max(min((temperature - 25) / 25, 1.0), 0.0)
            )
            risk_map[track_id] = round(risk, 1)

    state["weather_risk"] = risk_map

    high_tracks   = [t for t, r in risk_map.items() if r > 70]
    medium_tracks = [t for t, r in risk_map.items() if 40 < r <= 70]

    strategies = []

    if high_tracks:
        strategies.append(_strategy(
            "W1", "emergency_weather_protocol",
            confidence=0.92,
            actions=[f"close_track_{t}+reroute_all_trains" for t in high_tracks],
        ))
        strategies.append(_strategy(
            "W2", "speed_restriction_high_risk_tracks",
            confidence=0.75,
            actions=[f"reduce_speed_40kmh_{t}+intensive_monitoring" for t in high_tracks],
        ))

    if medium_tracks:
        strategies.append(_strategy(
            "W3", "caution_medium_risk_tracks",
            confidence=0.60,
            actions=[f"reduce_speed_60kmh_{t}+30min_sensor_poll" for t in medium_tracks],
        ))

    if high_tracks or medium_tracks:
        strategies.append(_strategy(
            "W4", "monitor_and_reassess",
            confidence=0.40,
            actions=["continuous_weather_monitoring+alert_crew"],
        ))
    else:
        strategies = [_strategy(
            "W5", "nominal_weather_conditions",
            confidence=0.99,
            actions=["no_weather_action_required"],
        )]

    state["weather_strategies"] = strategies

    high_str = ", ".join(f"{t} ({r:.0f}%)" for t, r in risk_map.items() if r > 60)
    _log(state, "Weather",
         f"max risk {max(risk_map.values(), default=0):.0f}% | high-risk: {high_str or 'none'} | "
         f"strategy {strategies[0]['strategy_id']}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 2 — TRACK
# Reads  : snapshot.tracks[], weather_risk
# Outputs: track_strategies[]
# Rule   : health thresholds → close / restrict / monitor
# ─────────────────────────────────────────────────────────────────────────────

def track_node(state: RailState) -> RailState:
    tracks = state["snapshot"].get("tracks", {})
    strategies = []

    for track_id, track in tracks.items():
        health = track.get("health", 1.0)

        if health <= 0.2 or track.get("status") == "CLOSED":
            strategies.append({
                "strategy_id": f"T_CLOSE_{track_id}",
                "track_id": track_id,
                "action": "close_track",
            })
        elif health <= 0.5:
            strategies.append({
                "strategy_id": f"T_RESTRICT_{track_id}",
                "track_id": track_id,
                "action": "speed_limit_60",
            })
        elif health <= 0.7:
            strategies.append({
                "strategy_id": f"T_MONITOR_{track_id}",
                "track_id": track_id,
                "action": "monitor",
            })

    if not strategies:
        strategies.append({"strategy_id": "T_OK", "action": "healthy"})

    state["track_strategies"] = strategies
    _log(state, "Track",
         f"{sum(1 for s in strategies if 'CLOSE' in s['strategy_id'])} closures, "
         f"{sum(1 for s in strategies if 'RESTRICT' in s['strategy_id'])} restrictions, "
         f"{sum(1 for s in strategies if 'MONITOR' in s['strategy_id'])} monitored")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 3 — SIGNAL
# Reads  : snapshot.tracks[], weather_risk
# Outputs: signal_strategies[]
# Rule   : rule table — risk level → signal aspect (IRGSR-style)
# ─────────────────────────────────────────────────────────────────────────────

def signal_node(state: RailState) -> RailState:
    tracks       = state["snapshot"].get("tracks", {})
    weather_risk = state["weather_risk"]
    strategies   = []
    max_signal_risk = 0.0

    for track_id, track in tracks.items():
        health      = track.get("health", 1.0)
        w_risk_norm = weather_risk.get(track_id, 0) / 100.0
        risk_score  = round(0.6 * (1 - health) + 0.4 * w_risk_norm, 3)
        max_signal_risk = max(max_signal_risk, risk_score)
        sid_base    = track_id.replace(" ", "_")

        if risk_score > 0.65:
            strategies.append(_strategy(
                f"S_RED_{sid_base}",
                f"set_{track_id}_RED+halt_all_trains+dispatch_signal_tech",
                confidence=round(risk_score, 2),
                actions=[
                    f"signal_{track_id}_RED",
                    f"halt_trains_approaching_{track_id}",
                    f"dispatch_signal_tech_{track_id}_ETA_20min",
                ],
            ))
        elif risk_score > 0.35:
            strategies.append(_strategy(
                f"S_YELLOW_{sid_base}",
                f"set_{track_id}_YELLOW+speed_30kmh+caution_adjacent",
                confidence=round(risk_score, 2),
                actions=[
                    f"signal_{track_id}_YELLOW",
                    f"speed_limit_30kmh_{track_id}",
                    f"caution_signal_adjacent_sections_{track_id}",
                ],
            ))
        else:
            strategies.append(_strategy(
                f"S_GREEN_{sid_base}",
                f"keep_{track_id}_GREEN+routine_check",
                confidence=0.95,
                actions=[f"signal_{track_id}_GREEN"],
            ))

    if not strategies:
        strategies.append(_strategy(
            "S_OK", "all_signals_nominal",
            confidence=0.99,
            actions=["routine_signal_monitoring"],
        ))

    state["signal_strategies"] = strategies
    reds    = [s["strategy_id"] for s in strategies if s["strategy_id"].startswith("S_RED")]
    yellows = [s["strategy_id"] for s in strategies if s["strategy_id"].startswith("S_YELLOW")]
    _log(state, "Signal",
         f"max section risk {max_signal_risk:.2f} | {len(reds)} RED, {len(yellows)} YELLOW")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# AGENT 4 — ROUTING
# Reads  : track_strategies, graph, snapshot.trains
# Outputs: routing_strategies[]
# Algorithm: Dijkstra on the network minus closed edges
# ─────────────────────────────────────────────────────────────────────────────

def routing_node(state: RailState) -> RailState:
    G = state["graph"].copy()
    trains = state["snapshot"].get("trains", {})
    tracks = state["snapshot"].get("tracks", {})

    closed_tracks = [
        s["track_id"] for s in state["track_strategies"]
        if s.get("action") == "close_track"
    ]

    closed_edges = set()
    for track_id in closed_tracks:
        track = tracks.get(track_id, {})
        src, dst = track.get("source"), track.get("destination")
        if src and dst:
            closed_edges.add(frozenset({src, dst}))
            if G.has_edge(src, dst):
                G.remove_edge(src, dst)

    routing_strategies = []

    for train_id, train in trains.items():
        route = train.get("route", [])
        affected = any(
            frozenset({route[i], route[i + 1]}) in closed_edges
            for i in range(len(route) - 1)
        )
        if not affected:
            continue

        origin = train.get("current_station") or (route[0] if route else None)
        if origin not in G:
            origin = route[0] if route else None

        if origin == route[-1]:
            # Already at the final stop — nothing to reroute; the return
            # service gets planned on a later run once the train turns around.
            continue

        try:
            new_route = nx.shortest_path(G, source=origin, target=route[-1])
            if len(new_route) < 2:
                continue
            routing_strategies.append({
                "strategy_id": f"R_REROUTE_{train_id}",
                "train_id": train_id,
                "old_route": route,
                "new_route": new_route,
            })
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            routing_strategies.append({
                "strategy_id": f"R_HOLD_{train_id}",
                "train_id": train_id,
                "old_route": route,
            })

    if not routing_strategies:
        routing_strategies.append({"strategy_id": "R_NOMINAL"})

    state["routing_strategies"] = routing_strategies
    rerouted = [s["train_id"] for s in routing_strategies if s["strategy_id"].startswith("R_REROUTE")]
    held     = [s["train_id"] for s in routing_strategies if s["strategy_id"].startswith("R_HOLD")]
    _log(state, "Routing",
         f"{len(rerouted)} rerouted {rerouted or ''} | {len(held)} held {held or ''}"
         if rerouted or held else "all routes nominal")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# PLANNER AGENT
# Reads  : all specialist strategies
# Outputs: 3 candidate plans
#   Plan A — Safety First        (closures + RED signals + reroutes)
#   Plan B — Balanced Response   (restrictions + YELLOW signals + reroutes)
#   Plan C — Minimal Intervention(monitoring + GREEN signals, no reroutes)
# ─────────────────────────────────────────────────────────────────────────────

def planner_node(state: RailState) -> RailState:
    weather = state["weather_strategies"]
    track   = state["track_strategies"]
    signal  = state["signal_strategies"]
    routing = state["routing_strategies"]

    plans = [
        {
            "plan_id": "A",
            "plan_name": PLAN_NAMES["A"],
            "actions":
                weather +
                [x for x in track if "CLOSE" in x["strategy_id"]] +
                [x for x in signal if "RED" in x["strategy_id"]] +
                routing,
        },
        {
            "plan_id": "B",
            "plan_name": PLAN_NAMES["B"],
            "actions":
                weather +
                [x for x in track if "RESTRICT" in x["strategy_id"] or "MONITOR" in x["strategy_id"]] +
                [x for x in signal if "YELLOW" in x["strategy_id"]] +
                routing,
        },
        {
            "plan_id": "C",
            "plan_name": PLAN_NAMES["C"],
            "actions":
                weather +
                [x for x in track if "MONITOR" in x["strategy_id"] or x["strategy_id"] == "T_OK"] +
                [x for x in signal if "GREEN" in x["strategy_id"]] +
                [{"strategy_id": "R_NOMINAL"}],
        },
    ]

    state["plans"] = plans
    _log(state, "Planner", f"generated {len(plans)} candidate plans (A safety-first, B balanced, C minimal)")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE  (not an agent — infrastructure)
# Reads  : plans[], twin snapshot
# Process: apply each plan's actions to a scoring model of the twin
# Outputs: { plan_id, delay, risk, passenger_impact, congestion }
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_plan(plan: dict, snapshot: dict) -> dict:
    delay = 0
    risk = 0.25
    passenger_impact = 0
    congestion = 0.0

    trains = snapshot.get("trains", {})
    tracks = snapshot.get("tracks", {})

    # Baseline risk: unhealthy open tracks raise risk before mitigation
    for track in tracks.values():
        health = track.get("health", 1.0)
        if health <= 0.5:
            risk += 0.15
        elif health <= 0.7:
            risk += 0.05

    # Edges of failed tracks (broken or closed) and which trains cross them
    failed_edges = {
        frozenset({t["source"], t["destination"]})
        for t in tracks.values()
        if t.get("health", 1.0) <= 0.2 or t.get("status") == "CLOSED"
    }

    def _crosses_failure(route: List[str]) -> bool:
        return any(
            frozenset({route[i], route[i + 1]}) in failed_edges
            for i in range(len(route) - 1)
        )

    handled_trains = {
        a.get("train_id") for a in plan["actions"]
        if isinstance(a, dict) and a.get("strategy_id", "").startswith(("R_REROUTE_", "R_HOLD_"))
    }

    for action in plan["actions"]:
        if not isinstance(action, dict):
            continue

        strategy = action.get("strategy_id", "")

        if strategy.startswith("T_CLOSE_"):
            delay += 30
            risk -= 0.20
            congestion += 0.20

        elif strategy.startswith("T_RESTRICT_"):
            delay += 10
            risk -= 0.10
            congestion += 0.05

        elif strategy.startswith("T_MONITOR_"):
            delay += 5
            risk -= 0.05

        elif strategy.startswith("S_RED_"):
            delay += 15
            risk -= 0.10

        elif strategy.startswith("S_YELLOW_"):
            delay += 5
            risk -= 0.05

        elif strategy.startswith("R_REROUTE_"):
            train_id = action.get("train_id")
            delay += 15
            congestion += 0.10
            if train_id in trains:
                passenger_impact += trains[train_id].get("passengers", 0)

        elif strategy.startswith("R_HOLD_"):
            train_id = action.get("train_id")
            delay += 45
            congestion += 0.30
            if train_id in trains:
                passenger_impact += trains[train_id].get("passengers", 0)

    # Trains left running toward a failed track with no reroute/hold in this
    # plan are stranded: heavy delay, full passenger impact, elevated risk.
    if failed_edges:
        for train_id, train in trains.items():
            if train_id in handled_trains:
                continue
            if _crosses_failure(train.get("route", [])):
                delay += 60
                risk += 0.25
                congestion += 0.15
                passenger_impact += train.get("passengers", 0)

    risk = max(0.0, min(1.0, risk))
    congestion = round(min(congestion, 1.0), 2)

    return {
        "plan_id": plan["plan_id"],
        "plan_name": plan["plan_name"],
        "delay": delay,
        "risk": round(risk, 2),
        "passenger_impact": passenger_impact,
        "congestion": congestion,
    }


def simulation_node(state: RailState) -> RailState:
    results = []
    for plan in state["plans"]:
        result = _simulate_plan(plan, state["snapshot"])
        results.append(result)
        _log(state, "Simulator",
             f"{result['plan_name']}: delay {result['delay']}min, risk {result['risk']:.2f}, "
             f"{result['passenger_impact']} passengers, congestion {result['congestion']:.2f}")
    state["simulation_results"] = results
    return state


# ─────────────────────────────────────────────────────────────────────────────
# MASTER AGENT
# Reads  : simulation_results[]
# Algorithm: min-max normalisation + MCDM weighted scoring
# Weights: delay×0.35, risk×0.40, passengers×0.15, congestion×0.10
# Outputs: best_plan (with 0-100 suitability score; higher = better)
# ─────────────────────────────────────────────────────────────────────────────

def _minmax(values: List[float]) -> List[float]:
    lo, hi = min(values), max(values)
    if hi == lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]

def master_node(state: RailState) -> RailState:
    results = state["simulation_results"]

    norm = {
        "delay":            _minmax([r["delay"] for r in results]),
        "risk":             _minmax([r["risk"] for r in results]),
        "passenger_impact": _minmax([r["passenger_impact"] for r in results]),
        "congestion":       _minmax([r["congestion"] for r in results]),
    }

    for i, result in enumerate(results):
        cost = sum(MCDM_WEIGHTS[k] * norm[k][i] for k in MCDM_WEIGHTS)
        # Suitability score: 0-100, higher is better
        result["score"] = round((1.0 - cost) * 100, 1)

    ranked = sorted(results, key=lambda r: r["score"], reverse=True)
    best_plan = ranked[0]
    state["best_plan"] = best_plan

    _log(state, "Master",
         f"selected {best_plan['plan_name']} (score {best_plan['score']}/100) | "
         f"ranking: {' > '.join(p['plan_id'] for p in ranked)}")
    return state


# ─────────────────────────────────────────────────────────────────────────────
# LANGGRAPH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def build_nx_graph(state_snapshot: dict) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(state_snapshot.get("graph", {}).get("nodes", []))
    G.add_edges_from(state_snapshot.get("graph", {}).get("edges", []))
    return G


_builder = StateGraph(RailState)

for _name, _fn in [
    ("weather",    weather_node),
    ("track",      track_node),
    ("signal",     signal_node),
    ("routing",    routing_node),
    ("planner",    planner_node),
    ("simulation", simulation_node),
    ("master",     master_node),
]:
    _builder.add_node(_name, _fn)

_builder.set_entry_point("weather")

for _a, _b in [
    ("weather",    "track"),
    ("track",      "signal"),    # signal reads track health + weather risk
    ("signal",     "routing"),
    ("routing",    "planner"),
    ("planner",    "simulation"),
    ("simulation", "master"),
    ("master",     END),
]:
    _builder.add_edge(_a, _b)

pipeline = _builder.compile()


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL STATE FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _initial_state(snapshot: dict, extra_log: List[dict] | None = None) -> RailState:
    return RailState(
        snapshot=snapshot,
        graph=build_nx_graph(snapshot),
        weather_risk={},
        weather_strategies=[],
        track_strategies=[],
        signal_strategies=[],
        routing_strategies=[],
        plans=[],
        simulation_results=[],
        best_plan={},
        log=extra_log or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE FORMATTING  (matches the operator console contract)
# ─────────────────────────────────────────────────────────────────────────────

def _plan_payload(result: dict, plan: dict) -> dict:
    action_labels = []
    for a in plan["actions"]:
        sid = a.get("strategy_id", "")
        if sid.startswith("T_CLOSE_"):
            action_labels.append(f"Close track {a.get('track_id', sid.removeprefix('T_CLOSE_'))}")
        elif sid.startswith("T_RESTRICT_"):
            action_labels.append(f"Speed-limit 60 km/h on {a.get('track_id')}")
        elif sid.startswith("T_MONITOR_"):
            action_labels.append(f"Monitor {a.get('track_id')}")
        elif sid.startswith("S_RED_"):
            action_labels.append(f"Signal RED on {sid.removeprefix('S_RED_')}")
        elif sid.startswith("S_YELLOW_"):
            action_labels.append(f"Signal YELLOW on {sid.removeprefix('S_YELLOW_')}")
        elif sid.startswith("R_REROUTE_"):
            action_labels.append(f"Reroute {a.get('train_id')}")
        elif sid.startswith("R_HOLD_"):
            action_labels.append(f"Hold {a.get('train_id')} at station")
        elif sid.startswith("W") and a.get("strategy_name") not in (None, "nominal_weather_conditions"):
            action_labels.append(f"Weather protocol {sid}")
    if not action_labels:
        action_labels = ["Maintain nominal operations"]

    # Per-plan map data so the console can preview any candidate plan
    plan_failed = sorted({
        a["track_id"] for a in plan["actions"]
        if isinstance(a, dict) and a.get("strategy_id", "").startswith("T_CLOSE_")
    })
    plan_rerouted = [
        {"id": a["train_id"], "newRoute": a["new_route"]}
        for a in plan["actions"]
        if isinstance(a, dict) and a.get("strategy_id", "").startswith("R_REROUTE_")
    ]
    plan_held = [
        a["train_id"] for a in plan["actions"]
        if isinstance(a, dict) and a.get("strategy_id", "").startswith("R_HOLD_")
    ]

    return {
        "id": result["plan_id"],
        "name": result["plan_name"],
        "delay_min": result["delay"],
        "risk": result["risk"],
        "passengers_impacted": result["passenger_impact"],
        "congestion": result["congestion"],
        "score": result["score"],
        "actions": action_labels,
        "failed_tracks": plan_failed,
        "rerouted_trains": plan_rerouted,
        "held_trains": plan_held,
    }


def _template_explanation(final: RailState) -> str:
    """Deterministic explanation of the Master Agent's choice — no LLM needed."""
    results = sorted(final["simulation_results"], key=lambda r: r["score"], reverse=True)
    best, worst = results[0], results[-1]
    reroutes = sum(
        1 for s in final["routing_strategies"]
        if s["strategy_id"].startswith("R_REROUTE_")
    )
    parts = [
        f"{best['plan_name']} scored {best['score']}/100 — the best balance of "
        f"delay ({best['delay']} min), residual risk ({best['risk']:.2f}), "
        f"passenger impact ({best['passenger_impact']}) and congestion ({best['congestion']:.2f})."
    ]
    if reroutes:
        parts.append(f"It keeps {reroutes} affected train(s) moving via alternate corridors.")
    if worst["plan_id"] != best["plan_id"]:
        parts.append(
            f"{worst['plan_name']} was rejected (score {worst['score']}/100): "
            f"{worst['delay']} min of delay and risk {worst['risk']:.2f} make it the costliest option."
        )
    return " ".join(parts)


def _llm_explanation(final: RailState) -> str | None:
    """Ask Claude for a dispatcher-style justification. Returns None when the
    API key is absent or the call fails — callers fall back to the template."""
    if anthropic is None or not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        # Hard budget: a slow LLM call must fall back to the template well
        # inside the console's 8s POST / 60s SSE windows.
        client = anthropic.Anthropic(timeout=8.0, max_retries=0)
        summary = {
            "candidate_plans": [
                {k: r[k] for k in ("plan_id", "plan_name", "delay", "risk", "passenger_impact", "congestion", "score")}
                for r in final["simulation_results"]
            ],
            "selected": final["best_plan"]["plan_id"],
            "reroutes": [
                {"train": s["train_id"], "new_route": s["new_route"]}
                for s in final["routing_strategies"]
                if s["strategy_id"].startswith("R_REROUTE_")
            ],
            "closed_tracks": [
                s["track_id"] for s in final["track_strategies"]
                if s["strategy_id"].startswith("T_CLOSE_")
            ],
        }
        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=1024,
            system=(
                "You are the Master Agent of RailMind, a railway operations AI. "
                "Given simulation results for candidate response plans, explain to a human "
                "dispatcher in 2-3 plain sentences why the selected plan is the right call. "
                "Be concrete about the tradeoffs. No preamble, no markdown."
            ),
            messages=[{"role": "user", "content": json.dumps(summary)}],
        )
        if response.stop_reason == "refusal":
            return None
        return next((b.text for b in response.content if b.type == "text"), None)
    except Exception:
        return None


def _explanation(final: RailState) -> dict:
    llm_text = _llm_explanation(final)
    if llm_text:
        return {"text": llm_text.strip(), "source": "claude"}
    return {"text": _template_explanation(final), "source": "rules"}


def _format_response(final: RailState, twin_source: str) -> dict:
    best = final["best_plan"]
    plans_by_id = {p["plan_id"]: p for p in final["plans"]}

    failed_tracks = sorted({
        s["track_id"] for s in final["track_strategies"]
        if s["strategy_id"].startswith("T_CLOSE_")
    })

    rerouted_trains = [
        {"id": s["train_id"], "newRoute": s["new_route"]}
        for s in final["routing_strategies"]
        if s["strategy_id"].startswith("R_REROUTE_")
    ]

    reroutes = [
        {"train": s["train_id"], "from": s.get("old_route", []), "to": s["new_route"]}
        for s in final["routing_strategies"]
        if s["strategy_id"].startswith("R_REROUTE_")
    ]

    held_trains = [
        s["train_id"] for s in final["routing_strategies"]
        if s["strategy_id"].startswith("R_HOLD_")
    ]

    trains = [
        {
            "id": tid,
            "route": t.get("route", []),
            "passengers": t.get("passengers", 0),
            "current_station": t.get("current_station"),
        }
        for tid, t in final["snapshot"].get("trains", {}).items()
    ]

    weather_risk = final["weather_risk"]

    return {
        "twin_source": twin_source,
        "explanation": _explanation(final),
        "recommended_action": _plan_payload(best, plans_by_id[best["plan_id"]]),
        "candidate_plans": [
            _plan_payload(r, plans_by_id[r["plan_id"]])
            for r in final["simulation_results"]
        ],
        "agent_outputs": {
            "weather": {
                "strategy": final["weather_strategies"][0]["strategy_id"] if final["weather_strategies"] else "W5",
                "risk_score": round(max(weather_risk.values(), default=0) / 100, 2),
                "high_risk_tracks": [t for t, r in weather_risk.items() if r > 60],
            },
            "track": {
                "actions": [s["strategy_id"] for s in final["track_strategies"]],
            },
            "signal": {
                "actions": [
                    s["strategy_id"] for s in final["signal_strategies"]
                    if not s["strategy_id"].startswith("S_GREEN")
                ] or [s["strategy_id"] for s in final["signal_strategies"]][:6],
            },
            "routing": {
                "affected_trains": [r["train"] for r in reroutes] + held_trains,
                "reroutes": reroutes,
            },
        },
        "execution_log": list(reversed(final["log"])),  # newest first
        "failed_tracks": failed_tracks,
        "rerouted_trains": rerouted_trains,
        "held_trains": held_trains,
        "trains": trains,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {
        "service": "RailMind Agents",
        "status": "ok",
        "endpoints": [
            "/state", "/run", "/run-stream", "/apply-plan", "/reset",
            "/simulate-track-failure/{track_id}", "/simulate-weather/{track_id}", "/docs",
        ],
    }


@app.get("/state")
def live_state():
    """Live twin snapshot for the console's polling loop (trains move each tick)."""
    snapshot, source = fetch_snapshot()
    trains = [
        {
            "id": tid,
            "route": t.get("route", []),
            "passengers": t.get("passengers", 0),
            "current_station": t.get("current_station"),
            "route_index": t.get("route_index", 0),
            "progress": t.get("progress", 0.0),
            "held": t.get("held", False),
            "delayed_minutes": round(t.get("delayed_minutes", 0.0), 1),
        }
        for tid, t in snapshot.get("trains", {}).items()
    ]
    tracks = [
        {
            "id": tid,
            "health": t.get("health", 1.0),
            "status": t.get("status", "OPEN"),
            "source": t.get("source"),
            "destination": t.get("destination"),
        }
        for tid, t in snapshot.get("tracks", {}).items()
    ]
    weather = {
        tid: (c.value if hasattr(c, "value") else str(c))
        for tid, c in (snapshot.get("weather") or {}).items()
    } if isinstance(snapshot.get("weather"), dict) else {}
    return {"twin_source": source, "trains": trains, "tracks": tracks, "weather": weather}


@app.post("/reset")
def reset():
    """Reset the twin to its baseline state (clears failures, weather, reroutes)."""
    try:
        _twin_reset()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Twin reset failed: {e}")
    return {"status": "success", "message": "Twin reset to baseline."}


@app.post("/run")
def run_pipeline():
    """Fetch the digital twin state and run the full agent pipeline."""
    snapshot, source = fetch_snapshot()
    final = pipeline.invoke(_initial_state(snapshot))
    return _format_response(final, source)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/run-stream")
def run_pipeline_stream(inject_track: str | None = None, weather_track: str | None = None,
                        condition: str = "STORM"):
    """Run the pipeline and stream agent log events live over SSE.

    Optional query params apply an injection first (persistently, like the
    POST endpoints): ?inject_track=T23 or ?weather_track=T05&condition=STORM.
    """
    known_tracks = fetch_snapshot()[0].get("tracks", {})
    for tid in (inject_track, weather_track):
        if tid and tid not in known_tracks:
            raise HTTPException(status_code=404, detail=f"Track '{tid}' not found.")

    extra_log: List[dict] = []
    try:
        if inject_track:
            _twin_close_track(inject_track)
            extra_log.append({"t": time.strftime("%H:%M:%S"), "source": "Sensor",
                              "message": f"[INJECTED] Track {inject_track} failure detected — closed"})
        if weather_track:
            cond = condition.upper()
            if cond in WEATHER_CONDITION_RISK:
                _twin_set_weather(weather_track, cond)
                extra_log.append({"t": time.strftime("%H:%M:%S"), "source": "Sensor",
                                  "message": f"[INJECTED] {cond} reported on track {weather_track}"})
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Twin unreachable: {e}")

    snapshot, source = fetch_snapshot()

    def event_stream():
        state = _initial_state(snapshot, extra_log=list(extra_log))
        emitted = 0
        for entry in extra_log:
            yield _sse("log", entry)
            emitted += 1

        final_state = dict(state)
        for update in pipeline.stream(state):
            for node_state in update.values():
                if not isinstance(node_state, dict):
                    continue
                final_state.update(node_state)
                log = final_state.get("log", [])
                while emitted < len(log):
                    yield _sse("log", log[emitted])
                    emitted += 1
                    time.sleep(0.15)  # let the timeline animate

        yield _sse("result", _format_response(final_state, source))
        yield _sse("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/simulate-track-failure/{track_id}")
def simulate_track_failure(track_id: str):
    """Inject a persistent track failure into the twin and run the pipeline.

    Failures accumulate — inject several tracks to build a cascade scenario.
    Use /reset to return to baseline.
    """
    snapshot, _ = fetch_snapshot()
    if track_id not in snapshot.get("tracks", {}):
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")

    try:
        _twin_close_track(track_id)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Twin unreachable: {e}")

    snapshot, source = fetch_snapshot()
    injected_log = [{
        "t": time.strftime("%H:%M:%S"),
        "source": "Sensor",
        "message": f"[INJECTED] Track {track_id} failure detected — closed",
    }]
    final = pipeline.invoke(_initial_state(snapshot, extra_log=injected_log))
    return {"injected_failure": track_id, **_format_response(final, source)}


@app.post("/simulate-weather/{track_id}")
def simulate_weather(track_id: str, condition: str = "STORM"):
    """Inject persistent severe weather on a track and run the pipeline."""
    condition = condition.upper()
    if condition not in WEATHER_CONDITION_RISK:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown condition '{condition}'. Use one of {list(WEATHER_CONDITION_RISK)}.",
        )

    snapshot, _ = fetch_snapshot()
    if track_id not in snapshot.get("tracks", {}):
        raise HTTPException(status_code=404, detail=f"Track '{track_id}' not found.")

    try:
        _twin_set_weather(track_id, condition)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Twin unreachable: {e}")

    snapshot, source = fetch_snapshot()
    injected_log = [{
        "t": time.strftime("%H:%M:%S"),
        "source": "Sensor",
        "message": f"[INJECTED] {condition} reported on track {track_id}",
    }]
    final = pipeline.invoke(_initial_state(snapshot, extra_log=injected_log))
    return {"injected_weather": {"track_id": track_id, "condition": condition},
            **_format_response(final, source)}


@app.post("/apply-plan")
def apply_plan(plan_id: str | None = None):
    """Close the loop: run the pipeline, then execute the chosen plan on the twin.

    Closes flagged tracks and applies the plan's reroutes so the living twin
    reflects the decision — trains follow their new routes on subsequent ticks.
    Defaults to the Master Agent's recommended plan.
    """
    snapshot, source = fetch_snapshot()
    final = pipeline.invoke(_initial_state(snapshot))

    target_id = (plan_id or final["best_plan"]["plan_id"]).upper()
    plan = next((p for p in final["plans"] if p["plan_id"] == target_id), None)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Plan '{target_id}' not found.")

    applied = []
    try:
        for action in plan["actions"]:
            if not isinstance(action, dict):
                continue
            sid = action.get("strategy_id", "")
            if sid.startswith("T_CLOSE_"):
                _twin_close_track(action["track_id"])
                applied.append(f"Closed track {action['track_id']}")
            elif sid.startswith("R_REROUTE_"):
                _twin_reroute_train(action["train_id"], action["new_route"])
                applied.append(f"Rerouted {action['train_id']}")
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Twin unreachable while applying plan: {e}")

    final["log"].append({
        "t": time.strftime("%H:%M:%S"),
        "source": "Executor",
        "message": f"Plan {target_id} executed on the live twin: {len(applied)} action(s)",
    })
    return {"executed_plan": target_id, "applied_actions": applied,
            **_format_response(final, source)}
