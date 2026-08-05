# RailMind 🚆🧠

**RailMind** is an autonomous multi-agent railway operating system that monitors, predicts, and optimizes railway operations in real time.

A **digital twin** of a 21-station Indian railway network feeds a **LangGraph pipeline of specialist AI agents** (weather, track, signal, routing) that generate candidate response plans, simulate each one on a cloned twin, and rank them with **MCDM scoring** — all surfaced live on a mission-control **operator console**.

> **Zero-setup demo:** every layer is offline-first. No API keys, no internet required — the whole stack runs on your laptop.

---

## Architecture

```text
┌──────────────────────────┐     ┌──────────────────────────────────────┐
│  Digital Twin (Django)   │     │  Agent Orchestrator (FastAPI +       │
│  :8000                   │◄────│  LangGraph)  :8001                   │
│                          │     │                                      │
│  • 21 stations, 33       │     │  Weather → Track → Signal → Routing  │
│    corridors (real geo)  │     │      └────────────┬─────────────┘    │
│  • Trains + weather      │     │                Planner               │
│  • Clone / close-track / │     │          (Plans A / B / C)           │
│    reroute / risk APIs   │     │                   │                  │
└──────────────────────────┘     │         Simulation Engine            │
                                 │      (clone twin, score plans)       │
        shared canonical         │                   │                  │
        network dataset          │            Master Agent              │
   railmind/data/                │      (min-max MCDM ranking)          │
   india_network.json            └───────────────────┬──────────────────┘
                                                     │ REST (CORS)
                                 ┌───────────────────▼──────────────────┐
                                 │  Operator Console (React + Vite)     │
                                 │  :8080 — live map, fleet, plans,     │
                                 │  agent timeline, failure injection   │
                                 └──────────────────────────────────────┘
```

| Layer | Tech | Directory |
|---|---|---|
| Core engine (twin, graph, simulator) | Python, NetworkX, Pydantic | [`railmind/`](railmind/) |
| Digital twin API | Django + DRF (+ Swagger docs) | [`railmind_django/`](railmind_django/) |
| Agent orchestrator | FastAPI + LangGraph | [`agents/`](agents/) |
| Operator console | React 19, TanStack Start, Tailwind | [`railmind-projec/`](railmind-projec/) |

All layers share one canonical network — [`railmind/data/india_network.json`](railmind/data/india_network.json) — so station and track IDs line up end to end.

---

## Quickstart

Prerequisites: **Python 3.11+**, **Node 20+**.

### 1. Digital twin (Django) — port 8000

```bash
pip install -r railmind_django/requirements.txt
cd railmind_django
python manage.py runserver 8000
```

Interactive API docs: http://127.0.0.1:8000/api/docs/

### 2. Agent orchestrator (FastAPI) — port 8001

```bash
pip install -r agents/requirements.txt
cd agents
uvicorn all_agent:app --port 8001
```

Interactive API docs: http://127.0.0.1:8001/docs

The orchestrator auto-discovers its twin: `TWIN_BASE_URL` env var → local Django (`:8000`) → hosted twin → **embedded twin** (built in-process from the bundled dataset). It works even if you skip step 1.

### 3. Operator console (React) — port 8080

```bash
cd railmind-projec
npm install
npm run dev
```

Open http://localhost:8080. If the agent service is unreachable, the console falls back to a built-in mock pipeline on the same network — the UI always works.

### Core engine only (no servers)

```bash
python -m railmind.main          # twin + future simulator CLI demo
python railmind/plot_results.py  # network + scenario charts (needs matplotlib)
```

---

## Demo script (2 minutes)

1. Open the console — the live map shows the 21-station network with per-track health.
2. Click **Run Full Simulation** — agents evaluate the healthy network; Master Agent recommends minimal intervention.
3. Select track **T23 · Nagpur ↔ Raipur** and click **Inject Failure** — watch:
   - Incident panel flips to **ACTIVE** with affected trains + passengers,
   - the failed track turns red, rerouted trains draw blue alternate paths,
   - each agent card fills with its decisions,
   - three candidate plans appear, scored 0–100, best one highlighted,
   - the Agent Timeline replays the whole pipeline event by event.
4. Click **Storm** — the Weather Agent escalates to the emergency protocol (W1) and signals react.
5. URL shortcuts for instant demos: `/?autorun` runs the pipeline on load, `/?inject=T23` injects a failure on load.

---

## How decisions are made

1. **Specialist agents** each read the twin snapshot and emit strategies:
   - *Weather* — per-track risk from live conditions (storm/rain/fog) or sensor feeds
   - *Track* — health thresholds → close / speed-restrict / monitor
   - *Signal* — combined health + weather risk → RED / YELLOW / GREEN (IRGSR-style rule table)
   - *Routing* — Dijkstra reroutes for every train crossing a failed section
2. **Planner** composes three doctrines: **A — Safety First**, **B — Balanced Response**, **C — Minimal Intervention**.
3. **Simulation engine** applies each plan to a cloned twin and measures delay, residual risk, passenger impact and congestion — including stranded-train penalties for plans that ignore failures.
4. **Master Agent** min-max normalises the four criteria and ranks plans with MCDM weights (risk 0.40, delay 0.35, passengers 0.15, congestion 0.10) → one recommended action with a 0–100 suitability score.

---

## Key API endpoints

| Service | Endpoint | Purpose |
|---|---|---|
| Twin | `GET /api/state/` | Full network snapshot (tracks, trains, weather, graph) |
| Twin | `POST /api/copy/` | Clone the twin into a sandbox "future" session |
| Twin | `POST /api/track/close/` · `POST /api/route/find/` | Mutate / query the twin |
| Twin | `POST /api/reset/` | Rebuild baseline twin between demos |
| Agents | `POST /run` | Run the full pipeline on the live twin |
| Agents | `POST /simulate-track-failure/{track_id}` | Inject a failure, then run the pipeline |
| Agents | `POST /simulate-weather/{track_id}?condition=STORM` | Inject weather, then run the pipeline |

---

## Deployment

[`render.yaml`](render.yaml) is a Render blueprint for both backend services (`railmind-twin`, `railmind-agents`). After the twin deploys, set `TWIN_BASE_URL` on the agents service to its public URL. The console is an SSR app — publish it through Lovable (or any Node host) with `VITE_API_BASE_URL` pointing at the deployed agent service; locally it needs no config at all.

---

## Configuration

| Variable | Where | Default | Effect |
|---|---|---|---|
| `TWIN_BASE_URL` | agents | auto-discover | Point the orchestrator at a specific twin |
| `VITE_API_BASE_URL` | console | `http://127.0.0.1:8001` | Agent service URL |
| `RAILMIND_LIVE_OSM` | core | off | Load the real network from OpenStreetMap (Overpass) instead of the bundled dataset; results are cached, failures fall back offline |
