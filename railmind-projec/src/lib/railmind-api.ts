// RailMind API client. Talks to the FastAPI agent service when reachable and
// falls back to a fully-featured local mock (same canonical network) so the
// dashboard always works — even with no backend running.

import {
  STATIONS,
  TRACKS,
  TRAINS,
  findPath,
  stationName,
  trackById,
  type Train,
} from "./railmind-data";

export type AgentOutputs = {
  weather: { strategy: string; risk_score: number; high_risk_tracks: string[] };
  track: { actions: string[] };
  signal: { actions: string[] };
  routing: {
    affected_trains: string[];
    reroutes: { train: string; from: string[]; to: string[] }[];
  };
};

export type Plan = {
  id: string;
  name: string;
  delay_min: number;
  risk: number;
  passengers_impacted: number;
  congestion: number;
  score: number; // 0-100, higher is better
  actions: string[];
  // Per-plan map preview data (real backend)
  failed_tracks?: string[];
  rerouted_trains?: { id: string; newRoute: string[] }[];
  held_trains?: string[];
};

export type LogEvent = { t: string; source: string; message: string };

export type Explanation = { text: string; source: "claude" | "rules" | string };

export type RunResponse = {
  injected_failure?: string;
  twin_source?: string;
  explanation?: Explanation;
  recommended_action: Plan;
  candidate_plans: Plan[];
  agent_outputs: AgentOutputs;
  execution_log: LogEvent[];
  failed_tracks: string[];
  rerouted_trains: { id: string; newRoute: string[] }[];
  held_trains?: string[];
  trains?: Train[];
  executed_plan?: string;
  applied_actions?: string[];
};

export type LiveTrack = {
  id: string;
  health: number;
  status: string;
  source?: string;
  destination?: string;
};

export type LiveState = {
  twin_source: string;
  trains: Train[];
  tracks: LiveTrack[];
  weather: Record<string, string>;
};

const API_BASE_URL: string =
  (import.meta.env?.VITE_API_BASE_URL as string | undefined) ?? "http://127.0.0.1:8001";

function nowHMS(offset = 0) {
  const d = new Date(Date.now() + offset * 1000);
  return d.toTimeString().slice(0, 8);
}

// ── Mock pipeline (mirrors the FastAPI agent flow on the same network) ──────

export function buildMockRun(trackId?: string): RunResponse {
  const failed = trackId ?? "T23";
  const t = trackById(failed);
  const fromName = stationName(t?.from ?? "NAGPUR_JUNCT");
  const toName = stationName(t?.to ?? "RAIPUR_JUNCT");

  const crossesFailure = (route: string[]) =>
    route.some(
      (s, i) =>
        i < route.length - 1 &&
        !!t &&
        ((s === t.from && route[i + 1] === t.to) || (s === t.to && route[i + 1] === t.from)),
    );

  const affected = TRAINS.filter((tr) => crossesFailure(tr.route));
  const passengers = affected.reduce((a, b) => a + b.passengers, 0);

  const reroutes = affected.map((tr) => {
    const alt = findPath(tr.route[0], tr.route[tr.route.length - 1], [failed]);
    return { train: tr.id, from: tr.route, to: alt ?? tr.route };
  });

  const monitored = TRACKS.filter((x) => x.status === "monitored" && x.id !== failed).map(
    (x) => x.id,
  );

  const agent_outputs: AgentOutputs = {
    weather: { strategy: "W5", risk_score: 0.05, high_risk_tracks: [] },
    track: {
      actions: [`T_CLOSE_${failed}`, ...monitored.slice(0, 3).map((id) => `T_MONITOR_${id}`)],
    },
    signal: {
      actions: [`S_YELLOW_${failed}`, ...monitored.slice(0, 2).map((id) => `S_YELLOW_${id}`)],
    },
    routing: {
      affected_trains: affected.map((a) => a.id),
      reroutes,
    },
  };

  const plans: Plan[] = [
    {
      id: "A",
      name: "Plan A — Safety First",
      delay_min: 30 + reroutes.length * 15,
      risk: 0.18,
      passengers_impacted: passengers,
      congestion: 0.45,
      score: 84,
      actions: [
        `Close track ${failed}`,
        `Signal RED on ${failed}`,
        ...affected.map((a) => `Reroute ${a.id}`),
      ],
      failed_tracks: [failed],
      rerouted_trains: reroutes.map((r) => ({ id: r.train, newRoute: r.to })),
      held_trains: [],
    },
    {
      id: "B",
      name: "Plan B — Balanced Response",
      delay_min: 15 + reroutes.length * 15,
      risk: 0.24,
      passengers_impacted: passengers,
      congestion: 0.35,
      score: 95,
      actions: [
        `Speed-limit 60 km/h on ${failed}`,
        `Signal YELLOW on ${failed}`,
        ...affected.map((a) => `Reroute ${a.id}`),
      ],
      failed_tracks: [],
      rerouted_trains: reroutes.map((r) => ({ id: r.train, newRoute: r.to })),
      held_trains: [],
    },
    {
      id: "C",
      name: "Plan C — Minimal Intervention",
      delay_min: 60 * Math.max(affected.length, 1),
      risk: 0.85,
      passengers_impacted: passengers,
      congestion: 0.7,
      score: 21,
      actions: [`Monitor ${failed}`, "Keep all signals GREEN", "No reroutes"],
      failed_tracks: [],
      rerouted_trains: [],
      held_trains: [],
    },
  ];

  const recommended = plans.reduce((a, b) => (a.score >= b.score ? a : b));

  const log: LogEvent[] = [
    {
      t: nowHMS(-7),
      source: "Sensor",
      message: `Anomaly detected on track ${failed} (${fromName} ↔ ${toName})`,
    },
    {
      t: nowHMS(-6),
      source: "Weather",
      message: "Nominal conditions — no weather action required",
    },
    { t: nowHMS(-6), source: "Track", message: `Track Agent flagged ${failed} for closure` },
    { t: nowHMS(-5), source: "Signal", message: `Signal Agent set ${failed} to YELLOW` },
    ...affected.map((a, i) => ({
      t: nowHMS(-4 + i * 0.1),
      source: "Routing",
      message: `Routing Agent computed alternate path for ${a.id}`,
    })),
    {
      t: nowHMS(-3),
      source: "Planner",
      message: `Planner generated ${plans.length} candidate plans`,
    },
    {
      t: nowHMS(-2),
      source: "Simulator",
      message: "Simulator scored plans against the incident snapshot",
    },
    {
      t: nowHMS(0),
      source: "Master",
      message: `Master Agent selected ${recommended.name} (score ${recommended.score}/100)`,
    },
  ];
  log.reverse(); // newest first

  return {
    injected_failure: trackId,
    twin_source: "mock (offline)",
    explanation: {
      text:
        `${recommended.name} scored ${recommended.score}/100 — it reroutes ${reroutes.length} ` +
        `affected train(s) around ${failed} (${fromName} ↔ ${toName}) while keeping delay at ` +
        `${recommended.delay_min} min. Minimal intervention would strand passengers on a dead section.`,
      source: "rules",
    },
    recommended_action: recommended,
    candidate_plans: plans,
    agent_outputs,
    execution_log: log,
    failed_tracks: [failed],
    rerouted_trains: reroutes.map((r) => ({ id: r.train, newRoute: r.to })),
    held_trains: [],
    trains: TRAINS,
  };
}

// Mock approximation of a weather incident: storms degrade the track badly
// enough to close it. Shared by injectWeather's offline path and the
// storm-run retry in the dashboard so both report a weather-shaped incident.
export function buildMockWeatherRun(trackId: string, condition = "STORM"): RunResponse {
  const mock = buildMockRun(trackId);
  mock.agent_outputs.weather = { strategy: "W1", risk_score: 0.85, high_risk_tracks: [trackId] };
  mock.execution_log = [
    {
      t: nowHMS(0),
      source: "Weather",
      message: `${condition} reported on ${trackId} — emergency weather protocol`,
    },
    ...mock.execution_log,
  ];
  return mock;
}

// ── Real backend adapter ────────────────────────────────────────────────────

function isFiniteNumber(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

// Every metric the console renders must be a real number: normalizePlan's 0s
// exist to survive *optional* gaps, but a plan missing its core metrics would
// otherwise render "Risk 0.00" — a confident lie on an ops console. Reject it.
function isValidPlan(x: unknown): x is Plan {
  if (!x || typeof x !== "object") return false;
  const p = x as Record<string, unknown>;
  return (
    typeof p.id === "string" &&
    isFiniteNumber(p.delay_min) &&
    isFiniteNumber(p.risk) &&
    isFiniteNumber(p.passengers_impacted) &&
    isFiniteNumber(p.congestion) &&
    isFiniteNumber(p.score) &&
    Array.isArray(p.actions)
  );
}

export function isValidRunResponse(x: unknown): x is RunResponse {
  if (!x || typeof x !== "object") return false;
  const r = x as Record<string, unknown>;
  return (
    isValidPlan(r.recommended_action) &&
    Array.isArray(r.candidate_plans) &&
    r.candidate_plans.every(isValidPlan) &&
    Array.isArray(r.execution_log)
  );
}

// Second belt behind isValidRunResponse: defaults the *optional* fields the
// render path dereferences (preview arrays etc.) so they can never crash it.
function normalizePlan(p: Plan): Plan {
  return {
    ...p,
    delay_min: p.delay_min ?? 0,
    risk: p.risk ?? 0,
    passengers_impacted: p.passengers_impacted ?? 0,
    congestion: p.congestion ?? 0,
    score: p.score ?? 0,
    actions: p.actions ?? [],
  };
}

export function normalize(r: RunResponse): RunResponse {
  return {
    ...r,
    recommended_action: normalizePlan(r.recommended_action),
    candidate_plans: (r.candidate_plans ?? []).map(normalizePlan),
    execution_log: r.execution_log ?? [],
    failed_tracks: r.failed_tracks ?? [],
    rerouted_trains: r.rerouted_trains ?? [],
    held_trains: r.held_trains ?? [],
    trains: r.trains && r.trains.length > 0 ? r.trains : TRAINS,
    agent_outputs: {
      weather: {
        strategy: r.agent_outputs?.weather?.strategy ?? "—",
        risk_score: r.agent_outputs?.weather?.risk_score ?? 0,
        high_risk_tracks: r.agent_outputs?.weather?.high_risk_tracks ?? [],
      },
      track: { actions: r.agent_outputs?.track?.actions ?? [] },
      signal: { actions: r.agent_outputs?.signal?.actions ?? [] },
      routing: {
        affected_trains: r.agent_outputs?.routing?.affected_trains ?? [],
        reroutes: r.agent_outputs?.routing?.reroutes ?? [],
      },
    },
  };
}

async function tryFetch(
  url: string,
  init?: RequestInit,
  timeoutMs = 8000,
): Promise<RunResponse | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    const res = await fetch(url, { ...init, signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return null;
    const json: unknown = await res.json();
    return isValidRunResponse(json) ? normalize(json) : null;
  } catch {
    return null;
  }
}

/**
 * Run the pipeline. `mockTrackId`/`mockKind` only shape the offline mock: pass
 * the track (and incident kind) of an injection that already reached the
 * backend so a fallback after a mid-stream failure reports the right incident
 * — weather-shaped for storms — instead of the T23 track-failure default.
 */
export async function runSimulation(
  mockTrackId?: string,
  mockKind: "track" | "weather" = "track",
): Promise<RunResponse> {
  const real = await tryFetch(`${API_BASE_URL}/run`, { method: "POST" }, 20000);
  if (real) return real;
  await new Promise((r) => setTimeout(r, 600));
  return mockKind === "weather" && mockTrackId
    ? buildMockWeatherRun(mockTrackId)
    : buildMockRun(mockTrackId);
}

/** Live twin snapshot for the polling loop. Null when the backend is offline. */
export async function fetchLiveState(): Promise<LiveState | null> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 5000);
    const res = await fetch(`${API_BASE_URL}/state`, { signal: controller.signal });
    clearTimeout(timer);
    if (!res.ok) return null;
    const json = (await res.json()) as LiveState;
    return Array.isArray(json.trains) && Array.isArray(json.tracks) ? json : null;
  } catch {
    return null;
  }
}

export async function applyPlan(planId?: string): Promise<RunResponse | null> {
  const url = planId
    ? `${API_BASE_URL}/apply-plan?plan_id=${encodeURIComponent(planId)}`
    : `${API_BASE_URL}/apply-plan`;
  // Applying reroutes to a (possibly remote) twin is the slowest call — give
  // it real headroom so a successful execution is never mislabeled as offline.
  return tryFetch(url, { method: "POST" }, 25000);
}

export async function resetTwin(): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 8000);
    const res = await fetch(`${API_BASE_URL}/reset`, {
      method: "POST",
      signal: controller.signal,
    });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

export type StreamParams = {
  injectTrack?: string;
  weatherTrack?: string;
  condition?: string;
};

export type StreamOutcome = {
  result: RunResponse | null;
  /**
   * True when the backend received the request (any SSE event arrived). An
   * injection has then already been applied server-side — a caller that wants
   * a fallback must use plain /run, never the injecting endpoints again.
   */
  sawEvents: boolean;
};

/**
 * Run the pipeline with live SSE log streaming. Calls onLog per agent event.
 * Resolves with the final response plus whether the stream ever delivered
 * anything (see StreamOutcome for the re-injection contract).
 */
export function runSimulationStream(
  onLog: (e: LogEvent) => void,
  params: StreamParams = {},
): Promise<StreamOutcome> {
  return new Promise((resolve) => {
    if (typeof EventSource === "undefined") {
      resolve({ result: null, sawEvents: false });
      return;
    }
    const qs = new URLSearchParams();
    if (params.injectTrack) qs.set("inject_track", params.injectTrack);
    if (params.weatherTrack) {
      qs.set("weather_track", params.weatherTrack);
      qs.set("condition", params.condition ?? "STORM");
    }
    const url = `${API_BASE_URL}/run-stream${qs.size ? `?${qs}` : ""}`;

    let settled = false;
    let sawEvents = false;
    let result: RunResponse | null = null;
    const source = new EventSource(url);

    const finish = () => {
      if (settled) return;
      settled = true;
      source.close();
      resolve({ result, sawEvents });
    };

    const guard = setTimeout(finish, 60000);

    source.addEventListener("log", (ev) => {
      sawEvents = true;
      try {
        onLog(JSON.parse((ev as MessageEvent).data) as LogEvent);
      } catch {
        // ignore malformed log frames
      }
    });
    source.addEventListener("result", (ev) => {
      sawEvents = true;
      try {
        const parsed: unknown = JSON.parse((ev as MessageEvent).data);
        if (isValidRunResponse(parsed)) result = normalize(parsed);
      } catch {
        result = null;
      }
    });
    source.addEventListener("done", () => {
      clearTimeout(guard);
      finish();
    });
    source.onerror = () => {
      clearTimeout(guard);
      finish();
    };
  });
}

// Injection endpoints do strictly more work than /run (twin mutation
// round-trips + full pipeline + LLM budget) — give them the same 20s budget.
export async function injectTrackFailure(trackId: string): Promise<RunResponse> {
  const real = await tryFetch(
    `${API_BASE_URL}/simulate-track-failure/${trackId}`,
    { method: "POST" },
    20000,
  );
  if (real) return real;
  await new Promise((r) => setTimeout(r, 600));
  return buildMockRun(trackId);
}

export async function injectWeather(trackId: string, condition = "STORM"): Promise<RunResponse> {
  const real = await tryFetch(
    `${API_BASE_URL}/simulate-weather/${trackId}?condition=${encodeURIComponent(condition)}`,
    { method: "POST" },
    20000,
  );
  if (real) return real;
  await new Promise((r) => setTimeout(r, 600));
  return buildMockWeatherRun(trackId, condition);
}

export function incidentSummary(r: RunResponse | null) {
  if (!r || !r.failed_tracks?.length) return null;
  const failed = r.failed_tracks[0];
  const t = trackById(failed);
  const from = t ? stationName(t.from) : "—";
  const to = t ? stationName(t.to) : "—";
  const affected = r.agent_outputs.routing.affected_trains;
  const fleet = r.trains ?? TRAINS;
  const passengers = fleet
    .filter((tr) => affected.includes(tr.id))
    .reduce((a, b) => a + b.passengers, 0);
  return { trackId: failed, from, to, affectedCount: affected.length, passengers };
}

export { STATIONS, TRACKS, TRAINS };
