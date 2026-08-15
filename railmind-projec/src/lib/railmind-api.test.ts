import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildMockRun,
  buildMockWeatherRun,
  isValidRunResponse,
  normalize,
  runSimulation,
  type RunResponse,
} from "./railmind-api";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("buildMockRun", () => {
  it("defaults to the T23 demo incident", () => {
    expect(buildMockRun().failed_tracks).toContain("T23");
  });

  it("shapes the incident around an explicit track id", () => {
    const run = buildMockRun("T05");
    expect(run.failed_tracks).toContain("T05");
    expect(run.failed_tracks).not.toContain("T23");
    expect(run.recommended_action.actions.join(" ")).toContain("T05");
  });
});

describe("buildMockWeatherRun", () => {
  it("shapes the mock as a weather incident, not a bare track failure", () => {
    const run = buildMockWeatherRun("T11");
    expect(run.failed_tracks).toContain("T11");
    expect(run.agent_outputs.weather.strategy).toBe("W1");
    expect(run.agent_outputs.weather.high_risk_tracks).toContain("T11");
    expect(run.execution_log[0].source).toBe("Weather");
    expect(run.execution_log[0].message).toContain("STORM reported on T11");
  });
});

describe("isValidRunResponse", () => {
  it("accepts a complete run", () => {
    expect(isValidRunResponse(buildMockRun())).toBe(true);
  });

  it("rejects a response whose plan is missing risk", () => {
    // normalizePlan would otherwise render the gap as a confident "Risk 0.00".
    const run = buildMockRun() as unknown as { candidate_plans: Record<string, unknown>[] };
    delete run.candidate_plans[1].risk;
    expect(isValidRunResponse(run)).toBe(false);
  });

  it("rejects a recommendation with a non-finite score", () => {
    const run = buildMockRun();
    run.recommended_action.score = NaN;
    expect(isValidRunResponse(run)).toBe(false);
  });
});

describe("normalize", () => {
  it("defaults every nested field the render path dereferences", () => {
    // Second belt behind isValidRunResponse (which now rejects a response this
    // sparse): even fed directly, it must come out fully defaulted, not crash.
    const sparse = {
      twin_source: "embedded",
      explanation: { text: "x", source: "rules" },
      recommended_action: { id: "A", name: "Plan A" },
      candidate_plans: [{ id: "B", name: "Plan B" }],
      execution_log: [],
    } as unknown as RunResponse;

    const r = normalize(sparse);
    expect(r.recommended_action.risk).toBe(0);
    expect(r.recommended_action.actions).toEqual([]);
    expect(r.candidate_plans[0].score).toBe(0);
    expect(r.failed_tracks).toEqual([]);
    expect(r.rerouted_trains).toEqual([]);
    expect(r.held_trains).toEqual([]);
    expect(r.agent_outputs.routing.affected_trains).toEqual([]);
    expect(r.agent_outputs.routing.reroutes).toEqual([]);
    expect(r.agent_outputs.weather.high_risk_tracks).toEqual([]);
    expect(r.agent_outputs.track.actions).toEqual([]);
    expect(r.trains?.length).toBeGreaterThan(0);
  });
});

describe("runSimulation", () => {
  it("falls back to a mock shaped by mockTrackId when the backend is offline", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    const run = await runSimulation("T11");
    expect(run.failed_tracks).toContain("T11");
    expect(run.twin_source).toBeTruthy();
  });

  it("falls back to the default incident with no hint", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    const run = await runSimulation();
    expect(run.failed_tracks).toContain("T23");
  });

  it("falls back to a weather-shaped mock when the injection was a storm", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    const run = await runSimulation("T11", "weather");
    expect(run.failed_tracks).toContain("T11");
    expect(run.agent_outputs.weather.strategy).toBe("W1");
    expect(run.execution_log[0].message).toContain("STORM reported on T11");
  });
});
