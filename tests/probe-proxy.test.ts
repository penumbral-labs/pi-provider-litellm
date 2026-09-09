import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";

const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-probe-"));
process.env.PI_CODING_AGENT_DIR = agentDir;
vi.mock("@earendil-works/pi-coding-agent", () => ({
  getAgentDir: () => process.env.PI_CODING_AGENT_DIR ?? agentDir,
}));

import {
  acceptanceOracle,
  chatReasoningCarrier,
  compareLive,
  errorClass,
  normalizeBaseUrl,
  type ProbeSnapshot,
  parseProbeArgs,
  probeDiscovery,
  protocolPrediction,
  publicCatalogOptions,
  readSnapshot,
  reasoningPrediction,
  runLiveMatrix,
} from "../scripts/probe-proxy.js";

const snapshot: ProbeSnapshot = {
  modelInfo: {
    data: [
      {
        model_name: "route-gpt",
        litellm_params: { model: "azure/gpt-5", allowed_openai_params: ["reasoning_effort"] },
        model_info: {
          base_model: "openai/gpt-5",
          litellm_provider: "azure",
          supports_reasoning: true,
          supports_low_reasoning_effort: true,
          max_input_tokens: 1000,
          max_output_tokens: 100,
        },
      },
    ],
  },
  modelGroupInfo: { data: [] },
  models: { data: [{ id: "route-gpt", owned_by: "openai" }] },
};

afterAll(async () => {
  if (originalAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
  else process.env.PI_CODING_AGENT_DIR = originalAgentDir;
  await rm(agentDir, { recursive: true, force: true });
});

afterEach(() => vi.unstubAllGlobals());

const probeModel = {
  id: "route-gpt",
  deployments: 1,
  publicSources: [],
  liteLLMFlags: {},
  api: "openai-completions",
  reasoning: true,
  thinkingLevelMap: { low: "low", high: null, xhigh: null },
  limits: { context: 1000, output: 100 },
  cost: {},
  predictions: { protocol: "openai-responses", reasoning: { low: true, high: false } },
};

describe("parseProbeArgs", () => {
  it("parses snapshot and bounded live options", () => {
    expect(
      parseProbeArgs([
        "--base-url",
        "https://proxy.example/v1/",
        "--api-key",
        "secret",
        "--src",
        "/tmp/tree/src",
        "--live",
        "--max-requests",
        "12",
      ]),
    ).toMatchObject({ baseUrl: "https://proxy.example", src: "/tmp/tree/src", live: true, maxRequests: 12 });
  });

  it("normalizes proxy roots", () => {
    expect(normalizeBaseUrl("https://proxy.example/v1/")).toBe("https://proxy.example");
  });
});

describe("readSnapshot", () => {
  it("does not infer companions from a filename without model-info", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-snapshot-"));
    const file = join(dir, "snapshot.json");
    const value = { data: [{ model_name: "route-gpt" }] };
    await writeFile(file, JSON.stringify(value));

    await expect(readSnapshot(file)).resolves.toEqual({ modelInfo: value });
    expect(JSON.parse(await readFile(file, "utf8"))).toEqual(value);
  });
});

describe("predictions", () => {
  it("treats the Azure GA boundary as Responses-compatible", () => {
    expect(
      protocolPrediction({
        model_name: "route-gpt",
        litellm_params: { model: "azure/gpt-5", api_version: "2025-03-01" } as never,
        model_info: { base_model: "openai/gpt-5", litellm_provider: "azure" },
      }),
    ).toBe("openai-responses");
  });

  it.each(["responses", "response", "Responses"])("predicts Responses for a %s mode row", (mode) => {
    expect(protocolPrediction({ model_name: "route", model_info: { mode } as never })).toBe("openai-responses");
  });

  it("predicts rejection when reasoning_effort is unavailable", () => {
    expect(reasoningPrediction({ model_name: "route-gpt" }, ["low", "medium", "high"])).toMatchObject({
      low: false,
      medium: false,
      high: false,
    });
  });

  it("treats omitted standard levels as unavailable when a public effort list exists", () => {
    expect(
      reasoningPrediction(
        {
          model_name: "route-gpt",
          litellm_params: { model: "openai/route-gpt", allowed_openai_params: ["reasoning_effort"] } as never,
        },
        ["low", "high"],
      ),
    ).toMatchObject({ off: false, minimal: false, low: true, medium: false, high: true });
  });

  it("ignores public effort lists with no recognized values", () => {
    expect(
      reasoningPrediction(
        {
          model_name: "route-gpt",
          litellm_params: { model: "openai/route-gpt", allowed_openai_params: ["reasoning_effort"] } as never,
        },
        ["adaptive"],
      ),
    ).toMatchObject({ off: true, minimal: true, low: true, medium: true, high: true });
  });

  it("seeds off and minimal as selectable when reasoning_effort is available", () => {
    expect(
      reasoningPrediction(
        {
          model_name: "route-gpt",
          litellm_params: { model: "openai/route-gpt", allowed_openai_params: ["reasoning_effort"] } as never,
        },
        undefined,
      ),
    ).toMatchObject({ off: true, minimal: true, low: true, medium: true, high: true });
  });

  it.each([
    [false, false],
    [true, true],
  ])("maps supports_none_reasoning_effort=%s to off=%s", (flag, expected) => {
    expect(
      reasoningPrediction(
        {
          model_name: "route-gpt",
          litellm_params: { model: "openai/route-gpt", allowed_openai_params: ["reasoning_effort"] } as never,
          model_info: { supports_none_reasoning_effort: flag } as never,
        },
        undefined,
      ).off,
    ).toBe(expected);
  });
});

describe("probeDiscovery", () => {
  it("uses discovery's models.dev cache path for snapshot and live reports", () => {
    const cachePath = join(getAgentDir(), "litellm-models-dev.json");
    expect(publicCatalogOptions(undefined)).toEqual({ cachePath });
    expect(publicCatalogOptions(snapshot)).toEqual({ offline: true, cachePath });
  });

  it("isolates snapshot catalog evidence from the home agent directory", async () => {
    const cachePath = join(agentDir, "litellm-models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: Date.now(),
        catalog: {
          openai: {
            models: {
              "gpt-5": { reasoning_options: { type: "effort", values: ["minimal"] } },
            },
          },
        },
      }),
    );
    const dir = await mkdtemp(join(tmpdir(), "probe-isolated-catalog-"));
    const file = join(dir, "snapshot.json");
    await writeFile(file, JSON.stringify(snapshot));

    const report = await probeDiscovery({ snapshot: file, src: join(process.cwd(), "src") });

    expect(report.models[0]?.thinkingLevelMap?.minimal).toBeUndefined();
    expect(process.env.PI_CODING_AGENT_DIR).toBe(agentDir);
  });

  it("maps wildcard-expanded model ids back to wildcard deployment rows", async () => {
    const wildcardSnapshot: ProbeSnapshot = {
      modelInfo: {
        data: [
          {
            model_name: "team/*",
            litellm_params: { model: "openai/*", allowed_openai_params: ["reasoning_effort"] },
            model_info: { supports_reasoning: true, supports_low_reasoning_effort: true },
          },
        ],
      },
      modelGroupInfo: { data: [] },
      models: { data: [{ id: "team/chat", owned_by: "openai" }] },
    };
    const dir = await mkdtemp(join(tmpdir(), "probe-wildcard-info-"));
    const file = join(dir, "snapshot.json");
    await writeFile(file, JSON.stringify(wildcardSnapshot));

    const report = await probeDiscovery({ snapshot: file, src: join(process.cwd(), "src") });

    expect(report.models.find((model) => model.id === "team/chat")).toMatchObject({
      deployments: 1,
      liteLLMFlags: { supports_low_reasoning_effort: true },
    });
  });

  it.each([false, true])("reports supports_none_reasoning_effort=%s and predicts off consistently", async (flag) => {
    const noneFlagSnapshot: ProbeSnapshot = {
      modelInfo: {
        data: [
          {
            model_name: "route-gpt",
            litellm_params: { model: "openai/gpt-5", allowed_openai_params: ["reasoning_effort"] },
            model_info: { supports_reasoning: true, supports_none_reasoning_effort: flag },
          },
        ],
      },
      modelGroupInfo: { data: [] },
      models: { data: [] },
    };
    const dir = await mkdtemp(join(tmpdir(), "probe-none-flag-"));
    const file = join(dir, "snapshot.json");
    await writeFile(file, JSON.stringify(noneFlagSnapshot));

    const report = await probeDiscovery({ snapshot: file, src: join(process.cwd(), "src") });

    expect(report.models[0]).toMatchObject({
      liteLLMFlags: { supports_none_reasoning_effort: flag },
      predictions: { reasoning: { off: flag } },
    });
  });

  it("reduces duplicate deployment report evidence conservatively", async () => {
    const duplicateSnapshot: ProbeSnapshot = {
      modelInfo: {
        data: [
          {
            model_name: "route-gpt",
            litellm_params: { model: "openai/gpt-5", allowed_openai_params: ["reasoning_effort"] },
            model_info: {
              mode: "responses",
              supports_low_reasoning_effort: true,
              supports_xhigh_reasoning_effort: true,
            },
          },
          {
            model_name: "route-gpt",
            litellm_params: { model: "anthropic/claude-opus-5" },
            model_info: { mode: "chat", supports_low_reasoning_effort: false },
          },
        ],
      },
      modelGroupInfo: { data: [] },
      models: { data: [] },
    };
    const dir = await mkdtemp(join(tmpdir(), "probe-duplicate-info-"));
    const file = join(dir, "snapshot.json");
    await writeFile(file, JSON.stringify(duplicateSnapshot));

    const report = await probeDiscovery({ snapshot: file, src: join(process.cwd(), "src") });

    expect(report.models[0]).toMatchObject({
      deployments: 2,
      publicSources: [],
      liteLLMFlags: { supports_low_reasoning_effort: false, supports_xhigh_reasoning_effort: false },
      predictions: {
        protocol: "openai-completions",
        reasoning: { low: false, xhigh: false },
      },
    });
    expect(report.models[0]).not.toHaveProperty("identity");
  });

  it.each([
    { mode: "snapshot", snapshot: true, expectedModelsDev: false },
    { mode: "live", snapshot: false, expectedModelsDev: undefined },
  ])(
    "passes the expected models.dev setting to $mode discovery",
    async ({ snapshot: useSnapshot, expectedModelsDev }) => {
      const dir = await mkdtemp(join(tmpdir(), "probe-discovery-options-"));
      const sourceDir = join(dir, "src");
      await mkdir(sourceDir);
      await writeFile(
        join(sourceDir, "discover.ts"),
        [
          "export async function discoverModels(baseUrl, apiKey, options) {",
          "  globalThis.__probeDiscoveryOptions = options;",
          '  return { source: "test", models: [] };',
          "}",
        ].join("\n"),
      );
      const file = join(dir, "snapshot.json");
      await writeFile(file, JSON.stringify({ modelInfo: { data: [] } }));
      if (!useSnapshot) {
        vi.stubGlobal("fetch", async (input: string | URL | Request) => {
          const url = String(input);
          if (url.endsWith("/model/info")) return new Response(JSON.stringify({ data: [] }));
          if (url === "https://models.dev/api.json") return new Response(JSON.stringify({}));
          throw new Error(`unexpected URL: ${url}`);
        });
      }

      await probeDiscovery(
        useSnapshot
          ? { snapshot: file, src: dir }
          : { baseUrl: "https://proxy.example/v1", apiKey: "secret", src: dir },
      );

      expect(
        (globalThis as typeof globalThis & { __probeDiscoveryOptions?: Record<string, unknown> })
          .__probeDiscoveryOptions,
      ).toEqual({ silent: true, modelsDev: expectedModelsDev });
      delete (globalThis as typeof globalThis & { __probeDiscoveryOptions?: Record<string, unknown> })
        .__probeDiscoveryOptions;
    },
  );

  it("fails when model info is empty for a non-empty /model/info discovery", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-empty-info-"));
    const sourceDir = join(dir, "src");
    await mkdir(sourceDir);
    const discoverFile = join(sourceDir, "discover.ts");
    await writeFile(
      discoverFile,
      'export async function discoverModels() { return { source: "model_info", models: [{ id: "route-gpt", api: "openai-completions", contextWindow: 1000, maxTokens: 100, cost: {} }] }; }\n',
    );
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ data: [] }), { status: 200 }));

    await expect(probeDiscovery({ baseUrl: "https://proxy.example/v1", apiKey: "secret", src: dir })).rejects.toThrow(
      "/model/info returned zero rows for 1 discovered models",
    );
  });

  it("bounds the follow-up /model/info read", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-bounded-info-"));
    const sourceDir = join(dir, "src");
    await mkdir(sourceDir);
    await writeFile(
      join(sourceDir, "discover.ts"),
      'export async function discoverModels() { return { source: "model_info", models: [] }; }\n',
    );
    const signals: Array<AbortSignal | null | undefined> = [];
    vi.stubGlobal("fetch", async (input: string | URL | Request, init?: RequestInit) => {
      if (String(input).endsWith("/model/info")) signals.push(init?.signal);
      return new Response(JSON.stringify({}), { status: 200 });
    });

    await probeDiscovery({ baseUrl: "https://proxy.example/v1", apiKey: "secret", src: dir });

    expect(signals).toHaveLength(1);
    expect(signals[0]).toBeInstanceOf(AbortSignal);
  });

  it("reports a /v1/models fallback discovery from route-only rows", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-fallback-info-"));
    const sourceDir = join(dir, "src");
    await mkdir(sourceDir);
    await writeFile(
      join(sourceDir, "discover.ts"),
      'export async function discoverModels() { return { source: "models_list", models: [{ id: "route-gpt", api: "openai-completions", reasoning: false, contextWindow: 1000, maxTokens: 100, cost: {} }] }; }\n',
    );
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 }));

    const report = await probeDiscovery({ baseUrl: "https://proxy.example/v1", apiKey: "secret", src: dir });

    expect(report.source).toBe("models_list");
    expect(report.models).toMatchObject([{ id: "route-gpt", deployments: 1, publicSources: [] }]);
  });

  it("injects snapshot fetch and reports identity, flags, selected metadata, and predictions", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-proxy-"));
    const file = join(dir, "snapshot.json");
    await writeFile(file, JSON.stringify(snapshot));

    const report = await probeDiscovery({ snapshot: file, src: join(process.cwd(), "src") });

    expect(report.source).toBe("model_info");
    expect(report.models).toHaveLength(1);
    expect(report.models[0]).toMatchObject({
      id: "route-gpt",
      deployments: 1,
      identity: { provider: "openai", modelId: "gpt-5", family: "openai" },
      liteLLMFlags: { supports_low_reasoning_effort: true },
      api: "openai-completions",
      reasoning: true,
      limits: { context: 1000, output: 100 },
      predictions: { protocol: "openai-responses", reasoning: { low: true, medium: true, high: true } },
    });
  });
});

describe("live outcomes", () => {
  it("routes Responses models through /v1/responses with the Responses request body", async () => {
    const requests: Array<{ url: string; body: unknown }> = [];
    vi.stubGlobal("fetch", async (input: string | URL | Request, init?: RequestInit) => {
      requests.push({ url: String(input), body: JSON.parse(String(init?.body)) });
      return new Response(JSON.stringify({ output: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    const results = await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [{ ...probeModel, id: "responses-model", api: "openai-responses" }],
      { levels: ["off", "high"] },
    );

    expect(requests).toEqual([
      {
        url: "https://proxy.example/v1/responses",
        body: {
          model: "responses-model",
          input: "Reply with one word.",
          max_output_tokens: 16,
          reasoning: { effort: "none" },
        },
      },
      {
        url: "https://proxy.example/v1/responses",
        body: {
          model: "responses-model",
          input: "Reply with one word.",
          max_output_tokens: 16,
          reasoning: { effort: "high" },
        },
      },
    ]);
    expect(results).toMatchObject([
      { path: "responses", model: "responses-model", level: "off", status: 200, accepted: true },
      { path: "responses", model: "responses-model", level: "high", status: 200, accepted: true },
    ]);
  });

  it("sends the production thinking carrier for a deepseek-format Chat model without effort support", async () => {
    const requests: unknown[] = [];
    vi.stubGlobal("fetch", async (_input: string | URL | Request, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({ choices: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [{ ...probeModel, id: "kimi-thinking", compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false } }],
      { levels: ["off", "high"] },
    );

    expect(requests).toEqual([
      {
        model: "kimi-thinking",
        max_tokens: 16,
        messages: [{ role: "user", content: "Reply with one word." }],
        thinking: { type: "disabled" },
      },
      {
        model: "kimi-thinking",
        max_tokens: 16,
        messages: [{ role: "user", content: "Reply with one word." }],
        thinking: { type: "enabled" },
      },
    ]);
  });

  it("applies discovered visibility policy to the live Chat request", async () => {
    const requests: Record<string, unknown>[] = [];
    vi.stubGlobal("fetch", async (_input: string | URL | Request, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({ choices: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    const moonshotPolicy = {
      normalizeStrictToolMessages: true,
      normalizeThinkTags: true,
      suppressReasoningVisibility: true,
    };
    const hostedKimiPolicy = {
      normalizeStrictToolMessages: true,
      normalizeThinkTags: true,
      suppressReasoningVisibility: false,
    };
    await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [
        { ...probeModel, id: "moonshot-route" },
        { ...probeModel, id: "azure-kimi-route" },
      ],
      { levels: ["low"] },
      new Map([
        ["moonshot-route", moonshotPolicy],
        ["azure-kimi-route", hostedKimiPolicy],
      ]),
    );

    expect(requests[0]).toMatchObject({
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });
    expect(requests[1]).not.toHaveProperty("include_reasoning");
    expect(requests[1]).not.toHaveProperty("reasoning_content");
    expect(requests[1]).not.toHaveProperty("merge_reasoning_content_in_choices");
  });

  it("applies discovered level mappings to Chat and Responses probes", async () => {
    const requests: Array<{ url: string; body: Record<string, unknown> }> = [];
    vi.stubGlobal("fetch", async (input: string | URL | Request, init?: RequestInit) => {
      requests.push({ url: String(input), body: JSON.parse(String(init?.body)) });
      return new Response(JSON.stringify({ choices: [], output: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [
        { ...probeModel, id: "kimi-k3-chat", thinkingLevelMap: { max: "high" } },
        { ...probeModel, id: "kimi-k3-responses", api: "openai-responses", thinkingLevelMap: { max: "high" } },
      ],
      { levels: ["max"] },
    );

    expect(requests[0]?.body).toMatchObject({ reasoning_effort: "high" });
    expect(requests[1]?.body).toMatchObject({ reasoning: { effort: "high" } });
  });

  it("sends raw denied levels so the live probe verifies rejection", async () => {
    const requests: Record<string, unknown>[] = [];
    vi.stubGlobal("fetch", async (_input: string | URL | Request, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({ choices: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [{ ...probeModel, id: "denied-max", thinkingLevelMap: { max: null } }],
      { levels: ["max"] },
    );

    expect(requests[0]).toMatchObject({ reasoning_effort: "max" });
  });

  it("sends both Chat carriers when a deepseek-format model also accepts reasoning_effort", () => {
    expect(chatReasoningCarrier("low", { thinkingFormat: "deepseek", supportsReasoningEffort: true })).toEqual({
      thinking: { type: "enabled" },
      reasoning_effort: "low",
    });
    expect(chatReasoningCarrier("max", undefined, { max: "high" })).toEqual({ reasoning_effort: "high" });
    expect(chatReasoningCarrier("off")).toEqual({ reasoning_effort: "none" });
  });

  it("skips every level for a model discovered without reasoning", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const results = await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [{ ...probeModel, id: "plain", reasoning: false, thinkingLevelMap: undefined }],
      { levels: ["low", "high"] },
    );

    expect(results).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("skips Chat levels for a model with no reasoning carrier", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    const results = await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [{ ...probeModel, id: "closed", compat: { supportsReasoningEffort: false } }],
      { levels: ["low", "high"] },
    );

    expect(results).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("uses the production Anthropic messages carriers for disabled and adaptive thinking", async () => {
    const requests: unknown[] = [];
    vi.stubGlobal("fetch", async (_input: string | URL | Request, init?: RequestInit) => {
      requests.push(JSON.parse(String(init?.body)));
      return new Response(JSON.stringify({ content: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await runLiveMatrix(
      "https://proxy.example",
      "secret",
      [{ ...probeModel, id: "messages-model", api: "anthropic-messages" }],
      { levels: ["off", "high"] },
    );

    expect(requests).toEqual([
      {
        model: "messages-model",
        max_tokens: 16,
        messages: [{ role: "user", content: "Reply with one word." }],
        thinking: { type: "disabled" },
      },
      {
        model: "messages-model",
        max_tokens: 16,
        messages: [{ role: "user", content: "Reply with one word." }],
        thinking: { type: "adaptive" },
        output_config: { effort: "high" },
      },
    ]);
  });

  it("classifies LiteLLM errors from message objects and raw strings", () => {
    expect(errorClass({ error: { code: "400", message: "litellm.UnsupportedParamsError: unsupported" } })).toBe(
      "UnsupportedParamsError",
    );
    expect(errorClass('{"message":"litellm.UnsupportedParamsError: unsupported"}')).toBe("UnsupportedParamsError");
  });

  it("derives classified acceptance without classifying unknown outcomes", () => {
    expect(acceptanceOracle({ status: 200 })).toBe(true);
    expect(acceptanceOracle({ status: 400, errorClass: "UnsupportedParamsError" })).toBe(false);
    expect(acceptanceOracle({ status: 500, errorClass: "InternalServerError" })).toBeNull();
  });

  it("records unclassified outcomes and applies soundness and completeness", () => {
    expect(
      compareLive(
        [
          {
            path: "chat",
            model: "route-gpt",
            level: "medium",
            status: 500,
            errorClass: "InternalServerError",
            accepted: null,
          },
          { path: "chat", model: "route-gpt", level: "low", status: 400, accepted: false },
          { path: "chat", model: "route-gpt", level: "high", status: 200, accepted: true },
          { path: "chat", model: "route-gpt", level: "xhigh", status: 200, accepted: true },
        ],
        [probeModel],
      ),
    ).toEqual({
      mismatches: [
        "route-gpt chat medium: unclassified HTTP 500 InternalServerError",
        "route-gpt chat low: predicted selectable, actual rejected",
        "route-gpt chat high: accepted but not predicted selectable",
      ],
      informational: ["route-gpt chat xhigh: accepted but not offered"],
    });
  });

  it("reports minimal soundness mismatches and informational completeness gaps", () => {
    const minimalModel = {
      ...probeModel,
      thinkingLevelMap: { minimal: "minimal" },
      predictions: { ...probeModel.predictions, reasoning: { minimal: true } },
    };
    const minimalNotOfferedModel = {
      ...probeModel,
      thinkingLevelMap: { minimal: null },
      predictions: { ...probeModel.predictions, reasoning: { minimal: false } },
    };

    expect(
      compareLive(
        [{ path: "chat", model: "route-gpt", level: "minimal", status: 400, accepted: false }],
        [minimalModel],
      ).mismatches,
    ).toEqual(["route-gpt chat minimal: predicted selectable, actual rejected"]);
    expect(
      compareLive(
        [{ path: "chat", model: "route-gpt", level: "minimal", status: 200, accepted: true }],
        [minimalNotOfferedModel],
      ),
    ).toEqual({
      mismatches: [],
      informational: ["route-gpt chat minimal: accepted but not offered"],
    });
  });

  it("treats accepted but unoffered off as generation-contract information", () => {
    expect(
      compareLive(
        [{ path: "chat", model: "route-gpt", level: "off", status: 200, accepted: true }],
        [{ ...probeModel, thinkingLevelMap: { off: null } }],
      ),
    ).toEqual({
      mismatches: [],
      informational: ["route-gpt chat off: accepted but not offered"],
    });
  });

  it("reports accepted but unoffered Responses extension levels as informational", () => {
    expect(
      compareLive(
        [{ path: "responses", model: "route-gpt", level: "xhigh", status: 200, accepted: true }],
        [probeModel],
      ),
    ).toEqual({
      mismatches: [],
      informational: ["route-gpt responses xhigh: accepted but not offered"],
    });
  });

  it("reports oracle rows whose model is missing from discovery", () => {
    expect(
      compareLive([{ path: "chat", model: "missing-model", level: "low", status: 400, accepted: false }], [probeModel]),
    ).toEqual({
      mismatches: ["missing-model chat low: model missing from discovery"],
      informational: [],
    });
  });

  it("matches the committed production acceptance oracle with four informational xhigh rows", async () => {
    const report = await probeDiscovery({
      snapshot: join(process.cwd(), "tests/fixtures/proxy/prod-model-info-2026-09-04.json"),
    });
    const oracle = JSON.parse(
      await readFile(join(process.cwd(), "tests/fixtures/proxy/expected-acceptance-2026-09-04.json"), "utf8"),
    );

    expect(compareLive(oracle, report.models)).toEqual({
      mismatches: [],
      informational: [
        "o3 chat xhigh: accepted but not offered",
        "deepseek-v4-flash chat xhigh: accepted but not offered",
        "kimi-k3 chat xhigh: accepted but not offered",
        "glm-5 chat xhigh: accepted but not offered",
      ],
    });
  });
});
