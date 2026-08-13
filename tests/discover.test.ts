import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getSupportedThinkingLevels } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildCompat,
  discoverModels,
  enrichCachedModel,
  moonshotPolicy,
  normalizeBaseUrl,
  resolveModelInfoCatalog,
} from "../src/discover.js";
import type { DiscoveredModel } from "../src/types.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

// Discovered models become native models before pi-ai reads them, so level
// assertions go through the same shape the runtime sees.
function nativeModel(model: DiscoveredModel | undefined): Model<Api> {
  if (!model) throw new Error("expected a discovered model");
  return {
    ...model,
    provider: "litellm",
    api: model.api ?? "openai-completions",
    baseUrl: "https://x/v1",
  } as Model<Api>;
}

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const MODELS_DEV_CATALOG = {
  openai: {
    models: {
      "gpt-5.5": {
        name: "Models.dev GPT-5.5",
        reasoning: true,
        limit: { context: 1_050_000, output: 128_000 },
        cost: { input: 5, output: 30, cache_read: 0.5 },
      },
    },
  },
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe("normalizeBaseUrl", () => {
  it("strips trailing slashes", () => {
    expect(normalizeBaseUrl("https://x.example.com/")).toBe("https://x.example.com");
    expect(normalizeBaseUrl("https://x.example.com///")).toBe("https://x.example.com");
  });

  it("strips a single trailing /v1 suffix", () => {
    expect(normalizeBaseUrl("https://x.example.com/v1")).toBe("https://x.example.com");
    expect(normalizeBaseUrl("https://x.example.com/v1/")).toBe("https://x.example.com");
  });

  it("is case-insensitive on /v1", () => {
    expect(normalizeBaseUrl("https://x.example.com/V1")).toBe("https://x.example.com");
  });

  it("does not strip /v2 or /v1xxx", () => {
    expect(normalizeBaseUrl("https://x.example.com/v2")).toBe("https://x.example.com/v2");
    expect(normalizeBaseUrl("https://x.example.com/v1beta")).toBe("https://x.example.com/v1beta");
  });

  it("preserves a base path that is not /v1", () => {
    expect(normalizeBaseUrl("https://x.example.com/proxy")).toBe("https://x.example.com/proxy");
  });
});

describe("buildCompat", () => {
  it("returns supportsStore: false for non-anthropic models", () => {
    expect(buildCompat("openai/gpt-4o")).toEqual({ supportsStore: false });
    expect(buildCompat("gemini/gemini-2.0-flash")).toEqual({ supportsStore: false });
    expect(buildCompat("gpt-5.5")).toEqual({ supportsStore: false });
  });

  it("adds Moonshot-compatible tool calling flags for Kimi models", () => {
    expect(buildCompat("kimi-k2.6")).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
    expect(buildCompat("moonshotai/kimi-k2")).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
  });

  it("adds cacheControlFormat for anthropic-prefixed models", () => {
    expect(buildCompat("anthropic/claude-3-5-sonnet")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });

  it("adds cacheControlFormat for bare Claude aliases", () => {
    for (const id of ["claude-3-5-sonnet", "opus-4.7", "sonnet-4.6", "haiku-4.5"]) {
      expect(buildCompat(id)).toEqual({
        supportsStore: false,
        cacheControlFormat: "anthropic",
      });
    }
  });

  it("adds cacheControlFormat for routed Anthropic aliases", () => {
    expect(buildCompat("google/claude-sonnet-4-6")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });

  it("does not match non-Anthropic tokens that start with Anthropic family names", () => {
    expect(buildCompat("openai/sonnetic-gpt")).toEqual({ supportsStore: false });
    expect(buildCompat("vendor/opusflow")).toEqual({ supportsStore: false });
  });

  it("matches case-insensitively", () => {
    expect(buildCompat("Opus-4.7")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
    expect(buildCompat("CLAUDE-3-5-SONNET")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });
});

describe("moonshotPolicy", () => {
  it("keeps route-name evidence response-only unless deployment evidence authorizes the repair", () => {
    expect(moonshotPolicy("kimi-k2.6")).toEqual({ normalizeStrictToolMessages: false, normalizeThinkTags: true });
    expect(moonshotPolicy("moonshotai/kimi-k2", true)).toEqual({
      normalizeStrictToolMessages: true,
      normalizeThinkTags: true,
    });
  });

  it("leaves forced-thinking generations alone, which stream reasoning as its own field", () => {
    expect(moonshotPolicy("kimi-k2-thinking")).toEqual({
      normalizeStrictToolMessages: false,
      normalizeThinkTags: false,
    });
  });
});

describe("discoverModels via /model/info", () => {
  it("keeps models with a null mode", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "local/model", model_info: { mode: null } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["local/model"]);
  });

  it("parses a /model/info success response with cost mapping", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                max_input_tokens: 200000,
                max_output_tokens: 8192,
                supports_vision: true,
                supports_reasoning: false,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
            {
              model_name: "openai/gpt-4o",
              model_info: {
                mode: "chat",
                max_input_tokens: 128000,
                max_output_tokens: 16384,
              },
            },
            {
              model_name: "openai/text-embedding-3-large",
              model_info: { mode: "embedding" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    // embedding model filtered out by mode !== "chat"
    expect(result.models).toHaveLength(2);

    const anthropic = result.models.find((m) => m.id === "anthropic/claude-3-5-sonnet");
    expect(anthropic).toMatchObject({
      id: "anthropic/claude-3-5-sonnet",
      name: "anthropic/claude-3-5-sonnet",
      contextWindow: 200000,
      maxTokens: 8192,
      input: ["text", "image"],
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
    // cost is per-token in LiteLLM, per-million-tokens in pi-ai
    expect(anthropic?.cost).toEqual({ input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 });

    const openai = result.models.find((m) => m.id === "openai/gpt-4o");
    expect(openai).toMatchObject({
      id: "openai/gpt-4o",
      input: ["text", "image"],
      compat: { supportsStore: false },
    });
  });

  it("uses catalog costs when /model/info omits costs for Anthropic aliases", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "opus-4.8", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models[0]).toMatchObject({ api: "openai-completions", compat: { cacheControlFormat: "anthropic" } });
    expect(result.models[0]?.cost).toEqual({ input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 });
  });

  it("preserves catalog pricing tiers for /model/info models", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "openai/gpt-5.5", model_info: { mode: "chat" } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.cost?.tiers).toEqual([
      { inputTokensAbove: 272000, input: 10, output: 45, cacheRead: 1, cacheWrite: 0 },
    ]);
  });

  it("preserves catalog max thinking metadata for /model/info models", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "openai/gpt-5.6-luna", model_info: { mode: "chat" } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.thinkingLevelMap).toMatchObject({ off: "none", xhigh: "xhigh", max: "max" });
  });

  it("reduces duplicate model ids conservatively instead of merging richer fields", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "custom-model", model_info: { mode: "chat" } },
            {
              model_name: "custom-model",
              model_info: {
                mode: "chat",
                max_input_tokens: 200000,
                max_output_tokens: 8192,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "custom-model",
      name: "custom-model (incomplete metadata)",
      contextWindow: 128000,
      maxTokens: 8192,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("falls back to Chat and group guarantees for mixed deployments regardless of row order", async () => {
    const deployments = [
      {
        model_name: "shared-route",
        litellm_params: { model: "openai/gpt-4o" },
        model_info: {
          id: "deployment-a",
          mode: "responses",
          supports_reasoning: true,
          supports_vision: true,
          max_input_tokens: 128000,
          max_output_tokens: 16384,
          input_cost_per_token: 0.000005,
          output_cost_per_token: 0.000015,
          cache_read_input_token_cost: 0.0000025,
          cache_creation_input_token_cost: 0,
        },
      },
      {
        model_name: "shared-route",
        litellm_params: { model: "internal/unknown" },
        model_info: {
          id: "deployment-b",
          mode: "chat",
          supports_reasoning: false,
          supports_vision: false,
          max_input_tokens: 64000,
          max_output_tokens: 8192,
        },
      },
    ];
    const fetchMock = vi.spyOn(globalThis, "fetch");
    for (const rows of [deployments, [...deployments].reverse()]) {
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { data: rows }));
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.models).toEqual([
        expect.objectContaining({
          id: "shared-route",
          name: "shared-route (incomplete metadata)",
          reasoning: false,
          input: ["text"],
          contextWindow: 64000,
          maxTokens: 8192,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        }),
      ]);
      expect(result.models[0]).toHaveProperty("api", "openai-completions");
      expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    }
  });

  it.each([
    ["moonshot", "moonshot/kimi-k2.6"],
    ["gemini", "gemini/gemini-3-pro-preview"],
    ["xai", "xai/grok-4.5"],
  ])("maps the %s adapter conservatively to its catalog", async (adapter, backend) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: `${adapter}-route`,
            litellm_params: { model: backend },
            model_info: { mode: "chat", litellm_provider: adapter },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.name).toBe(`${adapter}-route`);
    expect(result.models[0]?.name).not.toContain("no metadata");
    expect(result.models[0]?.contextWindow).toBeGreaterThan(128_000);
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
    expect(result.models[0]?.id).toBe(`${adapter}-route`);
  });

  it("trims backend candidates before catalog resolution", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "spacey",
            model_info: { mode: "chat", base_model: " openai/gpt-4o " },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ name: "spacey", input: ["text", "image"], contextWindow: 128_000 });
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
  });

  it.each([
    ["azure/gpt-4o", 128_000, 16_384, 2.5],
    ["azure/gpt-5", 400_000, 128_000, 1.25],
  ])(
    "enriches an opaque Azure deployment from base_model %s",
    async (baseModel, contextWindow, maxTokens, inputCost) => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse(200, {
          data: [
            {
              model_name: "azure-production",
              litellm_params: { model: "azure/prod-deployment-01" },
              model_info: { mode: "chat", litellm_provider: "azure", base_model: baseModel },
            },
          ],
        }),
      );

      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

      expect(result.models[0]).toMatchObject({
        id: "azure-production",
        input: ["text", "image"],
        contextWindow,
        maxTokens,
        cost: { input: inputCost },
      });
    },
  );

  it("enriches every deployment in an opaque Azure group from base_model", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: ["a", "b"].map((deployment) => ({
          model_name: "azure-production-group",
          litellm_params: { model: `azure/prod-deployment-${deployment}` },
          model_info: {
            id: deployment,
            mode: "chat",
            litellm_provider: "azure",
            base_model: "azure/gpt-4o",
          },
        })),
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "azure-production-group",
      input: ["text", "image"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 2.5, output: 10 },
    });
  });

  it("enriches an opaque Bedrock ARN with Claude cache compatibility", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "bedrock-production",
            litellm_params: {
              model: "bedrock/arn:aws:bedrock:us-east-1:123456789012:inference-profile/production",
            },
            model_info: {
              mode: "chat",
              litellm_provider: "bedrock",
              base_model: "anthropic.claude-sonnet-4-5-20250929-v1:0",
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "bedrock-production",
      api: "anthropic-messages",
      input: ["text", "image"],
      contextWindow: 200_000,
      maxTokens: 64_000,
      compat: { supportsStrictTools: true },
    });
  });

  it("does not use a qualified public route as evidence for a multi-deployment group", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.5",
            litellm_params: { model: "openai/gpt-5.5" },
            model_info: { id: "known", mode: "chat" },
          },
          {
            model_name: "openai/gpt-5.5",
            litellm_params: { model: "internal/mystery" },
            model_info: { id: "unknown", mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("treats repeated identified deployment rows as one effective singleton", async () => {
    const deployment = {
      model_name: "openai/gpt-5.5",
      model_info: { id: "deployment-a", mode: "chat" },
    };
    const fetchMock = vi.spyOn(globalThis, "fetch");
    for (const data of [[deployment], [deployment, deployment]]) {
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { data }));
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.models[0]).toMatchObject({
        id: "openai/gpt-5.5",
        name: "openai/gpt-5.5",
        reasoning: true,
        contextWindow: 272_000,
        maxTokens: 128_000,
      });
    }
  });

  it("collects semantic generation from base_model after catalog resolution", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "public-route",
        litellm_params: { model: "moonshot/kimi-k2-0905-preview" },
        model_info: { litellm_provider: "moonshot", base_model: "kimi-k3" },
      }),
    ).toMatchObject({ provider: "moonshotai", semanticFamily: "kimi", semanticModel: "kimi-k3" });
  });

  it("reports a contradictory routing model and base_model as conflicting evidence", () => {
    const conflicting = resolveModelInfoCatalog({
      model_name: "public-route",
      litellm_params: { model: "openai/gpt-proxy" },
      model_info: { litellm_provider: "openai", base_model: "kimi-k3" },
    });

    // Neither candidate is authoritative for a deployment that names two
    // different backend families, so nothing is resolved from either one.
    expect(conflicting).toEqual({ semanticFamily: "conflicting" });
  });

  it("pairs catalog metadata with the family of the candidate that resolved it", () => {
    // The routing model resolves nothing, so `base_model` supplies both the
    // catalog metadata and the family; taking the family from the other
    // candidate would silently drop Anthropic cache-control support.
    expect(
      resolveModelInfoCatalog({
        model_name: "gateway-route",
        litellm_params: { model: "gateway/claude-router-prod" },
        model_info: { litellm_provider: "anthropic", base_model: "claude-sonnet-4-6" },
      }),
    ).toMatchObject({ provider: "anthropic", semanticFamily: "claude" });
  });

  it("refuses route-name family inference when deployments disagree about the backend", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "kimi-looking-route",
            litellm_params: { model: "moonshot/kimi-k3" },
            model_info: { id: "kimi", mode: "chat" },
          },
          {
            model_name: "kimi-looking-route",
            litellm_params: { model: "openai/gpt-4o" },
            model_info: { id: "openai", mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    // Per-field conservative meet: the shape-changing fields are withheld because
    // the OpenAI candidate would reject them, the safety restrictions the Moonshot
    // candidate needs survive, and nothing is inferred from the route name.
    expect(result.models[0]?.compat).not.toHaveProperty("maxTokensField");
    expect(result.models[0]?.compat).not.toHaveProperty("cacheControlFormat");
    expect(result.models[0]?.compat).toMatchObject({ supportsReasoningEffort: false, supportsStrictMode: false });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("uses base_model semantic evidence when the routing model is absent", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "public-route",
        model_info: { litellm_provider: "moonshot", base_model: "kimi-k3" },
      }),
    ).toMatchObject({ provider: "moonshotai", semanticFamily: "kimi", semanticModel: "kimi-k3" });
  });

  it("derives DeepSeek policy from an opaque Azure Foundry deployment and base_model", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "foundry-route",
            litellm_params: { model: "azure_ai/deepseek-production", allowed_openai_params: ["reasoning_effort"] },
            model_info: {
              mode: "chat",
              litellm_provider: "azure_ai",
              base_model: "deepseek/deepseek-v4-pro",
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "foundry-route",
      reasoning: true,
      thinkingLevelMap: { off: null, high: "high", max: "max" },
      compat: { thinkingFormat: "openai", supportsReasoningEffort: true },
    });
  });

  it("derives DeepSeek family and accepted controls from Azure Foundry backend evidence", async () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "public-route-without-family-text",
        litellm_params: { model: " azure_ai/DeepSeek-V4 ", allowed_openai_params: ["reasoning_effort"] },
        model_info: { mode: "chat", litellm_provider: "azure_ai" },
      }),
    ).toEqual({ semanticFamily: "deepseek", semanticModel: "deepseek-v4" });

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "foundry-route",
            litellm_params: { model: " azure_ai/DeepSeek-V4 ", allowed_openai_params: ["reasoning_effort"] },
            model_info: { mode: "chat", litellm_provider: "azure_ai" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "foundry-route",
      name: "foundry-route (incomplete metadata)",
      reasoning: true,
      contextWindow: 128_000,
      thinkingLevelMap: { off: null, high: "high", max: "max" },
      compat: { thinkingFormat: "openai", supportsReasoningEffort: true },
    });
  });

  it.each([
    {
      name: "Kimi K2.6",
      backend: "moonshot/kimi-k2.6",
      params: ["thinking"],
      expected: {
        reasoning: true,
        thinkingLevelMap: { off: "off", high: "high" },
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
      },
    },
    {
      name: "Kimi K2.7 Code",
      backend: "moonshot/kimi-k2.7-code",
      params: ["thinking"],
      expected: {
        reasoning: true,
        thinkingLevelMap: { off: null, high: "high" },
        compat: { supportsReasoningEffort: false, requiresReasoningContentOnAssistantMessages: true },
      },
    },
    {
      name: "Kimi K3",
      backend: "moonshot/kimi-k3",
      params: ["reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: { off: null, low: "low", high: "high", max: "max" },
        compat: {
          thinkingFormat: "openai",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
    {
      name: "direct DeepSeek V4",
      backend: "deepseek/deepseek-v4",
      params: ["thinking", "reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: { off: "off", high: "high", max: "max" },
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: true },
      },
    },
  ])("discovers evidence-based reasoning policy for $name", async ({ backend, params, expected }) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "public-route",
            litellm_params: { model: backend },
            model_info: { id: "deployment", mode: "chat", supported_openai_params: params },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject(expected);
  });

  it("does not infer controls from a Kimi-looking public route", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "kimi-k3-public-alias",
            litellm_params: { model: "openai/internal-route" },
            model_info: { mode: "chat", litellm_provider: "openai", supports_reasoning: true },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ reasoning: true });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    expect(result.models[0]?.compat).toEqual({ supportsStore: false });
    expect(result.models[0]).not.toHaveProperty("litellmPolicy");
  });

  it("withholds route-text Kimi strict repair and keeps only the response-side repair", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, { data: [{ model_name: "kimi-k2.6", model_info: { mode: "chat" } }] }),
    );

    const evidenceFree = await discoverModels("https://litellm.example.com", "sk-test", {});
    // A singleton route can borrow catalog controls and limits, but its name is
    // still not deployment-family evidence authorizing an outbound rewrite.
    expect(evidenceFree.models[0]?.litellmPolicy).toEqual({
      normalizeStrictToolMessages: false,
      normalizeThinkTags: true,
    });

    fetchMock.mockResolvedValueOnce(
      jsonResponse(200, {
        data: [
          {
            model_name: "kimi-k2.6",
            litellm_params: { model: "openai/internal-route" },
            model_info: { mode: "chat", litellm_provider: "openai" },
          },
        ],
      }),
    );

    const misleadingAlias = await discoverModels("https://litellm.example.com", "sk-test", {});
    expect(misleadingAlias.models[0]).not.toHaveProperty("litellmPolicy");
  });

  it("preserves qualified catalog capability without speculative controls", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "deepseek-route",
            litellm_params: { model: "deepseek/deepseek-v4-pro" },
            model_info: { mode: "chat", litellm_provider: "deepseek" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      reasoning: true,
      compat: {
        supportsStore: false,
        supportsReasoningEffort: false,
        requiresReasoningContentOnAssistantMessages: true,
      },
    });
    const model = result.models[0];
    expect(model?.api).toBe("openai-completions");
    if (!model?.api) throw new Error("discovered model has no API");
    expect(
      getSupportedThinkingLevels({
        ...model,
        api: model.api,
        provider: "litellm",
        baseUrl: "https://litellm.example.com",
      }),
    ).toEqual([]);
    expect(model.compat).not.toHaveProperty("thinkingFormat");
  });

  it("preserves explicit reasoning capability without speculative controls", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "k3-route",
            litellm_params: { model: "moonshot/kimi-k3" },
            model_info: { mode: "chat", supports_reasoning: true },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      reasoning: true,
      compat: {
        supportsReasoningEffort: false,
        requiresReasoningContentOnAssistantMessages: true,
      },
    });
    const model = result.models[0];
    expect(model?.api).toBe("openai-completions");
    if (!model?.api) throw new Error("discovered model has no API");
    expect(
      getSupportedThinkingLevels({
        ...model,
        api: model.api,
        provider: "litellm",
        baseUrl: "https://litellm.example.com",
      }),
    ).toEqual([]);
    expect(model.compat).not.toHaveProperty("thinkingFormat");
  });

  it("honors explicit K2.7 Code reasoning denial", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "code-route",
            litellm_params: { model: "moonshot/kimi-k2.7-code" },
            model_info: { mode: "chat", supports_reasoning: false, supported_openai_params: ["thinking"] },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.reasoning).toBe(false);
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("does not enrich an unqualified route from an unrelated provider catalog", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      // Reduced groups never take the ` (no metadata)` sentinel, which is what
      // authorizes catalog re-derivation from the id on a later cache read.
      id: "gpt-4o",
      name: "gpt-4o (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 16384,
    });
  });
});

describe("discoverModels wildcard expansion via /v1/models", () => {
  it("expands a wildcard /model/info entry and drops the literal wildcard id", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "lemonade/*", model_info: { mode: "chat" } },
            { model_name: "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL", model_info: { mode: "chat" } },
          ],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [
            { id: "lemonade/*", object: "model", owned_by: "openai" },
            { id: "lemonade/Bonsai-1.7B-gguf", object: "model", owned_by: "openai" },
            { id: "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL", object: "model", owned_by: "openai" },
            { id: "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL", object: "model", owned_by: "openai" },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models.map((m) => m.id).sort()).toEqual([
      "lemonade/Bonsai-1.7B-gguf",
      "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL",
      "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL",
    ]);
    // the raw wildcard id must NOT surface as a selectable model
    expect(result.models.some((m) => m.id.includes("*"))).toBe(false);
    // the concrete /model/info entry is not duplicated by /v1/models
    expect(result.models.filter((m) => m.id === "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL")).toHaveLength(1);
    // /v1/models was actually queried (the expansion path ran)
    expect(urls.some((u) => u.endsWith("/v1/models"))).toBe(true);
  });

  it("does not query /v1/models when /model/info has no wildcards", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models.map((m) => m.id)).toEqual(["openai/gpt-4o"]);
    expect(urls.some((u) => u.endsWith("/v1/models"))).toBe(false);
  });
});

describe("discoverModels native Messages selection", () => {
  it("persists adaptive-thinking compatibility from a real-shaped Opus 5 backend", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "claude-opus-5",
            model_info: {
              id: "deployment-opus-5",
              mode: "chat",
              litellm_provider: "bedrock_converse",
              base_model: "bedrock/us.anthropic.claude-opus-5",
              supports_reasoning: true,
              supported_openai_params: ["thinking", "reasoning_effort"],
            },
            litellm_params: { model: "bedrock/us.anthropic.claude-opus-5" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "claude-opus-5",
      api: "anthropic-messages",
    });
    expect(result.models[0]?.compat).toEqual({
      forceAdaptiveThinking: true,
      supportsTemperature: false,
      supportsStrictTools: true,
    });
  });

  it("selects Messages only for homogeneous strongly evidenced Claude groups", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "bedrock-claude",
            model_info: { id: "bedrock-a", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-6" },
          },
          {
            model_name: "bedrock-claude",
            model_info: { id: "bedrock-b", mode: "chat", litellm_provider: "bedrock_converse" },
            litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-6" },
          },
          {
            model_name: "mixed-route",
            model_info: { id: "mixed-claude", mode: "chat", litellm_provider: "anthropic" },
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
          },
          {
            model_name: "mixed-route",
            model_info: { id: "mixed-openai", mode: "chat", litellm_provider: "openai" },
            litellm_params: { model: "openai/gpt-4o" },
          },
          {
            model_name: "unknown-route",
            model_info: { id: "unknown", mode: "chat" },
            litellm_params: { model: "internal/private" },
          },
          {
            model_name: "claude-responses",
            model_info: { id: "responses", mode: "responses", litellm_provider: "anthropic" },
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
          },
          { model_name: "claude-public-only", model_info: { id: "public", mode: "chat" } },
          {
            model_name: "contradicted-claude",
            model_info: { id: "contradicted", mode: "chat", litellm_provider: "openai" },
            litellm_params: { model: "openai/claude-sonnet-4-6" },
          },
          {
            model_name: "anthropic-claude",
            model_info: { id: "anthropic", mode: "chat", litellm_provider: "anthropic" },
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
          },
          {
            model_name: "bedrock-adapter-claude",
            model_info: { id: "bedrock", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-6" },
          },
          {
            model_name: "bedrock-converse-claude",
            model_info: { id: "bedrock-converse", mode: "chat", litellm_provider: "bedrock_converse" },
            litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-6" },
          },
          {
            model_name: "vertex-claude",
            model_info: { id: "vertex", mode: "chat", litellm_provider: "vertex_ai" },
            litellm_params: { model: "vertex_ai/claude-sonnet-4-6" },
          },
          {
            model_name: "custom-claude",
            model_info: { id: "custom", mode: "chat", litellm_provider: "custom" },
            litellm_params: { model: "custom/claude-sonnet-4-6" },
          },
          {
            model_name: "known-nonclaude-prefix",
            model_info: { id: "known-prefix", mode: "chat" },
            litellm_params: { model: "openai/claude-sonnet-4-6" },
          },
          {
            model_name: "anthropic-unprefixed-claude",
            model_info: { id: "anthropic-unprefixed", mode: "chat", litellm_provider: "anthropic" },
            litellm_params: { model: "claude-3-5-sonnet-20241022" },
          },
          {
            model_name: "bedrock-unprefixed-claude",
            model_info: {
              id: "bedrock-unprefixed",
              mode: "chat",
              litellm_provider: "bedrock",
              base_model: "anthropic.claude-3-5-sonnet-20241022-v2:0",
            },
          },
          {
            model_name: "adapter-only-anthropic",
            model_info: { id: "adapter-only", mode: "chat", litellm_provider: "anthropic" },
            litellm_params: { model: "internal/private" },
          },
          {
            model_name: "adapter-absent-claude",
            model_info: { id: "adapter-absent", mode: "chat" },
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
          },
          {
            model_name: "anthropic-nonclaude",
            model_info: { id: "anthropic-nonclaude", mode: "chat", litellm_provider: "anthropic" },
            litellm_params: { model: "anthropic/amazon.nova-pro-v1:0" },
          },
          {
            model_name: "bedrock-nova-claude-base",
            model_info: {
              id: "nova-conflict",
              mode: "chat",
              litellm_provider: "bedrock",
              base_model: "anthropic/claude-3-5-sonnet-20241022",
            },
            litellm_params: { model: "bedrock/amazon.nova-pro-v1:0" },
          },
          {
            model_name: "azure-claude-base",
            model_info: {
              id: "azure-conflict",
              mode: "chat",
              litellm_provider: "anthropic",
              base_model: "vertex_ai/claude-sonnet-4@20250514",
            },
            litellm_params: { model: "azure/deployment-x" },
          },
          {
            model_name: "missing-mode-claude",
            model_info: { id: "missing-mode", litellm_provider: "anthropic" },
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
          },
          {
            model_name: "bedrock-nova",
            model_info: { id: "nova", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/amazon.nova-pro-v1:0" },
          },
          {
            model_name: "bedrock-converse-llama",
            model_info: { id: "llama", mode: "chat", litellm_provider: "bedrock_converse" },
            litellm_params: { model: "bedrock/meta.llama3-3-70b-instruct-v1:0" },
          },
          {
            model_name: "bedrock-mistral",
            model_info: { id: "mistral", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/mistral.mistral-large-2402-v1:0" },
          },
          {
            model_name: "bedrock-titan",
            model_info: { id: "titan", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/amazon.titan-text-premier-v1:0" },
          },
          {
            model_name: "bedrock-cohere",
            model_info: { id: "cohere", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/cohere.command-r-plus-v1:0" },
          },
          {
            model_name: "vertex-llama",
            model_info: { id: "vertex-llama", mode: "chat", litellm_provider: "vertex_ai" },
            litellm_params: { model: "vertex_ai/meta/llama-3.1-405b-instruct-maas" },
          },
          {
            model_name: "vertex-mistral",
            model_info: { id: "vertex-mistral", mode: "chat", litellm_provider: "vertex_ai" },
            litellm_params: { model: "vertex_ai/mistral-large@2411" },
          },
          {
            model_name: "redacted-bedrock",
            model_info: { id: "redacted-bedrock", mode: "chat", litellm_provider: "bedrock" },
          },
          {
            model_name: "redacted-anthropic",
            model_info: { id: "redacted-anthropic", mode: "chat", litellm_provider: "anthropic" },
          },
          {
            model_name: "mixed-claude-nova-forward",
            model_info: { id: "claude", mode: "chat", litellm_provider: "anthropic" },
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
          },
          {
            model_name: "mixed-claude-nova-forward",
            model_info: { id: "nova", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/amazon.nova-pro-v1:0" },
          },
          {
            model_name: "mixed-claude-nova-reverse",
            model_info: { id: "nova", mode: "chat", litellm_provider: "bedrock" },
            litellm_params: { model: "bedrock/amazon.nova-pro-v1:0" },
          },
          {
            model_name: "mixed-claude-nova-reverse",
            model_info: { id: "claude", mode: "chat", litellm_provider: "anthropic" },
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map(({ id, api }) => [id, api])).toEqual([
      ["bedrock-claude", "anthropic-messages"],
      ["mixed-route", "openai-completions"],
      ["unknown-route", "openai-completions"],
      ["claude-responses", "openai-responses"],
      ["claude-public-only", "openai-completions"],
      ["contradicted-claude", "openai-completions"],
      ["anthropic-claude", "anthropic-messages"],
      ["bedrock-adapter-claude", "anthropic-messages"],
      ["bedrock-converse-claude", "anthropic-messages"],
      ["vertex-claude", "anthropic-messages"],
      ["custom-claude", "openai-completions"],
      ["known-nonclaude-prefix", "openai-completions"],
      // Claude 3.5 predates the bundled catalog, so its Messages requirements are
      // unknown and the group reduces to Chat rather than guessing a thinking shape.
      ["anthropic-unprefixed-claude", "openai-completions"],
      ["bedrock-unprefixed-claude", "openai-completions"],
      ["adapter-only-anthropic", "openai-completions"],
      ["adapter-absent-claude", "openai-completions"],
      ["anthropic-nonclaude", "openai-completions"],
      ["bedrock-nova-claude-base", "openai-completions"],
      ["azure-claude-base", "openai-completions"],
      ["missing-mode-claude", "openai-completions"],
      ["bedrock-nova", "openai-completions"],
      ["bedrock-converse-llama", "openai-completions"],
      ["bedrock-mistral", "openai-completions"],
      ["bedrock-titan", "openai-completions"],
      ["bedrock-cohere", "openai-completions"],
      ["vertex-llama", "openai-completions"],
      ["vertex-mistral", "openai-completions"],
      ["redacted-bedrock", "openai-completions"],
      ["redacted-anthropic", "openai-completions"],
      ["mixed-claude-nova-forward", "openai-completions"],
      ["mixed-claude-nova-reverse", "openai-completions"],
    ]);
    expect(result.models[0]?.compat).toEqual({ forceAdaptiveThinking: true, supportsStrictTools: true });
  });
});

describe("discoverModels response-mode models", () => {
  it("keeps /model/info response-mode models with a Responses API override", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5.3-codex-openai",
              model_info: {
                mode: "responses",
                max_input_tokens: 272000,
                max_output_tokens: 128000,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.3-codex-openai",
      api: "openai-responses",
      contextWindow: 272000,
      maxTokens: 128000,
    });
  });

  it("keeps /health response-mode model_info fallbacks on Chat", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return jsonResponse(404, {});
      if (url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [{ model: "openai/gpt-5.3-codex-openai", model_id: "uuid-1" }],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-1")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5.3-codex-openai",
              model_info: { mode: "response" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.3-codex-openai",
      api: "openai-completions",
      compat: { supportsStore: false },
    });
  });

  function mockHealthDeployment(backend: string, params: string[], mode: string): void {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(404, {});
      if (url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, { healthy_endpoints: [{ model: "private-reasoning-route", model_id: "uuid-1" }] });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-1")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "private-reasoning-route",
              litellm_params: { model: backend },
              model_info: {
                mode,
                litellm_provider: "moonshot",
                supports_reasoning: true,
                supported_openai_params: params,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
  }

  it.each([
    { name: "K2.7 Code", backend: "moonshot/kimi-k2.7-code", params: ["thinking"], mode: "chat" },
    { name: "K2.7 Code", backend: "moonshot/kimi-k2.7-code", params: ["thinking"], mode: "responses" },
    { name: "K3", backend: "moonshot/kimi-k3", params: ["reasoning_effort"], mode: "chat" },
    { name: "K3", backend: "moonshot/kimi-k3", params: ["reasoning_effort"], mode: "responses" },
  ])(
    "preserves $name replay and vendor compat when /health keeps a $mode-mode deployment on Chat",
    async ({ backend, params, mode }) => {
      mockHealthDeployment(backend, params, mode);

      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

      expect(result.source).toBe("health");
      expect(result.models[0]).toMatchObject({
        id: "private-reasoning-route",
        api: "openai-completions",
        compat: {
          requiresReasoningContentOnAssistantMessages: true,
          // Route text does not name Moonshot, so these prove the reduction
          // supplied the vendor family rather than an alias guess.
          maxTokensField: "max_tokens",
          supportsDeveloperRole: false,
        },
        litellmPolicy: { normalizeStrictToolMessages: true },
      });
    },
  );

  it("reduces a /health responses-mode deployment exactly like the same Chat deployment", async () => {
    mockHealthDeployment("moonshot/kimi-k3", ["reasoning_effort"], "chat");
    const chat = await discoverModels("https://litellm.example.com", "sk-test", {});
    vi.restoreAllMocks();
    mockHealthDeployment("moonshot/kimi-k3", ["reasoning_effort"], "responses");
    const responses = await discoverModels("https://litellm.example.com", "sk-test", {});

    // The downgrade rewrites the mode before reduction instead of patching the
    // mapped model, so no capability can be lost on the way through.
    expect(responses.models).toEqual(chat.models);
    expect(responses.models[0]?.compat).toMatchObject({ supportsReasoningEffort: true, thinkingFormat: "openai" });
  });

  it("still drops a /health deployment whose mode this provider cannot serve", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(404, {});
      if (url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: "embed-route", model_id: "uuid-e" },
            { model: "chat-route", model_id: "uuid-c" },
          ],
        });
      }
      if (url.includes("uuid-e")) {
        return jsonResponse(200, {
          data: [{ model_name: "embed-route", model_info: { mode: "embedding" } }],
        });
      }
      if (url.includes("uuid-c")) {
        return jsonResponse(200, {
          data: [{ model_name: "chat-route", model_info: { mode: "responses" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["chat-route"]);
  });

  it("does not derive thinking controls from a health-only route name", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(404, {});
      if (url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, { healthy_endpoints: [{ model: "openai/gpt-5.5", model_id: "uuid-1" }] });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-1")) {
        return jsonResponse(200, { data: [{ model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    // The route name is not deployment evidence, and an ABSENT map would let
    // pi-ai offer every standard level, so each level is denied explicitly.
    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    });
    expect(getSupportedThinkingLevels(nativeModel(result.models[0]))).toEqual([]);
  });
});

describe("discoverModels fallback to /v1/models", () => {
  it("keeps unqualified fallback ids bounded instead of scanning every provider catalog", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-4o" }, { id: "kimi-k2.6" }, { id: "grok-4.5" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "gpt-4o", name: "gpt-4o (no metadata)" }),
        expect.objectContaining({
          id: "kimi-k2.6",
          name: "kimi-k2.6 (no metadata)",
          litellmPolicy: { normalizeStrictToolMessages: false, normalizeThinkTags: true },
        }),
        expect.objectContaining({ id: "grok-4.5", name: "grok-4.5 (no metadata)" }),
      ]),
    );
  });

  for (const status of [401, 403, 404]) {
    it(`falls back when /model/info returns ${status}`, async () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = input instanceof URL ? input.toString() : String(input);
        if (url.endsWith("/model/info")) return new Response(null, { status });
        if (url.endsWith("/v1/models")) {
          return jsonResponse(200, {
            data: [{ id: "openai/gpt-4o" }, { id: "anthropic/claude-3-5-sonnet" }],
          });
        }
        throw new Error(`unexpected URL: ${url}`);
      });
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.source).toBe("models_list");
      expect(result.models.map((m) => m.id).sort()).toEqual(["anthropic/claude-3-5-sonnet", "openai/gpt-4o"]);
      const anthropic = result.models.find((m) => m.id === "anthropic/claude-3-5-sonnet")!;
      expect(anthropic.name).toBe("anthropic/claude-3-5-sonnet (no metadata)");
      expect(anthropic.compat).toEqual({ supportsStore: false, cacheControlFormat: "anthropic" });
    });
  }

  it("skips models.dev enrichment when disabled", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "gpt-5.5", object: "model", owned_by: "openai" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
    });

    expect(urls).not.toContain("https://models.dev/api.json");
    expect(result.models[0]).toMatchObject({
      id: "gpt-5.5",
      name: "GPT-5.5",
      contextWindow: 272000,
    });
  });

  it("persists models.dev metadata after an initial cache miss", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    vi.spyOn(Date, "now").mockReturnValue(1_000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") return jsonResponse(200, MODELS_DEV_CATALOG);
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]?.contextWindow).toBe(1_050_000);
    expect(JSON.parse(await readFile(cachePath, "utf8"))).toEqual({
      fetchedAt: 1_000,
      catalog: MODELS_DEV_CATALOG,
    });
  });

  it("normalizes models.dev network metadata before caching it", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    vi.spyOn(Date, "now").mockReturnValue(1_000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") {
        return new Response(
          '{"openai":{"models":{"gpt-5.5":{"name":"Remote GPT","reasoning":"yes","modalities":{"input":["text",7,"image"]},"limit":{"context":1050000,"input":"many","output":1e400},"cost":{"input":5,"output":"expensive","cache_read":0.5,"cache_write":1e400}}}}}',
          { status: 200, headers: { "content-type": "application/json" } },
        );
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]).toMatchObject({
      name: "Remote GPT",
      reasoning: true,
      input: ["text", "image"],
      contextWindow: 1_050_000,
      maxTokens: 128_000,
    });
    expect(result.models[0]?.cost).toEqual({ input: 5, output: 0, cacheRead: 0.5, cacheWrite: 0 });
    expect(JSON.parse(await readFile(cachePath, "utf8"))).toEqual({
      fetchedAt: 1_000,
      catalog: {
        openai: {
          models: {
            "gpt-5.5": {
              name: "Remote GPT",
              modalities: { input: ["text", "image"] },
              limit: { context: 1_050_000 },
              cost: { input: 5, cache_read: 0.5 },
            },
          },
        },
      },
    });
  });

  it("lets initial cache-miss callers abort independently", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    const first = new AbortController();
    const later = new AbortController();
    const third = new AbortController();
    let modelsDevRequests = 0;
    let resolveModelsDev!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") {
        modelsDevRequests++;
        return new Promise<Response>((resolve, reject) => {
          resolveModelsDev = resolve;
          const signal = (init as { signal?: AbortSignal } | undefined)?.signal;
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const discover = (signal: AbortSignal) =>
      discoverModels("https://litellm.example.com", "sk-test", { modelsDevCachePath: cachePath, signal });
    const firstResult = discover(first.signal);
    const laterResult = discover(later.signal);
    const thirdResult = discover(third.signal);

    await vi.waitFor(() => expect(modelsDevRequests).toBe(1));
    first.abort(new Error("first aborted"));
    later.abort(new Error("later aborted"));
    await expect(firstResult).rejects.toThrow("first aborted");
    await expect(laterResult).rejects.toThrow("later aborted");
    resolveModelsDev(jsonResponse(200, MODELS_DEV_CATALOG));

    await expect(thirdResult).resolves.toMatchObject({
      models: [{ id: "gpt-5.5", contextWindow: 1_050_000 }],
    });
    expect(modelsDevRequests).toBe(1);
  });

  it("lets initial cache-miss callers time out independently", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    let modelsDevRequests = 0;
    let resolveModelsDev!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") {
        modelsDevRequests++;
        return new Promise<Response>((resolve, reject) => {
          resolveModelsDev = resolve;
          const signal = (init as { signal?: AbortSignal } | undefined)?.signal;
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const short = discoverModels("https://short.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
      timeoutMs: 30,
    });
    const shortTimeout = expect(short).rejects.toBeDefined();
    const long = discoverModels("https://long.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
      timeoutMs: 1_000,
    });

    await vi.waitFor(() => expect(modelsDevRequests).toBe(1));
    await shortTimeout;
    resolveModelsDev(jsonResponse(200, MODELS_DEV_CATALOG));
    await expect(long).resolves.toMatchObject({
      models: [{ id: "gpt-5.5", contextWindow: 1_050_000 }],
    });
    expect(modelsDevRequests).toBe(1);
  });

  it("uses a fresh persistent models.dev cache without fetching", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    await writeFile(cachePath, JSON.stringify({ fetchedAt: 1_000, catalog: MODELS_DEV_CATALOG }), "utf8");
    vi.spyOn(Date, "now").mockReturnValue(2_000);
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
    });

    expect(urls).not.toContain("https://models.dev/api.json");
    expect(result.models[0]?.contextWindow).toBe(1_050_000);
  });

  it("ignores malformed nested metadata in a fresh models.dev cache", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1_000,
        catalog: {
          openai: {
            models: {
              "gpt-5.5": {
                name: "Cached GPT",
                reasoning: "yes",
                modalities: { input: 42 },
                limit: { context: "many", output: {} },
                cost: { input: "expensive" },
              },
            },
          },
        },
      }),
      "utf8",
    );
    vi.spyOn(Date, "now").mockReturnValue(2_000);
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
    });

    expect(urls).not.toContain("https://models.dev/api.json");
    expect(result.models[0]).toMatchObject({
      name: "Cached GPT",
      reasoning: true,
      input: ["text", "image"],
      contextWindow: 272_000,
      maxTokens: 128_000,
    });
    expect(result.models[0]?.cost.input).toBe(5);
  });

  it("ignores inherited model keys in a fresh models.dev cache", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    await writeFile(cachePath, JSON.stringify({ fetchedAt: 1_000, catalog: { openai: { models: {} } } }), "utf8");
    vi.spyOn(Date, "now").mockReturnValue(2_000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "constructor", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]?.name).toBe("constructor (no metadata)");
  });

  it("returns stale metadata while refreshing it in the background", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    const staleCatalog = {
      openai: { models: { "gpt-5.5": { name: "Stale GPT", limit: { context: 200_000 } } } },
    };
    await writeFile(cachePath, JSON.stringify({ fetchedAt: 1, catalog: staleCatalog }), "utf8");
    vi.spyOn(Date, "now").mockReturnValue(28 * 24 * 60 * 60 * 1000 + 2);
    const refresh = deferred<Response>();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") return refresh.promise;
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]).toMatchObject({ name: "Stale GPT", contextWindow: 200_000 });
    refresh.resolve(jsonResponse(200, MODELS_DEV_CATALOG));
    await vi.waitFor(async () => {
      const cache = JSON.parse(await readFile(cachePath, "utf8"));
      expect(cache.catalog).toEqual(MODELS_DEV_CATALOG);
    });
  });

  it("keeps stale metadata when background refresh fails", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    const stale = { fetchedAt: 1, catalog: MODELS_DEV_CATALOG };
    await writeFile(cachePath, JSON.stringify(stale), "utf8");
    vi.spyOn(Date, "now").mockReturnValue(28 * 24 * 60 * 60 * 1000 + 2);
    const modelsDevRequests: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") {
        modelsDevRequests.push(url);
        return new Response(null, { status: 503 });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]?.contextWindow).toBe(1_050_000);
    await vi.waitFor(() => expect(modelsDevRequests).toHaveLength(1));
    expect(JSON.parse(await readFile(cachePath, "utf8"))).toEqual(stale);
  });

  it("deduplicates concurrent stale cache refreshes", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    await writeFile(cachePath, JSON.stringify({ fetchedAt: 1, catalog: MODELS_DEV_CATALOG }), "utf8");
    vi.spyOn(Date, "now").mockReturnValue(28 * 24 * 60 * 60 * 1000 + 2);
    const refresh = deferred<Response>();
    let refreshes = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") {
        refreshes++;
        return refresh.promise;
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await Promise.all([
      discoverModels("https://one.example.com", "sk-test", { modelsDevCachePath: cachePath }),
      discoverModels("https://two.example.com", "sk-test", { modelsDevCachePath: cachePath }),
    ]);

    expect(refreshes).toBe(1);
    refresh.resolve(jsonResponse(200, MODELS_DEV_CATALOG));
    await vi.waitFor(async () => {
      expect(JSON.parse(await readFile(cachePath, "utf8")).fetchedAt).toBe(Date.now());
    });
  });

  it("replaces a malformed models.dev cache from the network", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    await writeFile(cachePath, JSON.stringify({ fetchedAt: "invalid", catalog: [] }), "utf8");
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") return jsonResponse(200, MODELS_DEV_CATALOG);
      throw new Error(`unexpected URL: ${url}`);
    });

    await discoverModels("https://litellm.example.com", "sk-test", { modelsDevCachePath: cachePath });

    expect(urls).toContain("https://models.dev/api.json");
    expect(JSON.parse(await readFile(cachePath, "utf8")).catalog).toEqual(MODELS_DEV_CATALOG);
  });

  it("replaces a future-dated models.dev cache from the network", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
    const cachePath = join(dir, "litellm-models-dev.json");
    await writeFile(cachePath, JSON.stringify({ fetchedAt: 2_000, catalog: MODELS_DEV_CATALOG }), "utf8");
    vi.spyOn(Date, "now").mockReturnValue(1_000);
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url === "https://models.dev/api.json") return jsonResponse(200, MODELS_DEV_CATALOG);
      throw new Error(`unexpected URL: ${url}`);
    });

    await discoverModels("https://litellm.example.com", "sk-test", { modelsDevCachePath: cachePath });

    expect(urls).toContain("https://models.dev/api.json");
    expect(JSON.parse(await readFile(cachePath, "utf8")).fetchedAt).toBe(1_000);
  });

  it("uses Pi catalog metadata when models.dev is unavailable", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "gpt-5.5", object: "model", owned_by: "openai" }],
        });
      }
      if (url === "https://models.dev/api.json") return new Response(null, { status: 503 });
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(urls).toContain("https://models.dev/api.json");
    expect(result.source).toBe("models_list");
    expect(result.models[0]).toMatchObject({
      id: "gpt-5.5",
      name: "GPT-5.5",
      reasoning: true,
      input: ["text", "image"],
      contextWindow: 272000,
      maxTokens: 128000,
      compat: { supportsStore: false },
    });
  });

  it("enriches a bare Fable 5 fallback model from the Pi catalog", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "fable-5", object: "model", owned_by: "openai" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result).toMatchObject({
      source: "models_list",
      models: [
        {
          id: "fable-5",
          name: "Claude Fable 5",
          reasoning: true,
          thinkingLevelMap: { xhigh: "xhigh", max: "max" },
        },
      ],
    });
  });

  it("enriches a bare Opus 5 fallback model from the Pi catalog", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "opus-5", object: "model", owned_by: "openai" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result).toMatchObject({
      source: "models_list",
      models: [
        {
          id: "opus-5",
          name: "Claude Opus 5",
          reasoning: true,
          thinkingLevelMap: { xhigh: "xhigh", max: "max" },
        },
      ],
    });
  });

  it("uses models.dev metadata when LiteLLM returns provider ownership", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "gpt-5.5", object: "model", owned_by: "openai" }],
        });
      }
      if (url === "https://models.dev/api.json") {
        return jsonResponse(200, {
          openai: {
            models: {
              "gpt-5.5": {
                id: "gpt-5.5",
                name: "GPT-5.5",
                reasoning: true,
                modalities: { input: ["text", "image", "pdf"], output: ["text"] },
                limit: { context: 1_050_000, input: 922_000, output: 128_000 },
                cost: { input: 5, output: 30, cache_read: 0.5 },
              },
            },
          },
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(urls).toContain("https://models.dev/api.json");
    expect(result.source).toBe("models_list");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "gpt-5.5",
      name: "GPT-5.5",
      reasoning: true,
      thinkingLevelMap: { off: "none", xhigh: "xhigh" },
      input: ["text", "image"],
      contextWindow: 1050000,
      maxTokens: 128000,
      compat: { supportsStore: false },
    });
    expect(result.models[0]?.cost).toEqual({ input: 5, output: 30, cacheRead: 0.5, cacheWrite: 0 });
  });

  it("throws when /model/info returns a non-401/403/404 error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 500 }));
    await expect(discoverModels("https://litellm.example.com", "sk-test", {})).rejects.toThrow(/500/);
  });
});

describe("discoverModels fallback to /health", () => {
  it("reports completed health detail requests rather than endpoint indexes", async () => {
    const endpoints = Array.from({ length: 11 }, (_, index) => ({
      model: `model-${index + 1}`,
      model_id: `uuid-${index + 1}`,
    }));
    const pending: Array<(response: Response) => void> = [];
    const progress = vi.fn();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return Promise.resolve(new Response(null, { status: 404 }));
      if (url.endsWith("/v1/models")) return Promise.resolve(new Response(null, { status: 404 }));
      if (url.endsWith("/health")) return Promise.resolve(jsonResponse(200, { healthy_endpoints: endpoints }));
      if (url.endsWith("uuid-11")) {
        return Promise.resolve(jsonResponse(200, { data: [{ model_name: "model-11", model_info: { mode: "chat" } }] }));
      }
      return new Promise<Response>((resolve) => pending.push(resolve));
    });

    const discovery = discoverModels("https://litellm.example.com", "sk-test", { onProgress: progress });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(14));
    const reportedBeforeCompletion = progress.mock.calls.some(([message]) => message === "Fetched 11/11 models...");
    for (const [index, resolve] of pending.entries()) {
      resolve(jsonResponse(200, { data: [{ model_name: `model-${index + 1}`, model_info: { mode: "chat" } }] }));
    }
    await discovery;

    expect(reportedBeforeCompletion).toBe(false);
    expect(progress).toHaveBeenCalledWith("Fetched 11/11 models...");
  });

  it("uses /health and per-endpoint /model/info when OpenAI model listing is unavailable", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 404 });
      if (url.endsWith("/v1/models")) return new Response(null, { status: 404 });
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: "vertex/claude-sonnet", model_id: "uuid-1" },
            { model: "openai/gpt-4o-mini", model_id: "uuid-2" },
          ],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-1")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "vertex/claude-sonnet",
              model_info: {
                mode: "chat",
                max_input_tokens: 200000,
                supports_vision: true,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
              },
            },
          ],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-2")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-4o-mini",
              model_info: {
                mode: "chat",
                max_input_tokens: 128000,
                max_output_tokens: 16384,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(urls).toEqual([
      "https://litellm.example.com/model/info",
      "https://litellm.example.com/v1/models",
      "https://litellm.example.com/health",
      "https://litellm.example.com/model/info?litellm_model_id=uuid-1",
      "https://litellm.example.com/model/info?litellm_model_id=uuid-2",
    ]);
    expect(result.source).toBe("health");
    expect(result.models.map((model) => model.id)).toEqual(["vertex/claude-sonnet", "openai/gpt-4o-mini"]);
    expect(result.models[0]).toMatchObject({
      input: ["text", "image"],
      contextWindow: 200000,
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
  });

  it("uses healthy endpoint model names when /health entries do not include model ids", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 404 });
      if (url.endsWith("/v1/models")) return new Response(null, { status: 404 });
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: "azure/gpt-35-turbo", api_base: "https://azure.example.com" },
            { model: "anthropic/claude-3-5-sonnet", api_base: "https://anthropic.example.com" },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(urls).toEqual([
      "https://litellm.example.com/model/info",
      "https://litellm.example.com/v1/models",
      "https://litellm.example.com/health",
    ]);
    expect(result.source).toBe("health");
    expect(result.models.map((model) => model.id)).toEqual(["azure/gpt-35-turbo", "anthropic/claude-3-5-sonnet"]);
    expect(result.models[1]).toMatchObject({
      name: "anthropic/claude-3-5-sonnet (incomplete metadata)",
      contextWindow: 128000,
      maxTokens: 16384,
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
  });
});

describe("discoverModels timeout", () => {
  it("aborts the fetch after timeoutMs", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      return new Promise((_resolve, reject) => {
        const signal = (init as { signal?: AbortSignal } | undefined)?.signal;
        signal?.addEventListener("abort", () => reject(signal.reason ?? new Error("aborted")));
      });
    });
    const start = Date.now();
    await expect(discoverModels("https://litellm.example.com", "sk-test", { timeoutMs: 30 })).rejects.toBeDefined();
    expect(Date.now() - start).toBeLessThan(500);
  });
});

// The wire-level sweep in tests/provider-compat/stream.test.ts covers the
// `/model/info` group path. Capability degrades across the discovery fallbacks
// because the evidence does, so this pins the same invariant on every path: a
// reasoning model must never carry an ABSENT thinkingLevelMap together with a
// compat that denies effort and names no format, because pi-ai reads absent as
// "every standard level supported".
describe("discovery paths never advertise untransmittable levels", () => {
  const BACKENDS = [
    "moonshot/kimi-k2.6",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k3",
    "moonshot/kimi-k2-thinking",
    "moonshot/kimi-latest",
    "deepseek/deepseek-v4",
    "deepseek/deepseek-r1",
    "openai/o3",
    "internal/opaque",
  ];
  const ALIASES = ["kimi-k2.6", "moonshotai/kimi-k3", "deepseek/deepseek-v4-pro", "openai/gpt-5.5", "mystery-route"];

  function assertTransmissible(model: DiscoveredModel | undefined, label: string): void {
    if (!model) throw new Error(`${label}: no model discovered`);
    const native = nativeModel(model);
    const compat = model.compat as { supportsReasoningEffort?: boolean; thinkingFormat?: string } | undefined;
    const carries = compat?.thinkingFormat !== undefined || compat?.supportsReasoningEffort !== false;
    const offered = getSupportedThinkingLevels(native).filter((level) => level !== "off");
    if (!carries) expect(offered, `${label} advertises ${JSON.stringify(offered)} it cannot transmit`).toEqual([]);
    if (model.reasoning && !carries) expect(model.thinkingLevelMap, `${label} needs an explicit denial`).toBeDefined();
  }

  it.each(
    BACKENDS.flatMap((backend) =>
      [undefined, ["thinking"], ["reasoning_effort"]].map((params) => ({ backend, params })),
    ),
  )("/model/info $backend with $params", async ({ backend, params }) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "route",
              litellm_params: { model: backend, ...(params ? { allowed_openai_params: params } : {}) },
              model_info: { id: "d1", mode: "chat", supports_reasoning: true },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    assertTransmissible(result.models[0], `/model/info ${backend} ${JSON.stringify(params)}`);
  });

  it.each(ALIASES)("/v1/models alias %s", async (alias) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(403, {});
      if (url.endsWith("/v1/models")) return jsonResponse(200, { data: [{ id: alias }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    assertTransmissible(result.models[0], `/v1/models ${alias}`);
  });

  it.each(ALIASES)("/health endpoint list %s", async (alias) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(403, {});
      if (url.endsWith("/v1/models")) return jsonResponse(403, {});
      if (url.endsWith("/health")) return jsonResponse(200, { healthy_endpoints: [{ model: alias }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    assertTransmissible(result.models[0], `/health list ${alias}`);
  });

  it.each(ALIASES)("/health per-deployment detail behind alias %s", async (alias) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(403, {});
      if (url.endsWith("/v1/models")) return jsonResponse(403, {});
      if (url.endsWith("/health")) return jsonResponse(200, { healthy_endpoints: [{ model: alias, model_id: "u1" }] });
      if (url.includes("litellm_model_id=u1")) {
        // No `model_name`: the alias is the only identifier, i.e. route text only.
        return jsonResponse(200, { data: [{ model_info: { mode: "chat", supports_reasoning: true } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    assertTransmissible(result.models[0], `/health detail ${alias}`);
  });
});

describe("family evidence authority", () => {
  it.each([
    { name: "OpenAI declared alongside an unknown backend", second: "azure_ai/mystery-deployment" },
    { name: "OpenAI declared alongside a deployment with no routing model", second: undefined },
  ])("withholds route-name compat when only some deployments declare a family: $name", async ({ second }) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "claude-router",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "a", mode: "chat", litellm_provider: "openai" },
            },
            {
              model_name: "claude-router",
              ...(second ? { litellm_params: { model: second } } : {}),
              model_info: { id: "b", mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    // Partial evidence is still evidence that the route is not uniformly what
    // its name suggests, so Anthropic cache-control must not be relayed to a
    // group half-served by OpenAI.
    expect(result.models[0]?.compat).toEqual({ supportsStore: false });
    expect(result.models[0]?.compat).not.toHaveProperty("cacheControlFormat");
  });

  it("still trusts a family every deployment declares", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: ["anthropic/claude-sonnet-4-6", "anthropic/claude-opus-4-6"].map((model, index) => ({
            model_name: "claude-router",
            litellm_params: { model },
            model_info: { id: `d${index}`, mode: "chat", litellm_provider: "anthropic" },
          })),
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      api: "anthropic-messages",
      compat: { forceAdaptiveThinking: true, supportsStrictTools: true },
    });
  });
});

describe("cache-read and Responses paths honour the transmissibility gate", () => {
  const cachedModel = (id: string, compat: DiscoveredModel["compat"]): Model<Api> =>
    ({
      id,
      name: `${id} (no metadata)`,
      api: "openai-completions",
      provider: "litellm",
      baseUrl: "https://litellm.example.com/v1",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
      compat,
    }) as Model<Api>;

  it("does not let catalog enrichment advertise levels the cached compat cannot carry", () => {
    // The cached compat stays as stored, so a catalog upgrade that starts
    // resolving this id must not hand it levels that reach the wire as nothing.
    const enriched = enrichCachedModel(
      cachedModel("moonshotai/kimi-k2-thinking", buildCompat("moonshotai/kimi-k2-thinking")),
    );

    expect(enriched.reasoning).toBe(true);
    expect(getSupportedThinkingLevels(enriched)).toEqual([]);
  });

  it("still enriches levels when the cached compat can carry them", () => {
    const enriched = enrichCachedModel(cachedModel("opus-5", buildCompat("opus-5")));

    expect(enriched.name).toBe("Claude Opus 5");
    expect(getSupportedThinkingLevels(enriched)).not.toEqual([]);
  });

  it.each(["chat", "responses"])("keeps vendor compat and fails closed for a %s-mode Kimi group", async (mode) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "route",
            litellm_params: { model: "moonshot/kimi-k2.6" },
            model_info: { id: "d1", mode, litellm_provider: "moonshot", supports_reasoning: true },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    // Responses mode keeps its shared vendor restriction and must also honor the
    // Chat-only effort denial as authority for offering no selectable level.
    expect(result.models[0]?.compat).toMatchObject({ supportsDeveloperRole: false });
    expect(getSupportedThinkingLevels(nativeModel(result.models[0]))).toEqual([]);
  });

  it.each([
    {
      name: "/v1/models",
      fetch: async (input: Parameters<typeof fetch>[0]) => {
        const url = String(input);
        if (url.endsWith("/model/info")) return jsonResponse(403, {});
        if (url.endsWith("/v1/models")) return jsonResponse(200, { data: [{ id: "moonshotai/kimi-k3" }] });
        throw new Error(`unexpected URL: ${url}`);
      },
    },
    {
      name: "/health",
      fetch: async (input: Parameters<typeof fetch>[0]) => {
        const url = String(input);
        if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return jsonResponse(403, {});
        if (url.endsWith("/health")) {
          return jsonResponse(200, { healthy_endpoints: [{ model: "moonshotai/kimi-k3" }] });
        }
        throw new Error(`unexpected URL: ${url}`);
      },
    },
  ])("keeps vendor-denied fallback catalog levels closed through $name", async ({ fetch }) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(fetch);

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.reasoning).toBe(true);
    expect(result.models[0]?.compat).toMatchObject({ supportsReasoningEffort: false });
    expect(getSupportedThinkingLevels(nativeModel(result.models[0]))).toEqual([]);
  });

  it("reaches the same display conclusion for one route on every discovery source", async () => {
    const id = "kimi-k2.6-thinking";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: id,
            litellm_params: { model: "moonshot/kimi-k2.6" },
            model_info: { id: "d1", mode: "chat", litellm_provider: "moonshot" },
          },
        ],
      }),
    );
    const grouped = await discoverModels("https://litellm.example.com", "sk-test", {});

    vi.restoreAllMocks();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(403, {});
      if (url.endsWith("/v1/models")) return jsonResponse(200, { data: [{ id }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const listed = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    // A forced-thinking route streams reasoning as its own field, so no path may
    // decide to unwrap `<think>` while another decides not to. Only the grouped
    // path has deployment evidence that can authorize the outbound repair.
    expect(grouped.models[0]?.litellmPolicy).toEqual(moonshotPolicy(id, true));
    expect(listed.models[0]?.litellmPolicy).toEqual(moonshotPolicy(id));
    expect(grouped.models[0]?.litellmPolicy?.normalizeThinkTags).toBe(false);
    expect(listed.models[0]?.litellmPolicy?.normalizeThinkTags).toBe(false);
  });

  it("reports withheld catalog authority once with bounded route ids", async () => {
    const writes: string[] = [];
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation((chunk) => {
      writes.push(String(chunk));
      return true;
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: ["r1", "r2", "r3", "r4"].flatMap((route) => [
          {
            model_name: route,
            litellm_params: { model: "openai/gpt-4o" },
            model_info: { id: `${route}-a`, mode: "chat", litellm_provider: "openai" },
          },
          {
            model_name: route,
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
            model_info: { id: `${route}-b`, mode: "chat", litellm_provider: "anthropic" },
          },
        ]),
      }),
    );

    await discoverModels("https://litellm.example.com", "sk-test", {});
    stderr.mockRestore();

    const reports = writes.filter((line) => line.includes("missing or conflicting deployment provider evidence"));
    expect(reports).toHaveLength(1);
    expect(reports[0]).toContain("4 route group(s)");
    expect(reports[0]).toContain("(+1 more)");
    // Only public route ids, never credentials or litellm_params.
    expect(reports[0]).not.toMatch(/sk-|api_key/);
  });

  it("bounds and deduplicates withheld strict-tool-repair diagnostics", async () => {
    const writes: string[] = [];
    vi.spyOn(process.stderr, "write").mockImplementation((chunk) => {
      writes.push(String(chunk));
      return true;
    });
    const rows = (routes: string[]) =>
      routes.flatMap((route) => [
        {
          model_name: route,
          litellm_params: { model: "moonshot/kimi-k2.6" },
          model_info: { id: `${route}-kimi`, mode: "chat", litellm_provider: "moonshot" },
        },
        {
          model_name: route,
          litellm_params: { model: "internal/unidentified" },
          model_info: { id: `${route}-unknown`, mode: "chat" },
        },
      ]);
    const discoverInfo = async (routes: string[]) => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: rows(routes) }));
      await discoverModels("https://litellm.example.com", "sk-test", {});
      vi.mocked(globalThis.fetch).mockRestore();
    };

    await discoverInfo(["repair-a", "repair-b", "repair-c", "repair-d"]);
    await discoverInfo(["repair-a", "repair-b", "repair-c", "repair-d"]);
    await discoverInfo(["repair-a", "repair-new"]);

    const reports = writes.filter((line) => line.includes("strict tool-message repair is withheld"));
    expect(reports).toHaveLength(2);
    expect(reports[0]).toContain("4 route group(s)");
    expect(reports[0]).toContain("repair-a, repair-b, repair-c (+1 more)");
    expect(reports[0]).not.toContain("repair-d");
    expect(reports[1]).toContain("1 route group(s)");
    expect(reports[1]).toContain("repair-new");
    expect(reports[1]).not.toContain("repair-a");
    expect(reports.join("\n")).not.toMatch(/sk-|api_key|repair-a-kimi/);
  });

  it.each([
    {
      source: "/v1/models",
      fetch: async (input: Parameters<typeof fetch>[0]) => {
        const url = String(input);
        if (url.endsWith("/model/info")) return jsonResponse(403, {});
        if (url.endsWith("/v1/models")) return jsonResponse(200, { data: [{ id: "kimi-diagnostic-list" }] });
        throw new Error(`unexpected URL: ${url}`);
      },
    },
    {
      source: "/health",
      fetch: async (input: Parameters<typeof fetch>[0]) => {
        const url = String(input);
        if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return jsonResponse(403, {});
        if (url.endsWith("/health")) {
          return jsonResponse(200, { healthy_endpoints: [{ model: "kimi-diagnostic-health" }] });
        }
        throw new Error(`unexpected URL: ${url}`);
      },
    },
  ])("reports the withheld route-name repair through $source", async ({ fetch, source }) => {
    const writes: string[] = [];
    vi.spyOn(process.stderr, "write").mockImplementation((chunk) => {
      writes.push(String(chunk));
      return true;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(fetch);

    await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    const reports = writes.filter((line) => line.includes("strict tool-message repair is withheld"));
    expect(reports).toHaveLength(1);
    expect(reports[0]).toContain(source === "/health" ? "kimi-diagnostic-health" : "kimi-diagnostic-list");
  });
});

describe("more evidence never produces a less safe request", () => {
  const kimiGroup = (backends: string[]) =>
    backends.map((model, index) => ({
      model_name: "kimi-k2.6",
      litellm_params: { model },
      model_info: { id: `d${index}`, mode: "chat", supports_reasoning: true },
    }));

  it.each([
    {
      name: "every deployment identifies Moonshot",
      backends: ["azure_ai/kimi-k2.6-east", "azure_ai/kimi-k2.6-west"],
      mixed: false,
      unanimousKimi: true,
    },
    {
      name: "only one deployment identifies Moonshot",
      backends: ["azure_ai/kimi-k2.6-east", "azure_ai/k26-prod"],
      mixed: true,
      unanimousKimi: false,
    },
    {
      name: "no deployment identifies anything",
      backends: ["azure_ai/k26-prod-east", "azure_ai/k26-prod-west"],
      mixed: false,
      unanimousKimi: false,
    },
  ])("meets vendor policy per field when $name", async ({ backends, mixed, unanimousKimi }) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: kimiGroup(backends) }));

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    const model = result.models[0];

    // Partial family evidence is withheld outright, so it gets no vendor compat
    // and no request policy. What it must never do is advertise a level: the
    // closed serializer policy is what keeps that from fail-opening.
    const offered = getSupportedThinkingLevels(nativeModel(model)).filter((level) => level !== "off");
    // The adjudicated target: a deployment-evidenced safety restriction must not
    // disappear because a sibling is unlabeled, and no level may be advertised
    // that the group cannot carry — in every mixture.
    expect(offered).toEqual([]);
    // Response-side repair always survives Moonshot evidence; the outbound
    // rewrite requires every deployment to evidence the need.
    expect(model?.litellmPolicy?.normalizeThinkTags).toBe(true);
    expect(model?.litellmPolicy?.normalizeStrictToolMessages).toBe(unanimousKimi);
    expect(model?.compat).toMatchObject({ supportsReasoningEffort: false, supportsStrictMode: false });
    if (mixed) {
      // Shape-changing fields have no safe common value once a candidate is
      // unlabeled: `max_tokens` is rejected by newer OpenAI models.
      expect(model?.compat).not.toHaveProperty("maxTokensField");
    } else {
      expect(model?.compat).toMatchObject({ maxTokensField: "max_tokens" });
    }
    // Never catalog limits, pricing, or provider identity from a mixed group.
    expect(model?.contextWindow).toBe(128_000);
    expect(model?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  });

  it("still withholds additive vendor features when families disagree", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "claude-router",
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
            model_info: { id: "a", mode: "chat", litellm_provider: "anthropic" },
          },
          {
            model_name: "claude-router",
            litellm_params: { model: "openai/gpt-4o" },
            model_info: { id: "b", mode: "chat", litellm_provider: "openai" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    // cache_control on an OpenAI-served deployment breaks the request, so an
    // additive feature still needs unanimity.
    expect(result.models[0]?.compat).not.toHaveProperty("cacheControlFormat");
  });

  it("gates a level map stored by a release that predates the gate", () => {
    // Exactly what base published for a `/model/info` Kimi route: a catalog level
    // map beside a compat that denies effort and names no format.
    const stored = {
      id: "kimi-k3",
      name: "kimi-k3",
      api: "openai-completions",
      provider: "litellm",
      baseUrl: "https://litellm.example.com/v1",
      reasoning: true,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
      thinkingLevelMap: { off: null, minimal: null, low: "low", medium: null, high: "high", xhigh: null, max: "max" },
      compat: buildCompat("kimi-k3"),
    } as Model<Api>;

    expect(getSupportedThinkingLevels(enrichCachedModel(stored))).toEqual([]);
  });

  it("publishes the same levels for one route id on both fallback paths", async () => {
    const id = "anthropic/claude-sonnet-4-6";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(403, {});
      if (url.endsWith("/v1/models")) return jsonResponse(200, { data: [{ id }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const listed = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    vi.restoreAllMocks();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(403, {});
      if (url.endsWith("/v1/models")) return jsonResponse(403, {});
      if (url.endsWith("/health")) return jsonResponse(200, { healthy_endpoints: [{ model: id }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const health = await discoverModels("https://litellm.example.com", "sk-test", {});

    // Both fallbacks have the same evidence quality — a route name — so neither
    // may offer a level the other denies.
    expect(getSupportedThinkingLevels(nativeModel(health.models[0]))).toEqual(
      getSupportedThinkingLevels(nativeModel(listed.models[0])),
    );
    expect(getSupportedThinkingLevels(nativeModel(health.models[0]))).not.toEqual([]);
  });
});

describe("display normalization follows persisted policy, not route text", () => {
  it.each([
    {
      name: "an opaque route over a genuine Moonshot backend",
      route: "internal/prod-chat-7",
      backend: "moonshot/kimi-k2.6",
      expected: { normalizeStrictToolMessages: true, normalizeThinkTags: true },
    },
    {
      name: "a Kimi-looking route over a foreign backend",
      route: "kimi-k2.6",
      backend: "openai/gpt-4o",
      expected: undefined,
    },
    {
      name: "an opaque route over a forced-thinking Moonshot backend",
      route: "internal/prod-chat-8",
      backend: "moonshot/kimi-k2-thinking",
      expected: { normalizeStrictToolMessages: true, normalizeThinkTags: true },
    },
  ])("$name", async ({ route, backend, expected }) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: route,
            litellm_params: { model: backend },
            model_info: { id: "d1", mode: "chat", litellm_provider: backend.split("/")[0] },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    // The conclusion is persisted from deployment evidence, so a genuine backend
    // behind an opaque name is handled and a lookalike name over a foreign
    // backend is not. `message_end` reads only this field.
    expect(result.models[0]?.litellmPolicy).toEqual(expected);
  });
});
