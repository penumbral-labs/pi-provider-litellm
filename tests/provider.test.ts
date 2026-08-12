import type {
  Api,
  Credential,
  Model,
  ProviderAuth,
  ProviderModelsStore,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, type MockInstance, vi } from "vitest";
import { discoverModels } from "../src/discover.js";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveryResult } from "../src/types.js";

const apiSpies = vi.hoisted(() => ({ completions: vi.fn(), responses: vi.fn() }));
vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@earendil-works/pi-ai/compat")>()),
  openAICompletionsApi: () => ({ stream: apiSpies.completions, streamSimple: apiSpies.completions }),
  openAIResponsesApi: () => ({ stream: apiSpies.responses, streamSimple: apiSpies.responses }),
}));

const credential: Credential = { type: "api_key", key: "secret" };
const auth: ProviderAuth = {
  apiKey: { name: "API key", resolve: async () => ({ auth: { apiKey: "secret" } }) },
};

const discovered = (id: string): DiscoveryResult => ({
  source: "model_info",
  models: [
    {
      id,
      name: id,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    },
  ],
});

function native(id: string): Model<"openai-completions" | "openai-responses"> {
  return toNativeModels("litellm", "https://proxy.example/v1", discovered(id).models)[0];
}

function store(initial?: readonly Model<Api>[]) {
  let entry = initial ? { models: initial, checkedAt: 1 } : undefined;
  const value: ProviderModelsStore = {
    read: vi.fn(async () => entry),
    write: vi.fn(async (next) => {
      entry = next;
    }),
    delete: vi.fn(async () => {
      entry = undefined;
    }),
  };
  return value;
}

class PrototypeStore implements ProviderModelsStore {
  entry: Awaited<ReturnType<ProviderModelsStore["read"]>>;

  async read() {
    return this.entry;
  }

  async write(entry: Parameters<ProviderModelsStore["write"]>[0]) {
    this.entry = entry;
  }

  async delete() {
    this.entry = undefined;
  }
}

function context(modelsStore: ProviderModelsStore, allowNetwork: boolean): RefreshModelsContext {
  return { store: modelsStore, allowNetwork, credential };
}

function controller(overrides: Partial<Parameters<typeof createLiteLLMProvider>[0]> = {}) {
  return createLiteLLMProvider({
    id: "litellm",
    name: "LiteLLM",
    baseUrl: "https://proxy.example/v1",
    auth,
    discover: vi.fn(async () => discovered("fresh")),
    ...overrides,
  });
}

describe("toNativeModels", () => {
  it("converts discovery models into complete native models", () => {
    expect(toNativeModels("litellm", "https://proxy.example/v1", discovered("model-a").models)).toEqual([
      expect.objectContaining({
        id: "model-a",
        provider: "litellm",
        api: "openai-completions",
        baseUrl: "https://proxy.example/v1",
      }),
    ]);
  });

  it("preserves a discovered Responses API and defaults missing APIs to Completions", () => {
    const [responses, completions] = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("responses").models[0], api: "openai-responses" },
      discovered("completions").models[0],
    ]);

    expect(responses.api).toBe("openai-responses");
    expect(completions.api).toBe("openai-completions");
  });
});

describe("createLiteLLMProvider", () => {
  it("delegates native store restoration directly to createProvider", async () => {
    const modelsStore = store([native("stored")]);
    const value = controller();

    await value.refreshModels?.(context(modelsStore, false));

    expect(modelsStore.read).toHaveBeenCalledOnce();
  });

  it("restores stored models offline without discovery", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.refreshModels?.(context(store([native("stored")]), false));

    expect(value.getModels()).toEqual([native("stored")]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("re-enriches stale cached catalog aliases offline without discovery", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.refreshModels?.(
      context(
        store([
          {
            ...native("opus-5"),
            name: "opus-5 (no metadata)",
            reasoning: false,
            maxTokens: 16_384,
          },
        ]),
        false,
      ),
    );

    expect(value.getModels()).toEqual([
      expect.objectContaining({
        id: "opus-5",
        name: "Claude Opus 5",
        reasoning: true,
        thinkingLevelMap: { xhigh: "xhigh", max: "max" },
        provider: "litellm",
        api: "openai-completions",
        baseUrl: "https://proxy.example/v1",
      }),
    ]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("preserves conservative qualified-route metadata offline", async () => {
    const cached: Model<Api> = {
      ...native("openai/gpt-5.5"),
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
    };
    const value = controller();

    await value.refreshModels?.(context(store([cached]), false));

    expect(value.getModels()).toEqual([cached]);
  });

  it("keeps partially enriched stale cached aliases unchanged offline", async () => {
    const legacyFallback: Model<Api> = {
      ...native("opus-5"),
      name: "opus-5 (no metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
    };
    const partialCached: Model<Api>[] = [
      { ...legacyFallback, reasoning: true },
      { ...legacyFallback, input: ["text", "image"] },
      { ...legacyFallback, cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0 } },
      { ...legacyFallback, contextWindow: 128_001 },
      { ...legacyFallback, maxTokens: 16_385 },
    ];
    for (const cached of partialCached) {
      const value = controller();

      await value.refreshModels?.(context(store([cached]), false));

      expect(value.getModels()).toEqual([cached]);
    }
  });

  it("keeps unknown stale cached models unchanged offline", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const cached = { ...native("unknown-model"), name: "unknown-model (no metadata)" };
    const value = controller({ discover });

    await value.refreshModels?.(context(store([cached]), false));

    expect(value.getModels()).toEqual([cached]);
  });

  it("publishes and persists successful discovery", async () => {
    const modelsStore = store([native("old")]);
    const value = controller({ discover: vi.fn(async () => discovered("fresh")) });

    await value.refreshModels?.(context(modelsStore, true));

    expect(value.getModels()).toEqual([native("fresh")]);
    expect(modelsStore.write).toHaveBeenCalledOnce();
    expect(modelsStore.write).toHaveBeenCalledWith(expect.objectContaining({ models: [native("fresh")] }));
  });

  it("persists online discovery through a prototype-backed store", async () => {
    const modelsStore = new PrototypeStore();
    const value = controller({ discover: vi.fn(async () => discovered("fresh")) });

    await value.refreshModels?.(context(modelsStore, true));

    expect(value.getModels()).toEqual([native("fresh")]);
    expect(modelsStore.entry?.models).toEqual([native("fresh")]);
  });

  it("publishes discovered models with the credential URL", async () => {
    const value = controller({
      discover: vi.fn(async () => ({
        ...discovered("fresh"),
        baseUrl: "https://credential.example/v1",
      })),
    });

    await value.refreshModels?.(context(store(), true));

    expect(value.getModels()[0]?.baseUrl).toBe("https://credential.example/v1");
  });

  it("retains previous models when discovery rejects", async () => {
    const modelsStore = store([native("old")]);
    const discover = vi.fn(async () => {
      throw new Error("rejected");
    });
    const value = controller({ discover });

    await expect(value.refreshModels?.(context(modelsStore, true))).rejects.toThrow("rejected");

    expect(value.getModels()).toEqual([native("old")]);
    expect(modelsStore.write).not.toHaveBeenCalled();
  });

  it("retains previous models when discovery is aborted", async () => {
    const modelsStore = store([native("old")]);
    const abort = new AbortController();
    const discover = vi.fn(async () => {
      abort.abort();
      return discovered("fresh");
    });
    const value = controller({ discover });

    await value.refreshModels?.({ ...context(modelsStore, true), signal: abort.signal });

    expect(value.getModels()).toEqual([native("old")]);
    expect(modelsStore.write).not.toHaveBeenCalled();
  });

  it("shares one discovery across concurrent refreshes", async () => {
    let release!: (result: DiscoveryResult) => void;
    const pending = new Promise<DiscoveryResult>((resolve) => {
      release = resolve;
    });
    const discover = vi.fn(() => pending);
    const modelsStore = store([native("old")]);
    const value = controller({ discover });

    const first = value.refreshModels?.(context(modelsStore, true));
    const second = value.refreshModels?.(context(modelsStore, true));
    release(discovered("fresh"));
    await Promise.all([first, second]);

    expect(discover).toHaveBeenCalledOnce();
  });

  it("routes Responses models through the Responses API", async () => {
    apiSpies.responses.mockReturnValueOnce({});
    const responseModel = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("responses").models[0], api: "openai-responses" },
    ])[0];
    const value = controller();

    value.stream(responseModel, { messages: [] });

    expect(apiSpies.responses).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });
});

describe("discovery, store, and offline read parity", () => {
  afterEach(() => {
    fetchSpy?.mockRestore();
    fetchSpy = undefined;
  });

  let fetchSpy: MockInstance<typeof fetch> | undefined;

  function mockModelInfo(data: unknown): void {
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return new Response(JSON.stringify({ data }), {
          status: 200,
          headers: { "content-type": "application/json" },
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
  }

  function liveDiscovery() {
    return controller({
      discover: async () => discoverModels("https://proxy.example/v1", "sk-test", { modelsDev: false }),
    });
  }

  // A reduced `/model/info` group whose catalog authority was deliberately
  // withheld must not be re-enriched by the cache path. Rehydration has to
  // reproduce the discovered model exactly, or offline requests would be built
  // from metadata the proxy never provided.
  it.each([
    [
      "a singleton whose backend evidence resolves no catalog identity",
      [
        {
          model_name: "openai/gpt-5.5",
          model_info: { id: "only", mode: "chat" },
          litellm_params: { model: "openai/gpt-5.5-internal-preview" },
        },
      ],
    ],
    [
      "a group whose deployments disagree on backend",
      [
        { model_name: "openai/gpt-5.5", model_info: { id: "a", mode: "chat" } },
        {
          model_name: "openai/gpt-5.5",
          model_info: { id: "b", mode: "chat" },
          litellm_params: { model: "internal/mystery" },
        },
      ],
    ],
    [
      "conflicting variants of one deployment id",
      [
        { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat" } },
        { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat", max_input_tokens: 64_000 } },
      ],
    ],
  ])("rehydrates %s unchanged", async (_case, data) => {
    mockModelInfo(data);
    const modelsStore = store();

    const online = liveDiscovery();
    await online.refreshModels?.(context(modelsStore, true));
    const discoveredModels = online.getModels();

    expect(discoveredModels).toHaveLength(1);
    expect(discoveredModels[0]?.name).toBe("openai/gpt-5.5 (incomplete metadata)");
    expect(modelsStore.write).toHaveBeenCalledOnce();

    const offline = liveDiscovery();
    await offline.refreshModels?.(context(modelsStore, false));

    expect(offline.getModels()).toEqual(discoveredModels);
  });

  // Conflicting variants of one deployment id are the case that makes the marker
  // fix necessary: counting distinct ids would call this group a singleton, which
  // re-admits the public route name as catalog evidence and lets the model escape
  // marking entirely. `openai/gpt-5.5` resolves in the Pi catalog (272k context,
  // reasoning, tiered pricing), so every catalog value below proves it was refused.
  it("refuses route-name authority for conflicting variants of one deployment id", async () => {
    mockModelInfo([
      { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat" } },
      { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat", max_input_tokens: 64_000 } },
    ]);
    const modelsStore = store();

    const online = liveDiscovery();
    await online.refreshModels?.(context(modelsStore, true));
    const discoveredModels = online.getModels();

    expect(discoveredModels).toHaveLength(1);
    const model = discoveredModels[0];
    // 1. No singleton route-name fallback, and 2. conservative metadata.
    expect(model).toMatchObject({
      id: "openai/gpt-5.5",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 64_000,
      maxTokens: 16_384,
    });
    expect(model).not.toHaveProperty("thinkingLevelMap");
    expect(model?.cost.tiers).toBeUndefined();
    // 3. Marked, using the reduced-group marker rather than the fallback sentinel.
    expect(model?.name).toBe("openai/gpt-5.5 (incomplete metadata)");

    // 4. Discovery, store, and offline read parity.
    const offline = liveDiscovery();
    await offline.refreshModels?.(context(modelsStore, false));

    expect(offline.getModels()).toEqual(discoveredModels);
  });

  it("treats exact repeats of one deployment as a single enriched deployment", async () => {
    // The conflicting-variant rule must not cost us exact-repeat idempotency: one
    // row and the same row twice are one deployment, and a lone deployment may
    // still use its route name as a catalog hint.
    const deployment = { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat" } };
    const discoverOnce = async (data: unknown[]) => {
      mockModelInfo(data);
      const value = liveDiscovery();
      await value.refreshModels?.(context(store(), true));
      return value.getModels();
    };

    const single = await discoverOnce([deployment]);
    const repeated = await discoverOnce([deployment, deployment]);

    expect(repeated).toEqual(single);
    expect(single[0]?.name).toBe("openai/gpt-5.5");
    expect(single[0]).toMatchObject({ reasoning: true, contextWindow: 272_000 });
  });

  it("rehydrates an evidence-free /health route unchanged", async () => {
    // `/health` route text is not authorized for later re-enrichment, so its marker
    // must be the permanent one and a store round trip must be a no-op.
    fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      const json = (body: unknown) =>
        new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
      if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return new Response(null, { status: 404 });
      if (url.endsWith("/health")) return json({ healthy_endpoints: [{ model: "totally-unknown-route" }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const modelsStore = store();

    const online = liveDiscovery();
    await online.refreshModels?.(context(modelsStore, true));
    const discoveredModels = online.getModels();

    expect(discoveredModels).toHaveLength(1);
    expect(discoveredModels[0]?.name).toBe("totally-unknown-route (incomplete metadata)");
    expect(discoveredModels[0]?.name).not.toContain(" (no metadata)");

    const offline = liveDiscovery();
    await offline.refreshModels?.(context(modelsStore, false));

    expect(offline.getModels()).toEqual(discoveredModels);
  });

  it("still enriches an evidence-free /v1/models fallback model from the cache", async () => {
    // The distinction has to cut both ways: the fallback sentinel keeps working.
    const value = controller();

    await value.refreshModels?.(
      context(
        store([{ ...native("opus-5"), name: "opus-5 (no metadata)", reasoning: false, maxTokens: 16_384 }]),
        false,
      ),
    );

    expect(value.getModels()[0]).toMatchObject({ id: "opus-5", reasoning: true });
    expect(value.getModels()[0]?.name).not.toContain("no metadata");
  });
});
