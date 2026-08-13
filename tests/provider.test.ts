import {
  type Api,
  type Credential,
  createModels,
  createProvider,
  type Model,
  type ProviderAuth,
  type ProviderModelsStore,
  type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, type MockInstance, vi } from "vitest";
import { discoverModels } from "../src/discover.js";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveryResult, LiteLLMApi } from "../src/types.js";

const apiSpies = vi.hoisted(() => ({ anthropic: vi.fn(), completions: vi.fn(), responses: vi.fn() }));
vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@earendil-works/pi-ai/compat")>()),
  anthropicMessagesApi: () => ({ stream: apiSpies.anthropic, streamSimple: apiSpies.anthropic }),
  openAICompletionsApi: () => ({ stream: apiSpies.completions, streamSimple: apiSpies.completions }),
  openAIResponsesApi: () => ({ stream: apiSpies.responses, streamSimple: apiSpies.responses }),
}));

const credential: Credential = { type: "api_key", key: "secret" };

function credentialStore(entries: Record<string, Credential>) {
  return {
    read: vi.fn(async (provider: string) => entries[provider]),
    list: vi.fn(async () => []),
    modify: vi.fn(
      async (provider: string, update: (current: Credential | undefined) => Promise<Credential | undefined>) => {
        const next = await update(entries[provider]);
        if (next) entries[provider] = next;
        return entries[provider];
      },
    ),
    delete: vi.fn(async (provider: string) => {
      delete entries[provider];
    }),
  };
}
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
      api: "openai-completions",
    },
  ],
});

function native(id: string): Model<LiteLLMApi> {
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
    resolveCredentialRoot: ({ requestBaseUrl }) => requestBaseUrl ?? "https://proxy.example",
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
  let stderr: ReturnType<typeof vi.fn>;
  beforeEach(() => {
    vi.clearAllMocks();
    stderr = vi.fn().mockReturnValue(true);
    vi.spyOn(process.stderr, "write").mockImplementation(stderr as never);
  });
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

  it.each([
    {
      name: "Moonshot compat evidence under a neutral route name",
      id: "internal-reasoning-route",
      compat: {
        supportsStore: false,
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
        supportsStrictMode: false,
        maxTokensField: "max_tokens" as const,
      },
      policy: { normalizeStrictToolMessages: false, normalizeThinkTags: true },
    },
    {
      name: "a Moonshot-looking route name with generic compat evidence",
      id: "kimi-k2.6",
      compat: { supportsStore: false },
      policy: undefined,
    },
    {
      name: "an unknown route with no usable evidence",
      id: "mystery-route",
      compat: { supportsStore: false },
      policy: undefined,
    },
  ])("derives the cached request policy from $name", async ({ id, compat, policy }) => {
    // Cache entries predating request policies carry none, so only the
    // response-side conclusion is re-derived from stored compatibility evidence.
    // The fingerprint cannot prove unanimous deployment-family authority.
    const cached: Model<Api> = { ...native(id), compat };
    const value = controller();

    await value.refreshModels?.(context(store([cached]), false));

    const restored = value.getModels()[0] as Model<Api> & { litellmPolicy?: unknown };
    expect(restored.litellmPolicy).toEqual(policy);
  });

  it("never overwrites a request policy already stored on a cached model", async () => {
    const cached = {
      ...native("kimi-k2.6"),
      compat: { supportsStore: false },
      litellmPolicy: { normalizeStrictToolMessages: true, normalizeThinkTags: true },
    } as Model<Api>;
    const value = controller();

    await value.refreshModels?.(context(store([cached]), false));

    expect(value.getModels()).toEqual([cached]);
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

  it("reprojects stored models to the matching active credential host", () => {
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example/v1" });
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://proxy.example", [
      baseModel,
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
    ]);

    expect(value.filterModels?.(models, credential).map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://proxy.example/v1" },
      { api: "anthropic-messages", baseUrl: "https://proxy.example" },
    ]);
  });

  it("filters stale LiteLLM models without taking down other providers", async () => {
    const models = createModels({
      credentials: credentialStore({
        litellm: { type: "api_key", key: "secret", env: { LITELLM_BASE_URL: "https://active.example" } },
        other: { type: "api_key", key: "other" },
      }),
    });
    const value = controller({ resolveCredentialRoot: () => "https://active.example" });
    await value.refreshModels?.(context(store([native("stored")]), false));
    models.setProvider(value);
    models.setProvider(
      createProvider({
        id: "other",
        auth: { apiKey: { name: "Other", resolve: async () => ({ auth: { apiKey: "other" } }) } },
        models: [{ ...native("other"), provider: "other" }],
        api: { stream: apiSpies.completions, streamSimple: apiSpies.completions },
      }),
    );

    await expect(models.getAvailable()).resolves.toEqual([expect.objectContaining({ provider: "other", id: "other" })]);
  });

  it("rejects placeholder hosts as cached request targets", () => {
    const storedPlaceholder = controller({ resolveCredentialRoot: () => "https://active.example" });
    const placeholder = toNativeModels("litellm", "https://litellm.example.com", discovered("stored").models);

    expect(storedPlaceholder.filterModels?.(placeholder, credential)).toEqual([]);

    const activePlaceholder = controller({ resolveCredentialRoot: () => "https://litellm.example.com" });
    expect(activePlaceholder.filterModels?.([native("stored")], credential)).toEqual([]);
  });

  // The documentation host is a real, third-party-operated domain, so any spelling of it
  // that slipped through would send a live credential off the machine.
  const placeholderRoots = [
    "https://litellm.example.com",
    "https://litellm.example.com.",
    "https://litellm.example.com:8000",
    "https://LITELLM.EXAMPLE.COM",
    "http://litellm.example.com/v1",
  ] as const;
  const transports = [
    ["openai-completions", "completions"],
    ["openai-responses", "responses"],
    ["anthropic-messages", "anthropic"],
  ] as const;

  // One case per transport, so each spy assertion is independently load-bearing: a
  // protocol-specific bypass of the placeholder gate cannot hide behind a sibling.
  it.each(transports.flatMap(([api, spy]) => placeholderRoots.map((root) => [api, spy, root] as const)))(
    "never dispatches a %s request to placeholder host spelling %s %s",
    (api, spy, root) => {
      const value = controller({ resolveCredentialRoot: () => root });
      const model = { ...native("stored"), api, compat: {} } as Model<LiteLLMApi>;

      expect(value.filterModels?.([model], credential)).toEqual([]);
      expect(() => value.stream(model, { messages: [] })).toThrow(/placeholder LiteLLM model host/i);
      expect(apiSpies[spy]).not.toHaveBeenCalled();
    },
  );

  it("treats a trailing-dot host as the same host as its cached models", () => {
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example." });

    // A fully-qualified spelling of the active host must not read as a different host
    // and silently empty the model list. The dot survives into the request URL, which
    // resolves to the same host.
    expect(value.filterModels?.([native("stored")], credential)).toEqual([
      { ...native("stored"), baseUrl: "https://proxy.example./v1" },
    ]);
  });

  it("drops a model with an unsupported transport instead of failing the whole list", () => {
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example" });
    const usable = native("usable");
    // pi composes models.json overrides into the provider's list, where `api` is only
    // validated as a non-empty string, so an unknown transport can reach filterModels.
    const foreign = { ...native("foreign"), api: "google-generative-ai" } as unknown as Model<Api>;

    const available = value.filterModels?.([foreign, usable] as Model<LiteLLMApi>[], credential);

    expect(available).toEqual([{ ...usable, baseUrl: "https://proxy.example/v1" }]);
    expect(stderr).toHaveBeenCalledWith(
      "LiteLLM (litellm): Cached model uses unsupported LiteLLM transport google-generative-ai; a network refresh with a valid LiteLLM base URL is required\n",
    );
  });

  it("reports each availability reason once", () => {
    const value = controller({ resolveCredentialRoot: () => "https://active.example" });
    const stale = toNativeModels("litellm", "https://stale.example", discovered("stale").models);

    for (let attempt = 0; attempt < 3; attempt++) expect(value.filterModels?.(stale, credential)).toEqual([]);

    const staleReports = stderr.mock.calls.filter(([line]) => String(line).includes("stale LiteLLM model host"));
    expect(staleReports).toHaveLength(1);
    expect(String(staleReports[0]?.[0])).toBe(
      "LiteLLM (litellm): Cached model has stale LiteLLM model host stale.example; active credentials use active.example; a network refresh with a valid LiteLLM base URL is required\n",
    );
  });

  it("reports and rejects everything when no credential root is available", () => {
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(value.filterModels?.([native("stored")], credential)).toEqual([]);
    expect(() => value.stream(native("stored"), { messages: [] })).toThrow(
      /do not identify a LiteLLM model host.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("rejects non-http credential roots", () => {
    const value = controller({ resolveCredentialRoot: () => "file:///etc/passwd" });

    expect(value.filterModels?.([native("stored")], credential)).toEqual([]);
    expect(() => value.stream(native("stored"), { messages: [] })).toThrow(/invalid LiteLLM model URL/i);
    expect(apiSpies.completions).not.toHaveBeenCalled();
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

  it("routes Chat Completions models through the Completions API", () => {
    apiSpies.completions.mockReturnValueOnce({});
    const value = controller();

    value.stream(native("chat"), { messages: [] });

    expect(apiSpies.completions).toHaveBeenCalledOnce();
    expect(apiSpies.responses).not.toHaveBeenCalled();
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
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
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
  });

  it("blocks stale hosts on stream and streamSimple before protocol dispatch", () => {
    const value = controller({ resolveCredentialRoot: () => "https://active.example" });

    expect(() => value.stream(native("resumed"), { messages: [] })).toThrow(
      /stale LiteLLM model host.*network refresh/i,
    );
    expect(() => value.streamSimple(native("default"), { messages: [] })).toThrow(
      /stale LiteLLM model host.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks placeholder and malformed request hosts before protocol dispatch", () => {
    const placeholder = controller({ resolveCredentialRoot: () => "https://litellm.example.com" });
    const malformed = controller({ resolveCredentialRoot: () => "undefined" });

    expect(() => placeholder.stream(native("placeholder"), { messages: [] })).toThrow(
      /placeholder LiteLLM model host.*network refresh/i,
    );
    expect(() => malformed.streamSimple(native("malformed"), { messages: [] })).toThrow(
      /invalid LiteLLM model URL.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("keeps valid models from a mixed cache and filters malformed URLs", () => {
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example" });
    const stale = { ...native("stale"), baseUrl: "https://stale.example/v1" };
    const malformed = { ...native("malformed"), baseUrl: "undefined" };

    expect(value.filterModels?.([native("valid"), stale, malformed], credential)).toEqual([native("valid")]);
  });

  it("routes Messages models through the Anthropic API", () => {
    apiSpies.anthropic.mockReturnValueOnce({});
    const messagesModel = toNativeModels("litellm", "https://proxy.example/v1/", [
      { ...discovered("messages").models[0], api: "anthropic-messages", compat: {} },
    ])[0];
    const value = controller();

    value.stream(messagesModel, { messages: [] });

    expect(messagesModel.baseUrl).toBe("https://proxy.example");
    expect(apiSpies.anthropic).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
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
