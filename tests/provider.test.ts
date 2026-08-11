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
import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import { createLiteLLMProtocolApis } from "../src/protocols.js";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi, ModelProtocol } from "../src/types.js";

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

// Compile-time pairing assertions. Each @ts-expect-error IS the test: typecheck
// fails if the annotated line stops being an error, i.e. if the union stops
// rejecting a cross-protocol compat pairing.
//
// String values need `as const`. Without it the literal widens and TypeScript
// reports a widening error (TS2322) instead of the pairing error (TS2353), which
// would satisfy @ts-expect-error for the wrong reason.
{
  const messagesWithCompletionsField: DiscoveredModel = {
    ...discovered("messages").models[0],
    api: "anthropic-messages",
    compat: {
      // @ts-expect-error supportsStore is an OpenAI-completions field.
      supportsStore: false,
    },
  };
  const responsesWithCompletionsField: DiscoveredModel = {
    ...discovered("responses").models[0],
    api: "openai-responses",
    compat: {
      // @ts-expect-error maxTokensField is an OpenAI-completions field.
      maxTokensField: "max_tokens" as const,
    },
  };
  const responsesWithCacheControl: DiscoveredModel = {
    ...discovered("responses").models[0],
    api: "openai-responses",
    compat: {
      // @ts-expect-error cacheControlFormat is an OpenAI-completions field.
      cacheControlFormat: "anthropic" as const,
    },
  };
  // A ModelProtocol cannot be assembled from mismatched halves either, which is
  // what makes the discovery builders structurally safe rather than merely careful.
  const mismatchedProtocol: ModelProtocol = {
    api: "openai-responses",
    // @ts-expect-error completions compat cannot pair with the Responses protocol.
    compat: { supportsStore: false, maxTokensField: "max_tokens" as const },
  };
  void messagesWithCompletionsField;
  void responsesWithCompletionsField;
  void responsesWithCacheControl;
  void mismatchedProtocol;
}

// `api` must stay required and span exactly the protocols the registry implements.
expectTypeOf<DiscoveredModel["api"]>().toEqualTypeOf<LiteLLMApi>();
// Every ModelProtocol member must carry both fields, so a builder cannot omit one.
expectTypeOf<ModelProtocol>().toExtend<{ api: LiteLLMApi }>();

function native(id: string): Model<"anthropic-messages" | "openai-completions" | "openai-responses"> {
  return toNativeModels("litellm", "https://proxy.example/v1", discovered(id).models)[0];
}

// Mirrors a model pi assembles from ~/.pi/agent/models.json, where `api` is
// copied verbatim from user config and is not constrained to our protocols.
function foreignApiModel(id: string, api = "google-generative-ai"): Model<LiteLLMApi> {
  return { ...native(id), api } as unknown as Model<LiteLLMApi>;
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
    resolveCredentialRoot: ({ requestBaseUrl }) => requestBaseUrl ?? "https://proxy.example",
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

  it("derives each protocol base from one normalized proxy root", () => {
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://proxy.example/v1/", [
      baseModel,
      { ...baseModel, id: "responses", api: "openai-responses", compat: {} },
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
    ]);

    expect(models.map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://proxy.example/v1" },
      { api: "openai-responses", baseUrl: "https://proxy.example/v1" },
      { api: "anthropic-messages", baseUrl: "https://proxy.example" },
    ]);
  });
});

describe("createLiteLLMProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
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

  it("reprojects stored models to the matching active credential host", () => {
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example/v1" });
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://proxy.example", [
      baseModel,
      { ...baseModel, id: "responses", api: "openai-responses", compat: {} },
    ]);

    expect(value.filterModels?.(models, credential).map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://proxy.example/v1" },
      { api: "openai-responses", baseUrl: "https://proxy.example/v1" },
    ]);
  });

  it("does not list a protocol discovery cannot select", () => {
    // Pi dispatches a model whose api matches none of our listed models without
    // routing through this provider, so we cannot govern its host. Listing it would
    // imply a guarantee the guard does not provide.
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example" });
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://proxy.example", [
      baseModel,
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: undefined },
    ]);

    expect(value.filterModels?.(models, credential).map((model) => model.id)).toEqual(["model"]);
    expect(String(stderr.mock.calls.at(-1)?.[0])).toContain('declares unsupported protocol "anthropic-messages"');
  });

  it("derives the request base from the active root, not the cached model path", () => {
    // Same host, different path prefix, as a multi-tenant proxy produces. The cached
    // path must not survive: this fails if the reprojection reads model.baseUrl.
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example/tenant-a" });
    const cached = { ...native("tenant"), baseUrl: "https://proxy.example/tenant-b/v1" };

    expect(value.filterModels?.([cached], credential)?.map((model) => model.baseUrl)).toEqual([
      "https://proxy.example/tenant-a/v1",
    ]);
  });

  it("derives the request scheme from the active root", () => {
    // Host matches, scheme does not. The active credential decides, so a cached
    // http entry cannot downgrade an https proxy.
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example" });
    const cached = { ...native("scheme"), baseUrl: "http://proxy.example/v1" };

    expect(value.filterModels?.([cached], credential)?.map((model) => model.baseUrl)).toEqual([
      "https://proxy.example/v1",
    ]);
  });

  it("treats a differing port as a different host", () => {
    // The likeliest real mismatch: a local proxy moved port. Port is part of the
    // host comparison, so the cached entry is dropped rather than repointed.
    const value = controller({ resolveCredentialRoot: () => "http://localhost:8000" });
    const cached = { ...native("port"), baseUrl: "http://localhost:4000/v1" };

    expect(value.filterModels?.([cached], credential)).toEqual([]);
  });

  it("blocks a cached path mismatch on the request path too", () => {
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example/tenant-a" });
    const cached = { ...native("tenant"), baseUrl: "https://proxy.example/tenant-b/v1" };

    // Same host, so the model is usable, but the URL must come from the active root.
    const streamed = value.stream(cached, { messages: [] });
    void streamed;

    expect(apiSpies.completions).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "https://proxy.example/tenant-a/v1" }),
      expect.anything(),
      undefined,
    );
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

  it("drops models declaring an unsupported protocol without throwing out of filterModels", () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example" });

    expect(value.filterModels?.([foreignApiModel("gemini"), native("valid")], credential)).toEqual([native("valid")]);
    expect(stderr).toHaveBeenCalledWith(
      'LiteLLM (litellm): LiteLLM model gemini declares unsupported protocol "google-generative-ai"; ' +
        'set "api" to one of openai-completions, openai-responses in models.json\n',
    );
  });

  it("keeps other providers available when a LiteLLM model declares an unsupported protocol", async () => {
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const models = createModels({
      credentials: credentialStore({
        litellm: { type: "api_key", key: "secret", env: { LITELLM_BASE_URL: "https://proxy.example" } },
        other: { type: "api_key", key: "other" },
      }),
    });
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example" });
    await value.refreshModels?.(context(store([foreignApiModel("gemini")]), false));
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

  it("reports the configured env var when no base URL resolves and models are hidden", () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(value.filterModels?.([native("hidden"), native("also-hidden")], credential)).toEqual([]);
    expect(stderr).toHaveBeenCalledWith(
      "LiteLLM (litellm): 2 model(s) hidden because no LiteLLM base URL is configured; " +
        "set LITELLM_BASE_URL or run /login litellm\n",
    );
  });

  it("stays silent when an unconfigured provider has no models to hide", () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(value.filterModels?.([], credential)).toEqual([]);
    expect(stderr).not.toHaveBeenCalled();
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
      { ...discovered("responses").models[0], api: "openai-responses", compat: {} },
    ])[0];
    const value = controller();

    value.stream(responseModel, { messages: [] });

    expect(apiSpies.responses).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
  });

  it("blocks stale hosts on stream and streamSimple before protocol dispatch", async () => {
    const value = controller({ resolveCredentialRoot: () => "https://active.example" });

    expect(() => value.stream(native("resumed"), { messages: [] })).toThrow(
      /stale LiteLLM model host.*network refresh/i,
    );
    expect(() => value.streamSimple(native("default"), { messages: [] })).toThrow(
      /stale LiteLLM model host.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks placeholder and malformed request hosts before protocol dispatch", async () => {
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

  it("blocks unsupported protocols on stream before the registry lookup", () => {
    const value = controller();

    expect(() => value.stream(foreignApiModel("gemini"), { messages: [] })).toThrow(
      /declares unsupported protocol "google-generative-ai"/,
    );
    expect(() => value.streamSimple(foreignApiModel("gemini"), { messages: [] })).toThrow(
      /set "api" to one of openai-completions, openai-responses/,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
  });

  it("maps the Messages protocol to the bare proxy root in the registry", () => {
    // The registry entry stays correct and wired even though discovery cannot select
    // it: the Anthropic API appends /v1/messages to the base URL itself.
    const messagesModel = toNativeModels("litellm", "https://proxy.example/v1/", [
      { ...discovered("messages").models[0], api: "anthropic-messages", compat: undefined },
    ])[0];

    expect(messagesModel.baseUrl).toBe("https://proxy.example");
    expect(Object.keys(createLiteLLMProtocolApis())).toContain("anthropic-messages");
  });

  it("rejects a non-selectable protocol on stream and streamSimple before dispatch", () => {
    const messagesModel = toNativeModels("litellm", "https://proxy.example/v1/", [
      { ...discovered("messages").models[0], api: "anthropic-messages", compat: undefined },
    ])[0];
    const value = controller();

    expect(() => value.stream(messagesModel, { messages: [] })).toThrow(/declares unsupported protocol/);
    expect(() => value.streamSimple(messagesModel, { messages: [] })).toThrow(/declares unsupported protocol/);
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });

  it("rejects a stale-host model on stream and streamSimple before dispatch", () => {
    // The direct-id and session-restore paths reach stream without passing through
    // filterModels, so the guard has to reject there too, not only in the listing.
    const value = controller({ resolveCredentialRoot: () => "https://active.example" });
    const stale = { ...native("stale"), baseUrl: "https://stale.example/v1" };

    expect(() => value.stream(stale, { messages: [] })).toThrow(/stale LiteLLM model host/);
    expect(() => value.streamSimple(stale, { messages: [] })).toThrow(/stale LiteLLM model host/);
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });
});
