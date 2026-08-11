import type {
  Api,
  Credential,
  Model,
  ProviderAuth,
  ProviderModelsStore,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
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
      policy: { normalizeStrictToolMessages: true, normalizeThinkTags: true },
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
    // Cache entries predating request policies carry none, so the policy is
    // re-derived from stored compatibility evidence rather than the route name.
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
