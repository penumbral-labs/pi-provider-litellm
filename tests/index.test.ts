import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  type AuthContext,
  type AuthInteraction,
  type Credential,
  createModels,
  InMemoryCredentialStore,
  InMemoryModelsStore,
  type Provider,
  type ProviderModelsStore,
  type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension } from "./test-helpers.js";

const ENV_KEYS = [
  "LITELLM_BASE_URL",
  "LITELLM_API_KEY",
  "LITELLM_API_KEY_HELPER",
  "LITELLM_HEADERS",
  "LITELLM_OFFLINE",
  "LITELLM_VERBOSE_DISCOVERY",
  "LITELLM_ANTHROPIC_API_KEY",
  "LITELLM_ANTHROPIC_HEADERS",
  "LITELLM_DISCOVERY_TIMEOUT_MS",
  "LITELLM_MODELS_DEV",
  "LITELLM_GCLOUD_TOKEN_AUTH",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "STORED_LITELLM_KEY",
  "CUSTOM_LITELLM_KEY",
];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));
// The documentation placeholder. Reaching it with a real key is the failure these
// guards exist to prevent, so tests name it rather than repeating the literal.
const PLACEHOLDER_BASE_URL = "https://litellm.example.com";

vi.unmock("@earendil-works/pi-coding-agent");

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function makeAgentDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "pi-litellm-index-"));
}

function makeJwt(expSeconds: number): string {
  const encode = (value: unknown): string => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode({ exp: expSeconds })}.sig`;
}

async function writeHelper(
  agentDir: string,
  tokens: string[],
  helperPath = join(agentDir, "litellm-token-helper.sh"),
): Promise<string> {
  await writeFile(
    helperPath,
    `#!/usr/bin/env bash\ncount_file="${join(agentDir, "helper-count")}"\ncount=0\n[ -f "$count_file" ] && count=$(cat "$count_file")\ncase "$count" in\n${tokens.map((token, index) => `  ${index}) printf %s '${token}' ;;`).join("\n")}\n  *) printf %s '${tokens.at(-1)}' ;;\nesac\necho $((count + 1)) > "$count_file"\n`,
    { mode: 0o700 },
  );
  return helperPath;
}

async function readHelperCount(agentDir: string): Promise<number> {
  try {
    return Number(await readFile(join(agentDir, "helper-count"), "utf8"));
  } catch {
    return 0;
  }
}

function createModelsStore(models: readonly any[] = []): ProviderModelsStore {
  let entry: Awaited<ReturnType<ProviderModelsStore["read"]>> =
    models.length > 0 ? { models, checkedAt: Date.now() } : undefined;
  return {
    read: async () => entry,
    write: async (next) => {
      entry = next;
    },
    delete: async () => {
      entry = undefined;
    },
  };
}

async function refreshProvider(
  provider: Provider,
  options: Omit<RefreshModelsContext, "store"> & { store?: ProviderModelsStore },
): Promise<readonly unknown[]> {
  await provider.refreshModels?.({ ...options, store: options.store ?? createModelsStore() });
  return provider.getModels();
}

function resolveApiKey(provider: Provider, credential?: Extract<Credential, { type: "api_key" }>) {
  return provider.auth.apiKey?.resolve({
    credential,
    ctx: {
      env: async (name) => process.env[name],
      fileExists: async () => false,
    },
  });
}

function resolveApiKeyWithEnv(provider: Provider, env: Record<string, string | undefined>) {
  return provider.auth.apiKey?.resolve({
    ctx: { env: async (name) => env[name], fileExists: async () => false },
  });
}

function interaction(
  prompt: AuthInteraction["prompt"],
  notify: AuthInteraction["notify"] = vi.fn(),
  signal?: AbortSignal,
): AuthInteraction {
  return { prompt, notify, signal };
}

async function loginOAuth(
  provider: Provider,
  callbacks: {
    onPrompt: (prompt: { message: string; placeholder?: string }) => Promise<string>;
    onAuth?: (event: { url: string; instructions?: string }) => void;
    onProgress?: (message: string) => void;
    signal?: AbortSignal;
  },
) {
  return provider.auth.oauth?.login(
    interaction(
      (prompt) =>
        callbacks.onPrompt({
          message: prompt.message,
          placeholder: "placeholder" in prompt ? prompt.placeholder : undefined,
        }),
      (event) => {
        if (event.type === "auth_url") callbacks.onAuth?.(event);
        if (event.type === "progress") callbacks.onProgress?.(event.message);
      },
      callbacks.signal,
    ),
  );
}

// Start every test from an unset environment. afterEach restores the developer's
// real values, so without this a machine that exports LITELLM_BASE_URL runs a
// different suite than CI does, and "nothing configured" cases silently resolve a
// live host.
beforeEach(() => {
  for (const key of ENV_KEYS) delete process.env[key];
});

afterEach(() => {
  for (const key of ENV_KEYS) {
    const original = ORIGINAL_ENV.get(key);
    if (original === undefined) delete process.env[key];
    else process.env[key] = original;
  }
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("extension startup", () => {
  it("registers one complete native provider and one session handler", async () => {
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();

    await extension(pi);

    expect(pi.providers.map((provider) => provider.id)).toEqual(["litellm"]);
    expect(pi.providers[0]).toEqual(
      expect.objectContaining({
        name: "LiteLLM",
        stream: expect.any(Function),
        streamSimple: expect.any(Function),
        refreshModels: expect.any(Function),
      }),
    );
    expect(pi.handlers.get("session_start")).toHaveLength(1);
    expect(pi.commands.has("litellm-refresh")).toBe(false);
  });

  it("disables models.dev enrichment with LITELLM_MODELS_DEV=0", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    process.env.LITELLM_MODELS_DEV = "0";
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      force: true,
      credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
    });

    expect(urls).not.toContain("https://models.dev/api.json");
    expect(pi.providers[0]?.getModels()[0]?.id).toBe("gpt-5.5");
  });

  it("keeps one provider registration across Pi-managed refresh", async () => {
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] }),
    );
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      force: true,
      credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
    });

    expect(pi.providers.map((provider) => provider.id)).toEqual(["litellm"]);
  });

  it("fails closed on stale-host Pi-managed models offline", async () => {
    process.env.LITELLM_OFFLINE = "1";
    process.env.LITELLM_BASE_URL = "https://active.example.com";
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions",
      baseUrl: "https://stored.example.com/v1",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    };

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: false,
      credential: { type: "api_key", key: "sk-test" },
      store: createModelsStore([stored]),
    });

    expect(
      pi.providers[0]?.filterModels?.(pi.providers[0]!.getModels(), {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://active.example.com" },
      }),
    ).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(pi.providers).toHaveLength(1);
  });

  it("rejects placeholder credential hosts before cached models are available", async () => {
    process.env.LITELLM_OFFLINE = "1";
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://litellm.example.com/v1",
      reasoning: false,
      input: ["text"] as const,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    };

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: false,
      credential: { type: "api_key", key: "sk-test" },
      store: createModelsStore([stored]),
    });

    expect(
      pi.providers[0]?.filterModels?.(pi.providers[0]!.getModels(), {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://litellm.example.com" },
      }),
    ).toEqual([]);
  });

  it("restores same-host Pi-managed models offline with protocol projection", async () => {
    process.env.LITELLM_OFFLINE = "1";
    process.env.LITELLM_BASE_URL = "https://proxy.test/v1";
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://proxy.test/v1",
      reasoning: false,
      input: ["text"] as const,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    };

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: false,
      credential: {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://proxy.test/v1" },
      },
      store: createModelsStore([stored]),
    });

    expect(pi.providers[0]?.getModels()).toEqual([stored]);
    expect(
      pi.providers[0]?.filterModels?.(pi.providers[0]!.getModels(), {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://proxy.test/v1" },
      }),
    ).toEqual([stored]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("ignores legacy cache files without deleting them", async () => {
    process.env.LITELLM_OFFLINE = "1";
    const agentDir = await makeAgentDir();
    const cachePath = join(agentDir, "litellm-models.json");
    const legacyCache = JSON.stringify({ models: [{ id: "legacy-model" }] });
    await writeFile(cachePath, legacyCache, "utf8");
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);
    await refreshProvider(pi.providers[0]!, {
      allowNetwork: false,
      credential: { type: "api_key", key: "sk-test" },
      store: createModelsStore(),
    });

    expect(await readFile(cachePath, "utf8")).toBe(legacyCache);
    expect(pi.providers[0]?.getModels()).toEqual([]);
  });

  it("registers MCP tools after an online Pi-managed model restore", async () => {
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/mcp-rest/tools/list")) {
        return jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions",
      baseUrl: "https://proxy.example.com/v1",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    };

    await expect(
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: {
          type: "api_key",
          key: "sk-test",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
        store: createModelsStore([stored]),
      }),
    ).rejects.toThrow("unexpected URL");

    expect(requestedUrls).toEqual([
      "https://proxy.example.com/model/info",
      "https://proxy.example.com/mcp-rest/tools/list",
    ]);
    // MCP registration runs in the background so a hanging /mcp-rest endpoint
    // cannot block model refresh; wait for it to finish before asserting.
    await vi.waitFor(() => {
      expect(pi.tools.map((tool) => tool.name)).toContain("mcp_brave_search");
    });
    expect(pi.providers[0]?.getModels()).toEqual([stored]);
  });

  it("does not block model refresh on MCP discovery", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let mcpStarted!: () => void;
    let releaseMcp!: (response: Response) => void;
    const started = new Promise<void>((resolve) => {
      mcpStarted = resolve;
    });
    const pendingMcp = new Promise<Response>((resolve) => {
      releaseMcp = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        mcpStarted();
        return pendingMcp;
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    let refreshed = false;
    const refresh = refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://proxy.example.com" },
      },
    }).then(() => {
      refreshed = true;
    });
    await started;
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(refreshed).toBe(true);
    releaseMcp(
      jsonResponse(200, {
        tools: [
          {
            name: "search",
            description: "Search",
            inputSchema: { type: "object", properties: {} },
            mcp_info: { server_name: "brave" },
          },
        ],
      }),
    );
    await refresh;
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContain("mcp_brave_search"));
  });

  it("propagates the production refresh signal through MCP discovery", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    const controller = new AbortController();
    const reason = new Error("refresh cancelled");
    let mcpStarted!: () => void;
    const started = new Promise<void>((resolve) => {
      mcpStarted = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        mcpStarted();
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const refresh = refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://proxy.example.com" },
      },
      signal: controller.signal,
    });
    await started;
    controller.abort(reason);

    await expect(refresh).rejects.toBe(reason);
    expect(pi.providers[0]?.getModels().map((model) => model.id)).toEqual(["fresh-model"]);
  });

  it("retains Pi-managed models when discovery fails", async () => {
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const store = createModelsStore([
      {
        id: "stored-model",
        name: "Stored model",
        provider: "litellm",
        api: "openai-completions",
        baseUrl: "https://proxy.example.com/v1",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128_000,
        maxTokens: 4096,
      },
    ]);
    const credential = {
      type: "api_key" as const,
      key: "sk-test",
      env: { LITELLM_BASE_URL: "https://proxy.example.com" },
    };
    await refreshProvider(pi.providers[0]!, { allowNetwork: false, credential, store });

    await expect(
      refreshProvider(pi.providers[0]!, { allowNetwork: true, force: true, credential, store }),
    ).rejects.toThrow("offline");
    expect(pi.providers[0]?.getModels().map((model) => model.id)).toEqual(["stored-model"]);
    expect(pi.providers).toHaveLength(1);
  });

  it("registers the API key as an explicit environment reference", async () => {
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);

    expect(pi.providers[0]?.auth.apiKey).toMatchObject({ name: "LiteLLM API key", login: expect.any(Function) });
  });

  it('treats literal "undefined" env values as unset', async () => {
    process.env.LITELLM_BASE_URL = "undefined";
    process.env.LITELLM_API_KEY = "undefined";
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);

    expect(pi.providers[0]?.baseUrl).toBe("https://litellm.example.com/v1");
  });

  it("applies LiteLLM request compatibility hooks to configured provider aliases", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: "$LITELLM_ANTHROPIC_API_KEY",
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    process.env.LITELLM_ANTHROPIC_API_KEY = "anthropic-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const result = await pi.handlers.get("before_provider_request")?.[0]?.(
      { payload: { model: "kimi-k2.6" } },
      { model: { provider: "litellm-anthropic", id: "kimi-k2.6", api: "openai-completions" } },
    );

    expect(result).toMatchObject({
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });
  });

  it("returns a native API-key credential without discovery side effects", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    process.env.LITELLM_VERBOSE_DISCOVERY = "1";
    const seenRequests: Array<{ url: string; authorization: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({
        url,
        authorization: new Headers(init?.headers).get("authorization") ?? "",
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const prompt = vi.fn().mockResolvedValueOnce(" http://127.0.0.1:4000/v1 ").mockResolvedValueOnce(" sk-login ");
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(interaction(prompt));

    expect(prompt).toHaveBeenNthCalledWith(1, expect.objectContaining({ type: "text" }));
    expect(prompt).toHaveBeenNthCalledWith(2, expect.objectContaining({ type: "secret" }));
    expect(seenRequests).toEqual([]);
    expect(credential).toEqual({
      type: "api_key",
      key: "sk-login",
      env: { LITELLM_BASE_URL: "http://127.0.0.1:4000" },
    });
    delete process.env.LITELLM_VERBOSE_DISCOVERY;
  });

  it("checks command-backed auth without executing the helper", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["helper-key"]);
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const result = await pi.providers[0]?.auth.apiKey?.check?.({
      ctx: { env: async (name) => process.env[name], fileExists: async () => false },
    });

    expect(result).toEqual({ type: "api_key", source: "LITELLM_API_KEY_HELPER" });
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("resolves native auth from the injected context instead of process env", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://process.example.com";
    process.env.LITELLM_API_KEY = "process-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY: "context-key",
        LITELLM_HEADERS: '{"x-tenant":"context"}',
      }),
    ).resolves.toMatchObject({
      auth: {
        apiKey: "context-key",
        headers: { "x-tenant": "context" },
      },
      source: "LITELLM_API_KEY",
    });
  });

  it("executes only the helper supplied by the injected auth context", async () => {
    const agentDir = await makeAgentDir();
    const contextHelper = await writeHelper(agentDir, ["context-helper-key"]);
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY_HELPER: contextHelper,
        LITELLM_API_KEY: "context-env-key",
      }),
    ).resolves.toMatchObject({ auth: { apiKey: "context-helper-key" }, source: "LITELLM_API_KEY_HELPER" });
    expect(await readHelperCount(agentDir)).toBe(1);
  });

  it("rejects shell expressions in helper commands", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY_HELPER: "printf safe-key; printf injected-key",
      }),
    ).rejects.toThrow("shell syntax is not supported");
  });

  it("preserves backslashes in helper executable paths", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["backslash-key"], join(agentDir, "token\\helper.sh"));
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY_HELPER: `"${helperPath}"`,
      }),
    ).resolves.toMatchObject({ auth: { apiKey: "backslash-key" } });
  });

  it("resolves configured key templates from the injected auth context", async () => {
    const agentDir = await makeAgentDir();
    const lowerPriorityHelper = await writeHelper(agentDir, ["unexpected-helper-key"]);
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { litellm: { apiKey: "$CUSTOM_LITELLM_KEY" } } } }),
      "utf8",
    );
    process.env.CUSTOM_LITELLM_KEY = "process-configured-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        CUSTOM_LITELLM_KEY: "context-configured-key",
        LITELLM_API_KEY_HELPER: lowerPriorityHelper,
        LITELLM_API_KEY: "context-default-key",
      }),
    ).resolves.toMatchObject({
      auth: { apiKey: "context-configured-key" },
      source: "$CUSTOM_LITELLM_KEY",
    });
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("leaves model refresh to Pi after login", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers).toHaveLength(1);
    expect(pi.providers[0]?.getModels()).toEqual([]);

    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });

    const registeredModels = pi.providers[1]?.getModels() as unknown as Array<{ id: string }> | undefined;
    expect(pi.providers).toHaveLength(1);
    expect(registeredModels).toBeUndefined();
    expect(vi.mocked(globalThis.fetch).mock.calls.every(([url]) => !String(url).endsWith("/model/info"))).toBe(true);
  });

  it("streams OAuth requests to the credential host over a conflicting environment host", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://environment.example.com";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (!url.endsWith("/chat/completions")) throw new Error(`unexpected URL: ${url}`);
      return new Response(
        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n',
        { headers: { "content-type": "text/event-stream" } },
      );
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;
    const credentials = new InMemoryCredentialStore();
    await credentials.modify(provider.id, async () => ({
      type: "oauth",
      access: "sk-oauth",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://credential.example.com",
    }));
    const authContext: AuthContext = {
      env: async (name) => process.env[name],
      fileExists: async () => false,
    };
    const models = createModels({ credentials, modelsStore: new InMemoryModelsStore(), authContext });
    models.setProvider(provider);
    // Stored without the /v1 suffix so the asserted URL can only be produced by
    // the guard reprojecting through the protocol registry.
    const model = {
      id: "oauth-model",
      name: "OAuth model",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://credential.example.com",
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 4096,
      maxTokens: 1024,
    };

    const result = await models.complete(model, { messages: [] });

    expect(result.stopReason).toBe("stop");
    expect(requestedUrls).toEqual(["https://credential.example.com/v1/chat/completions"]);
  });

  // Drives real credential resolution through createModels so each transition
  // exercises the same auth -> stream ordering pi uses in production.
  async function oauthTransitionHarness(envBaseUrl?: string) {
    const agentDir = await makeAgentDir();
    if (envBaseUrl) process.env.LITELLM_BASE_URL = envBaseUrl;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const requests: Array<{ url: string; authorization: string | null }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({ url, authorization: new Headers(init?.headers as Record<string, string>).get("authorization") });
      if (!url.endsWith("/chat/completions")) throw new Error(`unexpected URL: ${url}`);
      return new Response(
        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n',
        { headers: { "content-type": "text/event-stream" } },
      );
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;
    const authContext: AuthContext = {
      env: async (name) => process.env[name],
      fileExists: async () => false,
    };
    const completeWith = async (credential: Credential, baseUrl: string) => {
      const credentials = new InMemoryCredentialStore();
      await credentials.modify(provider.id, async () => credential);
      const models = createModels({ credentials, modelsStore: new InMemoryModelsStore(), authContext });
      models.setProvider(provider);
      return models.complete(
        {
          id: "shared-model",
          name: "Shared model",
          provider: "litellm",
          api: "openai-completions" as const,
          baseUrl,
          reasoning: false,
          input: ["text"] as ("text" | "image")[],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 4096,
          maxTokens: 1024,
        },
        { messages: [] },
      );
    };
    const oauth = (access: string, baseUrl: string): Credential => ({
      type: "oauth",
      access,
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl,
    });
    return { provider, requests, urls: () => requests.map((entry) => entry.url), completeWith, oauth };
  }

  it("does not pin api-key requests to a previously authenticated OAuth host", async () => {
    const harness = await oauthTransitionHarness("https://environment.example.com");

    // Authenticate via SSO first, which is what makes the extension remember a
    // runtime host at all.
    await harness.completeWith(harness.oauth("sk-oauth", "https://sso.example.com"), "https://sso.example.com/v1");
    // Then switch to an api key for a different proxy, as /logout followed by
    // env-based auth does, without reloading the extension.
    const switched = await harness.completeWith(
      { type: "api_key", key: "sk-second", env: { LITELLM_BASE_URL: "https://second.example.com" } },
      "https://second.example.com/v1",
    );

    expect(switched.stopReason).toBe("stop");
    expect(harness.urls()).toEqual([
      "https://sso.example.com/v1/chat/completions",
      "https://second.example.com/v1/chat/completions",
    ]);
  });

  it("follows the live credential when an api key reuses a previous SSO access token", async () => {
    // Same key string, different proxies. Keying the remembered root on the token
    // alone is not enough here: the request base URL has to outrank it, or the
    // provider stays pinned to the SSO host and reports the live host as stale.
    const harness = await oauthTransitionHarness("https://environment.example.com");

    await harness.completeWith(harness.oauth("sk-shared", "https://sso.example.com"), "https://sso.example.com/v1");
    const switched = await harness.completeWith(
      { type: "api_key", key: "sk-shared", env: { LITELLM_BASE_URL: "https://second.example.com" } },
      "https://second.example.com/v1",
    );

    expect(switched.stopReason).toBe("stop");
    expect(harness.urls()).toEqual([
      "https://sso.example.com/v1/chat/completions",
      "https://second.example.com/v1/chat/completions",
    ]);
    expect(harness.requests.at(-1)?.authorization).toBe("Bearer sk-shared");
  });

  it("follows a refreshed OAuth access token to the credential host", async () => {
    // A refresh replaces the access token. If the remembered root stayed bound to
    // the pre-refresh token it would miss, and the conflicting environment host
    // would decide the request instead.
    const harness = await oauthTransitionHarness("https://environment.example.com");

    await harness.completeWith(harness.oauth("sk-before", "https://sso.example.com"), "https://sso.example.com/v1");
    const refreshed = await harness.completeWith(
      harness.oauth("sk-after", "https://sso.example.com"),
      "https://sso.example.com/v1",
    );

    expect(refreshed.stopReason).toBe("stop");
    expect(harness.urls()).toEqual([
      "https://sso.example.com/v1/chat/completions",
      "https://sso.example.com/v1/chat/completions",
    ]);
    expect(harness.requests.map((entry) => entry.authorization)).toEqual(["Bearer sk-before", "Bearer sk-after"]);
  });

  it("contains a stored invalid OAuth root at the real registry seam", async () => {
    // The throw originates inside getProviderAuth, which runs our own oauth.toAuth,
    // so a stub getProviderAuth cannot reach this path. Drive real resolution.
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, []));
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;
    const credentials = new InMemoryCredentialStore();
    // Exactly what /login litellm stored before the prompt validated the scheme.
    await credentials.modify(provider.id, async () => ({
      type: "oauth",
      access: "sk-sso",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "localhost:4000",
    }));
    const authContext: AuthContext = { env: async (name) => process.env[name], fileExists: async () => false };
    const models = createModels({ credentials, modelsStore: new InMemoryModelsStore(), authContext });
    models.setProvider(provider);
    const registry = { getProviderAuth: () => models.getAuth(provider.id), getProvider: () => provider };
    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];

    // A LiteLLM turn and a turn on another provider both have to survive it.
    await expect(
      beforeAgentStart?.({ systemPrompt: "p" }, { model: { provider: "litellm" }, modelRegistry: registry }),
    ).resolves.toBeUndefined();
    await expect(
      beforeAgentStart?.({ systemPrompt: "p" }, { model: { provider: "anthropic" }, modelRegistry: registry }),
    ).resolves.toBeUndefined();

    // One bounded diagnostic across both turns, and no request attempted.
    const litellmWrites = stderr.mock.calls.filter(([line]) => String(line).startsWith("LiteLLM (litellm):"));
    expect(litellmWrites).toHaveLength(1);
    expect(String(litellmWrites[0]?.[0])).toContain("Invalid LiteLLM base URL");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("recovers once the stored credential root is corrected", async () => {
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url.startsWith("https://recovered.example.com/")) {
        return jsonResponse(200, [{ name: "deep-research", description: "Research", enabled: true }]);
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;
    const credentials = new InMemoryCredentialStore();
    const authContext: AuthContext = { env: async (name) => process.env[name], fileExists: async () => false };
    const oauthCredential = (baseUrl: string): Credential => ({
      type: "oauth",
      access: "sk-sso",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl,
    });
    const startWith = async (baseUrl: string) => {
      await credentials.modify(provider.id, async () => oauthCredential(baseUrl));
      const models = createModels({ credentials, modelsStore: new InMemoryModelsStore(), authContext });
      models.setProvider(provider);
      return pi.handlers.get("before_agent_start")?.[0]?.(
        { systemPrompt: "Base prompt" },
        { modelRegistry: { getProviderAuth: () => models.getAuth(provider.id), getProvider: () => provider } },
      );
    };

    expect(await startWith("localhost:4000")).toBeUndefined();
    const recovered = await startWith("https://recovered.example.com");

    expect(recovered?.systemPrompt).toContain("Base prompt");
    expect(recovered?.systemPrompt).toContain("<litellm_skills>");
  });

  it.each([
    ["a scheme-less URL", "localhost:4000", /include the scheme/],
    ["a non-http scheme", "ftp://proxy.example.com", /must use http or https/],
    ["the documentation placeholder", PLACEHOLDER_BASE_URL, /documentation placeholder/],
  ])("rejects %s at api-key login before storing it", async (_label, typed, expected) => {
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await expect(pi.providers[0]?.auth.apiKey?.login?.(interaction(vi.fn().mockResolvedValue(typed)))).rejects.toThrow(
      expected,
    );
  });

  it("rejects a scheme-less URL at SSO login before storing it", async () => {
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async () => "localhost:4000",
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow(/include the scheme/);
  });

  it.each([
    ["an api key", { type: "api_key" as const, key: "sk-secret", env: { LITELLM_BASE_URL: PLACEHOLDER_BASE_URL } }],
    [
      "an SSO credential",
      {
        type: "oauth" as const,
        access: "sk-secret",
        refresh: "",
        expires: Number.MAX_SAFE_INTEGER,
        baseUrl: PLACEHOLDER_BASE_URL,
      },
    ],
  ])("never sends %s to the placeholder host during discovery", async (_label, credential) => {
    process.env.LITELLM_BASE_URL = PLACEHOLDER_BASE_URL;
    process.env.LITELLM_API_KEY = "sk-secret";
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await expect(refreshProvider(pi.providers[0]!, { allowNetwork: true, force: true, credential })).rejects.toThrow(
      /placeholder/,
    );

    // No request at all, so the real key cannot have reached the placeholder domain.
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("never exposes the key to LiteLLM tool surfaces on a placeholder host", async () => {
    process.env.LITELLM_BASE_URL = PLACEHOLDER_BASE_URL;
    process.env.LITELLM_API_KEY = "sk-secret";
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const skillTool = pi.tools.find((tool) => tool.name === "litellm_skill_list");

    await expect(
      skillTool?.execute?.("call-1", {}, undefined, undefined, {
        modelRegistry: {
          getProviderAuth: async () => ({
            auth: { apiKey: "sk-secret" },
            env: { LITELLM_BASE_URL: PLACEHOLDER_BASE_URL },
          }),
          getProvider: () => pi.providers[0],
        },
      }),
    ).rejects.toThrow(/no credentials for litellm/);

    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([
    ["placeholder", PLACEHOLDER_BASE_URL],
    ["scheme-less", "localhost:4000"],
  ])("keeps a %s base URL from failing agent start", async (_label, baseUrl) => {
    // before_agent_start runs on every turn, including turns that never touch
    // LiteLLM, so a bad base URL must not surface as an extension error there.
    // filterModels owns the user-facing explanation instead.
    process.env.LITELLM_BASE_URL = baseUrl;
    process.env.LITELLM_API_KEY = "sk-secret";
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];

    const result = await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => ({ auth: { apiKey: "sk-secret" }, env: { LITELLM_BASE_URL: baseUrl } }),
          getProvider: () => pi.providers[0],
        },
      },
    );

    expect(result).toBeUndefined();
    expect(fetchMock.mock.calls.map(([url]) => String(url))).toEqual([]);
  });

  it("rejects a stale-host model on the SSO path before any request", async () => {
    // oauth.toAuth deliberately returns no baseUrl, so Pi does not overwrite the
    // model's host before dispatch and the guard's host comparison is meaningful on
    // this path too. Previously Pi replaced it with the credential root, which made
    // the comparison compare the credential against itself and silently repointed a
    // stale model instead of rejecting it.
    const harness = await oauthTransitionHarness();

    const result = await harness.completeWith(
      harness.oauth("sk-sso", "https://sso.example.com"),
      "https://stale.example.com/v1",
    );

    expect(result.stopReason).toBe("error");
    expect(harness.urls()).toEqual([]);
  });

  it("lets an explicit request base URL outrank the remembered SSO root", async () => {
    // Isolates precedence from invalidation: the memo is still live here, because
    // OAuth is still the active credential. Only the ordering decides the host, so
    // this fails if the memo is consulted before the explicit request value.
    const harness = await oauthTransitionHarness();
    const auth = await harness.provider.auth.oauth?.toAuth({
      type: "oauth",
      access: "sk-memo",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://sso.example.com",
    });
    // toAuth must not carry a baseUrl: Pi would overwrite the model host with it and
    // defeat the stale-host comparison. The root reaches the request path via the
    // token-keyed runtime root instead.
    expect(auth?.baseUrl).toBeUndefined();
    const model = {
      id: "explicit-model",
      name: "Explicit model",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://explicit.example.com/v1",
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 4096,
      maxTokens: 1024,
    };

    const stream = harness.provider.stream(model, { messages: [] }, {
      apiKey: "sk-memo",
      env: { LITELLM_BASE_URL: "https://explicit.example.com" },
    } as never);
    for await (const _event of stream) {
      // drain
    }

    expect(harness.urls()).toEqual(["https://explicit.example.com/v1/chat/completions"]);
  });

  it("discards the remembered SSO root when an api key resolves", async () => {
    // Isolates invalidation from precedence. With no base URL configured anywhere
    // there is no request value to outrank the memo, so only discarding it on the
    // non-OAuth transition keeps the api key off the SSO host. Driven at the auth
    // seam because the transition is what matters, not the streaming stack.
    const harness = await oauthTransitionHarness();
    const model = {
      id: "orphan-model",
      name: "Orphan model",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://sso.example.com/v1",
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 4096,
      maxTokens: 1024,
    };
    const streamOnce = () => harness.provider.stream(model, { messages: [] }, { apiKey: "sk-shared" } as never);

    await harness.provider.auth.oauth?.toAuth({
      type: "oauth",
      access: "sk-shared",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://sso.example.com",
    });
    // Precondition: the remembered root is live and would carry this request.
    expect(streamOnce).not.toThrow();

    await harness.provider.auth.apiKey?.resolve({
      ctx: { env: async (name: string) => process.env[name] },
      credential: { type: "api_key", key: "sk-shared" },
    } as never);

    expect(streamOnce).toThrow(/do not identify a LiteLLM model host/);
  });

  it("does not lend the remembered SSO root to a different credential", async () => {
    // Pins the token binding specifically. Precedence cannot help here (no request
    // base URL) and neither can invalidation (no api key resolved), so an unbound
    // memo would hand this request the SSO host.
    const harness = await oauthTransitionHarness();
    const model = {
      id: "other-cred",
      name: "Other credential",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://sso.example.com/v1",
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 4096,
      maxTokens: 1024,
    };
    await harness.provider.auth.oauth?.toAuth({
      type: "oauth",
      access: "sk-owner",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://sso.example.com",
    });

    // The owning token still reaches it...
    expect(() => harness.provider.stream(model, { messages: [] }, { apiKey: "sk-owner" } as never)).not.toThrow();
    // ...but a different credential must not.
    expect(() => harness.provider.stream(model, { messages: [] }, { apiKey: "sk-other" } as never)).toThrow(
      /do not identify a LiteLLM model host/,
    );
  });

  it("prefers the credential's own host over ambient env on the Skills surface", async () => {
    // getRuntimeAuth reads auth.env[LITELLM_BASE_URL]; dropping that read silently
    // sends the credential's key to whatever host the environment names.
    process.env.LITELLM_BASE_URL = "https://environment.example.com";
    const requested: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      requested.push(String(input));
      if (String(input).endsWith("/model/info")) return jsonResponse(200, { data: [] });
      return jsonResponse(200, []);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const tool = pi.tools.find((candidate) => candidate.name === "litellm_skill_list");

    await tool?.execute?.("call-1", {}, undefined, undefined, {
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "sk-credential" },
          env: { LITELLM_BASE_URL: "https://credential.example.com" },
        }),
        getProvider: () => pi.providers[0],
      },
    });

    expect(requested.filter((url) => url.startsWith("https://credential.example.com/"))).not.toHaveLength(0);
    expect(requested.filter((url) => url.startsWith("https://environment.example.com/"))).toHaveLength(0);
  });

  it("drops null header sentinels instead of serializing them", async () => {
    // ProviderHeaders allows null as pi's header-removal sentinel. It must never reach
    // the wire as the string "null".
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/model/info")) return jsonResponse(200, { data: [] });
      return jsonResponse(200, []);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const tool = pi.tools.find((candidate) => candidate.name === "litellm_skill_list");

    await tool?.execute?.("call-1", {}, undefined, undefined, {
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "sk-test", headers: { "x-keep": "yes", "x-drop": null } },
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        }),
        getProvider: () => pi.providers[0],
      },
    });

    const headers = new Headers(fetchMock.mock.lastCall?.[1]?.headers as Record<string, string>);
    expect(headers.get("x-keep")).toBe("yes");
    expect(headers.has("x-drop")).toBe(false);
  });

  it("routes OAuth requests by remembered root when nothing supplies a request base URL", async () => {
    // No LITELLM_BASE_URL at all: OAuth resolution contributes no request env, so
    // the remembered root is the only thing that can identify the proxy.
    const harness = await oauthTransitionHarness();

    const result = await harness.completeWith(
      harness.oauth("sk-only", "https://sso.example.com"),
      "https://sso.example.com/v1",
    );

    expect(result.stopReason).toBe("stop");
    expect(harness.urls()).toEqual(["https://sso.example.com/v1/chat/completions"]);
  });

  it("registers unconfigured when the configured base URL is invalid", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "localhost:4000";
    process.env.LITELLM_API_KEY = "sk-test";
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await expect(extension(pi)).resolves.toBeUndefined();

    expect(pi.providers).toHaveLength(1);
    await expect(resolveApiKey(pi.providers[0]!)).resolves.toMatchObject({ auth: { apiKey: "sk-test" } });
    // A non-empty list is required: filterModels([]) returns [] even when fully fail-open.
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://localhost:4000/v1",
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    };
    expect(pi.providers[0]?.filterModels?.([stored], { type: "api_key", key: "sk-test" })).toEqual([]);
    expect(stderr).toHaveBeenCalledWith(
      "LiteLLM (litellm): Invalid LiteLLM base URL; a network refresh with a valid URL is required\n",
    );
  });

  it("leaves /login litellm to Pi's registered OAuth provider", async () => {
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.commands.has("login")).toBe(false);
    expect(pi.providers[0]?.auth.oauth).toBeDefined();
    expect(pi.handlers.has("input")).toBe(false);
  });

  it("uses the login cache timestamp for later stale auto-refresh", async () => {
    const agentDir = await makeAgentDir();
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    delete process.env.LITELLM_DISCOVERY_TIMEOUT_MS;
    const loginTime = new Date("2026-05-01T00:00:00.000Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(loginTime);

    let callCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        callCount++;
        return jsonResponse(200, {
          data: [{ model_name: `vidaimock-openai-${callCount}`, model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    expect(callCount).toBe(0);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });
    expect(callCount).toBe(0);
    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: { type: "oauth", ...credential } }), "utf8");

    vi.mocked(Date.now).mockReturnValue(loginTime + 25 * 60 * 60 * 1000);
    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler({ reason: "start" }, { sessionManager: { getSessionFile: () => undefined } });
    }

    expect(callCount).toBe(0);
  });

  it("does not re-run command-backed helpers after refreshing login credentials", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const now = new Date("2026-05-29T21:00:00.000Z").getTime();
    const first = makeJwt(Math.floor(now / 1000) + 60);
    const second = makeJwt(Math.floor(now / 1000) + 3600);
    const helperPath = await writeHelper(agentDir, [first, second, "unexpected-third-token"]);
    vi.spyOn(Date, "now").mockReturnValue(now);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] }),
    );

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(
      interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce(`!${helperPath}`)),
    );
    const firstAuth = await resolveApiKey(pi.providers[0]!, credential);
    const secondAuth = await resolveApiKey(pi.providers[0]!, credential);

    expect(firstAuth?.auth.apiKey).toBe(first);
    expect(secondAuth?.auth.apiKey).toBe(second);
    expect(await readHelperCount(agentDir)).toBe(2);
  });

  it("resolves opaque command-backed API keys for each request", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const helperPath = await writeHelper(agentDir, ["opaque-first", "opaque-second", "unexpected-third"]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] }),
    );

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(
      interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce(`!${helperPath}`)),
    );

    expect((await resolveApiKey(pi.providers[0]!, credential))?.auth.apiKey).toBe("opaque-first");
    expect((await resolveApiKey(pi.providers[0]!, credential))?.auth.apiKey).toBe("opaque-second");
    expect(await readHelperCount(agentDir)).toBe(2);
  });

  it("executes an OAuth refresh command only during refresh", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["refreshed-token", "unexpected-second-run"]);
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = {
      type: "oauth" as const,
      access: "expired-token",
      refresh: `!${helperPath}`,
      expires: 0,
      baseUrl: "https://proxy.example.com",
    };

    const refreshed = await pi.providers[0]?.auth.oauth?.refresh(credential);
    expect(await readHelperCount(agentDir)).toBe(1);
    const refreshedAuth = await pi.providers[0]?.auth.oauth?.toAuth(refreshed!);
    expect(refreshedAuth).toMatchObject({ apiKey: "refreshed-token" });
    expect(refreshedAuth).not.toHaveProperty("baseUrl");
    expect(await readHelperCount(agentDir)).toBe(1);
  });

  it("uses the refreshed OAuth access token during discovery", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["unexpected-helper-run"]);
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    process.env.LITELLM_HEADERS = '{"x-tenant":"tenant-a"}';
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] });
      return jsonResponse(200, { tools: [] });
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "5000";
    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: {
        type: "oauth",
        access: "already-refreshed",
        refresh: `!${helperPath}`,
        expires: Date.now() + 60_000,
        baseUrl: "https://current.example.com",
      },
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe("https://current.example.com/model/info");
    expect(new Headers(fetchMock.mock.calls[0]?.[1]?.headers).get("authorization")).toBe("Bearer already-refreshed");
    expect(new Headers(fetchMock.mock.calls[0]?.[1]?.headers).get("x-tenant")).toBe("tenant-a");
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("enterprise SSO login generates a virtual key and uses it as the access token", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenRequests: Array<{ url: string; method: string; authorization: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({
        url,
        method: String(init?.method ?? "GET"),
        authorization: new Headers(init?.headers).get("authorization") ?? "",
      });
      if (url.endsWith("/key/generate")) return jsonResponse(200, { key: "sk-virtual-abc" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const authInfos: Array<{ url: string; instructions?: string }> = [];
    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return `Bearer ${jwt}`;
        return "y";
      },
      onAuth: (info) => authInfos.push(info),
      signal: new AbortController().signal,
    });

    expect(authInfos).toEqual([
      {
        type: "auth_url",
        url: "https://proxy.example.com/sso/key/generate",
        instructions: "Authenticate via SSO, then copy your token from the LiteLLM UI.",
      },
    ]);
    expect(credential).toMatchObject({
      access: "sk-virtual-abc",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://proxy.example.com",
    });
    const ssoAuth = await pi.providers[0]?.auth.oauth?.toAuth(credential!);
    expect(ssoAuth).toMatchObject({ apiKey: "sk-virtual-abc" });
    expect(ssoAuth).not.toHaveProperty("baseUrl");
    expect(seenRequests).toContainEqual(
      expect.objectContaining({
        url: "https://proxy.example.com/key/generate",
        method: "POST",
        authorization: `Bearer ${jwt}`,
      }),
    );
    expect(seenRequests.every(({ url }) => !url.endsWith("/model/info"))).toBe(true);
  });

  it("enterprise SSO login strips Bearer prefix from pasted SSO token", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenAuthorizations: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenAuthorizations.push(new Headers(init?.headers).get("authorization") ?? "");
      if (url.endsWith("/key/generate")) return jsonResponse(200, { key: "sk-stripped" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return `  Bearer  ${jwt}  `;
        return "y";
      },
      signal: new AbortController().signal,
    });

    expect(seenAuthorizations[0]).toBe(`Bearer ${jwt}`);
  });

  it("enterprise SSO login honors the expiry returned with a generated virtual key", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const keyExpiresAt = new Date(Date.now() + 60 * 60 * 1000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/key/generate"))
        return jsonResponse(200, { key: "sk-expiring", expires: keyExpiresAt.toISOString() });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    expect(credential).toMatchObject({ access: "sk-expiring", refresh: "" });
    expect(credential?.expires).toBe(keyExpiresAt.getTime() - 5 * 60 * 1000);
  });

  it("enterprise SSO login falls back to JWT when virtual key generation times out", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const progress = vi.fn();
    const timeoutController = new AbortController();
    const nativeTimeout = AbortSignal.timeout.bind(AbortSignal);
    vi.spyOn(AbortSignal, "timeout")
      .mockImplementationOnce(() => timeoutController.signal)
      .mockImplementation(nativeTimeout);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (url.endsWith("/key/generate"))
        return new Promise<Response>((_, reject) => {
          if (init?.signal?.aborted) {
            reject(init.signal.reason);
            return;
          }
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason ?? new Error("aborted")), {
            once: true,
          });
        });
      if (url.endsWith("/model/info"))
        return Promise.resolve(jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] }));
      return Promise.reject(new Error(`unexpected URL: ${url}`));
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const controller = new AbortController();
    const loginPromise = loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      onProgress: progress,
      signal: controller.signal,
    });
    await vi.waitFor(() =>
      expect(fetchSpy.mock.calls.some(([input]) => String(input).endsWith("/key/generate"))).toBe(true),
    );
    timeoutController.abort(new Error("test timeout"));

    const credential = await loginPromise;
    expect(credential).toMatchObject({ access: jwt, refresh: "" });
    expect(progress).toHaveBeenCalledWith(expect.stringContaining("virtual key generation failed"));
  });

  it("enterprise SSO login rejects when the caller cancels virtual key generation", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      return new Promise<Response>((resolve, reject) => {
        if (init?.signal?.aborted) {
          reject(init.signal.reason);
          return;
        }
        if (url.endsWith("/key/generate")) {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
          return;
        }
        if (url.endsWith("/model/info")) {
          resolve(jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] }));
          return;
        }
        reject(new Error(`unexpected URL: ${url}`));
      });
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const controller = new AbortController();
    const reason = new Error("caller cancelled login");
    const loginPromise = loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: controller.signal,
    });
    await vi.waitFor(() =>
      expect(fetchSpy.mock.calls.some(([input]) => String(input).endsWith("/key/generate"))).toBe(true),
    );
    controller.abort(reason);

    await expect(loginPromise).rejects.toBe(reason);
  });

  it("enterprise SSO login uses JWT directly when user answers no to virtual key generation", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenRequests: Array<{ url: string; method: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({ url, method: String(init?.method ?? "GET") });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "no";
      },
      signal: new AbortController().signal,
    });

    expect(credential).toMatchObject({ access: jwt, refresh: "" });
    expect(credential?.expires).toBeLessThan(Number.MAX_SAFE_INTEGER);
    expect(seenRequests.every(({ url }) => !url.includes("key/generate"))).toBe(true);
  });

  it("enterprise SSO refresh rejects expiring generated virtual keys without a refresh path", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const keyExpiresAt = new Date(Date.now() + 60 * 60 * 1000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/key/generate"))
        return jsonResponse(200, { key: "sk-expiring", expires: keyExpiresAt.toISOString() });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    await expect(pi.providers[0]?.auth.oauth?.refresh(credential!)).rejects.toThrow(
      "LiteLLM credential cannot be refreshed; run /login litellm again",
    );
  });

  it("enterprise SSO login falls back to JWT when virtual key generation fails", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const progress = vi.fn();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/key/generate")) return jsonResponse(403, { error: "forbidden" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      onProgress: progress,
      signal: new AbortController().signal,
    });

    expect(credential).toMatchObject({ access: jwt, refresh: "" });
    expect(progress).toHaveBeenCalledWith(expect.stringContaining("virtual key generation failed"));
  });

  it("enterprise SSO login throws when SSO token is empty", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (options) => {
          if (options.placeholder) return "https://proxy.example.com";
          if (options.message.includes("Select login method")) return "2";
          return "";
        },
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow("SSO token is required");
  });
});

describe("multi-provider hardening", () => {
  it("does not register the default env key for an alias missing its apiKey", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": { baseUrl: "https://litellm-anthropic.example.com" },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(await resolveApiKey(pi.providers[0]!)).toMatchObject({ auth: { apiKey: "openai-key" } });
    expect(pi.providers[1]?.id).toBe("litellm-anthropic");
    expect(await resolveApiKey(pi.providers[1]!)).toBeUndefined();
  });

  it("sends custom headers when generating a login virtual key", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: { providers: { litellm: { headers: { "x-litellm-customer-id": "team-a" } } } },
      }),
      "utf8",
    );
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenRequests: Array<{ url: string; customer: string | null }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({ url, customer: new Headers(init?.headers).get("x-litellm-customer-id") });
      if (url.endsWith("/key/generate")) return jsonResponse(200, { key: "sk-virtual-abc" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-5", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    expect(seenRequests).toContainEqual({ url: "https://proxy.example.com/key/generate", customer: "team-a" });
  });

  it("drops non-primitive header values instead of stringifying them", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: "$LITELLM_ANTHROPIC_API_KEY",
              headers: { "x-obj": { team: "a" }, "x-null": null, "x-num": 30, "x-bool": false },
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[1]?.headers).toEqual({ "x-num": "30", "x-bool": "false" });
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("x-obj"));
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("x-null"));
  });
});
