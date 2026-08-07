import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ModelsStoreEntry } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension, type TestPi } from "./test-helpers.js";

vi.unmock("@earendil-works/pi-coding-agent");

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function refreshProvider(pi: TestPi, allowNetwork = true, signal?: AbortSignal): Promise<void> {
  let stored: ModelsStoreEntry | undefined;
  await pi.providers[0]?.refreshModels?.({
    allowNetwork,
    stored,
    publish: async ({ persist, update }) => {
      stored = persist ?? undefined;
      update?.();
      return true;
    },
    credential: {
      type: "api_key",
      key: process.env.LITELLM_API_KEY ?? "sk-test",
      env: { LITELLM_BASE_URL: process.env.LITELLM_BASE_URL ?? "https://litellm.example.com" },
    },
    signal: signal ?? new AbortController().signal,
  });
}

async function startSession(pi: TestPi, context: unknown): Promise<void> {
  const handler = pi.handlers.get("session_start")?.[0];
  expect(handler).toBeTypeOf("function");
  await handler!({ reason: "startup" }, context);
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  delete process.env.LITELLM_BASE_URL;
  delete process.env.LITELLM_API_KEY;
  delete process.env.LITELLM_HEADERS;
  delete process.env.LITELLM_DISCOVERY_TIMEOUT_MS;
  delete process.env.LITELLM_GCLOUD_TOKEN_AUTH;
  delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
  delete process.env.PI_OFFLINE;
});

describe("feature parity", () => {
  it("registers a command-backed gcloud token provider key when ADC auth is enabled", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const adcPath = join(agentDir, "adc.json");
    await writeFile(
      adcPath,
      JSON.stringify({
        type: "authorized_user",
        client_id: "client-id",
        client_secret: "client-secret",
        refresh_token: "refresh-token",
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_GCLOUD_TOKEN_AUTH = "1";
    process.env.GOOGLE_APPLICATION_CREDENTIALS = adcPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      pi.providers[0]?.auth.apiKey?.check?.({
        signal: new AbortController().signal,
        ctx: { env: async (name) => process.env[name], fileExists: async () => false },
      }),
    ).resolves.toEqual({ type: "api_key", source: "gcloud ADC" });
  });

  it("automatically discovers models and MCP tools when a session starts", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const longServerName = "corporate_router_catalog_and_governance_service_with_a_long_name";
    const longToolName = "search_every_authorized_enterprise_catalog_for_relevant_records";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url === "https://litellm.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: longToolName,
              description: "Search authorized enterprise records",
              inputSchema: { type: "object", properties: { query: { type: "string" } } },
              mcp_info: { server_name: longServerName, server_id: "enterprise-catalog" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const refresh = vi.fn().mockResolvedValue({ errors: new Map(), aborted: false });
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "startup-token", baseUrl: "https://litellm.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
        refresh,
      },
    };

    await startSession(pi, context);
    await vi.waitFor(() => expect(pi.tools.some((tool) => tool.name.startsWith("mcp_"))).toBe(true));

    const registeredName = pi.tools.find((tool) => tool.name.startsWith("mcp_"))?.name;
    expect(registeredName).toBe("mcp_corporate_router_catalog_and_governance_service_wit_99d4fe21");
    expect(registeredName).toHaveLength(64);
    expect(refresh).toHaveBeenCalledWith(
      expect.objectContaining({ allowNetwork: true, providers: ["litellm"], signal: expect.any(AbortSignal) }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "https://litellm.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer startup-token" }),
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("reports normal model refresh errors without exposing credentials", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { tools: [] }));
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "startup-secret", baseUrl: "https://litellm.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
        refresh: vi.fn().mockResolvedValue({
          errors: new Map([["litellm", new Error("router unavailable")]]),
          aborted: false,
        }),
      },
    };

    await startSession(pi, context);
    await vi.waitFor(() => expect(stderr).toHaveBeenCalled());

    const output = stderr.mock.calls.map(([message]) => String(message)).join("");
    expect(output).toContain("automatic model discovery failed (router unavailable)");
    expect(output).not.toContain("startup-secret");
  });

  it("suppresses aborted model refresh results", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { tools: [] }));
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "startup-token", baseUrl: "https://litellm.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
        refresh: vi.fn().mockResolvedValue({
          errors: new Map([["litellm", new Error("cancelled")]]),
          aborted: true,
        }),
      },
    };

    await startSession(pi, context);
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(stderr).not.toHaveBeenCalled();
  });

  it("waits for automatic MCP discovery before the first agent turn", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    let releaseCatalog!: (response: Response) => void;
    const catalog = new Promise<Response>((resolve) => {
      releaseCatalog = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/mcp-rest/tools/list")) return catalog;
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const auth = { auth: { apiKey: "startup-token", baseUrl: "https://litellm.example.com/v1" } };
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: {
        getProviderAuth: async () => auth,
        getProvider: () => pi.providers[0],
        refresh: vi.fn().mockResolvedValue({ errors: new Map(), aborted: false }),
      },
    };

    await startSession(pi, context);
    let firstTurnReady = false;
    const firstTurn = pi.handlers
      .get("before_agent_start")?.[0]?.({ systemPrompt: "Base prompt" }, context)
      .then(() => {
        firstTurnReady = true;
      });
    await new Promise<void>((resolve) => setImmediate(resolve));
    expect(firstTurnReady).toBe(false);

    releaseCatalog(
      jsonResponse(200, {
        tools: [
          {
            name: "search",
            inputSchema: { type: "object", properties: {} },
            mcp_info: { server_name: "enterprise" },
          },
        ],
      }),
    );
    await firstTurn;
    expect(pi.tools.map((tool) => tool.name)).toContain("mcp_enterprise_search");
  });

  it("lets the first turn proceed silently after automatic MCP discovery times out", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "5";
    await writeFile(join(agentDir, "settings.json"), JSON.stringify({ litellm: { skills: { enabled: false } } }));
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      return new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
      });
    });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "startup-token", baseUrl: "https://litellm.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
        refresh: vi.fn().mockResolvedValue({ errors: new Map(), aborted: false }),
      },
    };

    await startSession(pi, context);
    await expect(
      pi.handlers.get("before_agent_start")?.[0]?.({ systemPrompt: "Base prompt" }, context),
    ).resolves.toBeUndefined();

    expect(stderr).not.toHaveBeenCalled();
  });

  it("makes MCP tools available even while automatic model refresh is still running", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    let releaseModels!: () => void;
    const modelRefresh = new Promise<void>((resolve) => {
      releaseModels = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/mcp-rest/tools/list")) {
        return jsonResponse(200, {
          tools: [
            {
              name: "search",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "enterprise" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "startup-token", baseUrl: "https://litellm.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
        refresh: vi.fn(async () => {
          await modelRefresh;
          return { errors: new Map(), aborted: false };
        }),
      },
    };

    await startSession(pi, context);
    await expect(
      pi.handlers.get("before_agent_start")?.[0]?.({ systemPrompt: "Base prompt" }, context),
    ).resolves.toBeUndefined();
    expect(pi.tools.map((tool) => tool.name)).toContain("mcp_enterprise_search");

    releaseModels();
  });

  it.each(["1", "true", "TRUE", "yes", "YES"])(
    "honors PI_OFFLINE=%s during automatic discovery",
    async (offlineValue) => {
      const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
      process.env.PI_OFFLINE = offlineValue;
      const fetchMock = vi.spyOn(globalThis, "fetch");
      const extension = await loadExtension(agentDir);
      const pi = createPi();
      await extension(pi);
      const refresh = vi.fn();
      const context = {
        sessionManager: { getSessionFile: () => undefined },
        modelRegistry: {
          getProviderAuth: async () => ({
            auth: { apiKey: "startup-token", baseUrl: "https://litellm.example.com/v1" },
          }),
          refresh,
        },
      };

      await startSession(pi, context);
      await new Promise<void>((resolve) => setImmediate(resolve));

      expect(refresh).not.toHaveBeenCalled();
      expect(fetchMock).not.toHaveBeenCalled();
    },
  );

  it("keeps automatic discovery online with PI_OFFLINE=0", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.PI_OFFLINE = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const refresh = vi.fn().mockResolvedValue({ errors: new Map(), aborted: false });
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: { apiKey: "startup-token", baseUrl: "https://litellm.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
        refresh,
      },
    };

    await startSession(pi, context);
    await pi.handlers.get("before_agent_start")?.[0]?.({ systemPrompt: "Base prompt" }, context);

    expect(refresh).toHaveBeenCalledOnce();
    expect(globalThis.fetch).toHaveBeenCalled();
  });

  it("skips automatic discovery when no LiteLLM auth is configured", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const refresh = vi.fn();
    const context = {
      sessionManager: { getSessionFile: () => undefined },
      modelRegistry: { getProviderAuth: async () => undefined, refresh },
    };

    await startSession(pi, context);
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(refresh).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(pi.tools.some((tool) => tool.name.startsWith("mcp_"))).toBe(false);
  });

  it("registers discovered LiteLLM MCP tools as Pi tools", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url.endsWith("/mcp-rest/tools/list")) {
        return jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search the web",
              inputSchema: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContain("mcp_brave_search"));
  });

  it("refreshes the MCP catalog when default-provider auth changes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://first.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "first",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "first" },
            },
          ],
        });
      }
      if (url === "https://second.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "second",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "second" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://first.example.com";
    process.env.LITELLM_API_KEY = "first-token";
    process.env.LITELLM_HEADERS = '{"x-tenant":"first"}';
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    process.env.LITELLM_BASE_URL = "https://second.example.com";
    process.env.LITELLM_API_KEY = "second-token";
    process.env.LITELLM_HEADERS = '{"x-tenant":"second"}';
    await refreshProvider(pi);

    expect(fetchMock).toHaveBeenCalledWith(
      "https://first.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer first-token", "x-tenant": "first" }),
      }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "https://second.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer second-token", "x-tenant": "second" }),
      }),
    );
    expect(pi.tools.map((tool) => tool.name)).toContain("mcp_second_second");
  });

  it("shares in-flight MCP discovery between default-provider refreshes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    let resolveCatalog: (response: Response) => void = () => {};
    const catalog = new Promise<Response>((resolve) => {
      resolveCatalog = resolve;
    });
    let resolveSecondModel: (response: Response) => void = () => {};
    const secondModel = new Promise<Response>((resolve) => {
      resolveSecondModel = resolve;
    });
    let modelInfoRequests = 0;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        modelInfoRequests++;
        return modelInfoRequests === 1 ? jsonResponse(200, { data: [] }) : secondModel;
      }
      if (url === "https://litellm.example.com/mcp-rest/tools/list") return catalog;
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const firstRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://litellm.example.com/mcp-rest/tools/list", expect.anything()),
    );
    const secondRefresh = refreshProvider(pi);
    await vi.waitFor(() => expect(modelInfoRequests).toBe(2));
    resolveSecondModel(jsonResponse(200, { data: [] }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(
      fetchMock.mock.calls.filter(([url]) => url === "https://litellm.example.com/mcp-rest/tools/list"),
    ).toHaveLength(1);
    resolveCatalog(jsonResponse(200, []));
    await Promise.all([firstRefresh, secondRefresh]);
  });

  it("lets a live different-identity refresh continue after the active refresh aborts", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const firstAbort = new AbortController();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://first.example.com/mcp-rest/tools/list") {
        return new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal;
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        });
      }
      if (url === "https://second.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "second",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "second" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://first.example.com";
    process.env.LITELLM_API_KEY = "first-token";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const firstRefresh = refreshProvider(pi, true, firstAbort.signal);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://first.example.com/mcp-rest/tools/list", expect.anything()),
    );

    process.env.LITELLM_BASE_URL = "https://second.example.com";
    process.env.LITELLM_API_KEY = "second-token";
    const secondRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://second.example.com/model/info", expect.anything()),
    );
    firstAbort.abort(new Error("first refresh cancelled"));

    await expect(firstRefresh).rejects.toThrow("first refresh cancelled");
    await expect(secondRefresh).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("https://second.example.com/mcp-rest/tools/list", expect.anything());
    expect(pi.tools.map((tool) => tool.name)).toContain("mcp_second_second");
  });

  it("rejects an aborted same-identity MCP waiter with its own reason", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const secondAbort = new AbortController();
    let resolveCatalog: (response: Response) => void = () => {};
    const catalog = new Promise<Response>((resolve) => {
      resolveCatalog = resolve;
    });
    let modelInfoRequests = 0;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        modelInfoRequests++;
        return jsonResponse(200, { data: [] });
      }
      if (url === "https://litellm.example.com/mcp-rest/tools/list") return catalog;
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const firstRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://litellm.example.com/mcp-rest/tools/list", expect.anything()),
    );
    const secondRefresh = refreshProvider(pi, true, secondAbort.signal);
    await vi.waitFor(() => expect(modelInfoRequests).toBe(2));

    const reason = new Error("second refresh cancelled");
    secondAbort.abort(reason);
    try {
      const outcome = await Promise.race([
        secondRefresh.then(
          () => "resolved",
          (error: unknown) => error,
        ),
        new Promise((resolve) => setTimeout(() => resolve("pending"), 50)),
      ]);
      expect(outcome).toBe(reason);
    } finally {
      resolveCatalog(jsonResponse(200, []));
      await firstRefresh;
      await secondRefresh.catch(() => undefined);
    }
  });

  it("uses fresh Pi auth when a discovered MCP tool executes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://cached.example.com";
    process.env.LITELLM_API_KEY = "cached-token";

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://cached.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search the web",
              inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        });
      }
      if (url === "https://fresh.example.com/mcp-rest/tools/call") return jsonResponse(200, { result: "fresh" });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);
    const tool = pi.tools.find((candidate) => candidate.name === "mcp_brave_search");

    await tool?.execute?.("call-1", { query: "Pi" }, undefined, undefined, {
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: {
            apiKey: "fresh-token",
            baseUrl: "https://fresh.example.com/v1",
            headers: { "x-tenant": "fresh" },
          },
        }),
        getProvider: () => pi.providers[0],
      },
    });

    expect(fetchMock).toHaveBeenLastCalledWith(
      "https://fresh.example.com/mcp-rest/tools/call",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer fresh-token", "x-tenant": "fresh" }),
      }),
    );
  });

  it("uses fresh Pi auth when a registered Skills tool executes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://cached.example.com";
    process.env.LITELLM_API_KEY = "cached-token";

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://cached.example.com/mcp-rest/tools/list") return jsonResponse(200, []);
      if (url.startsWith("https://fresh.example.com/")) return jsonResponse(200, []);
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);
    const tool = pi.tools.find((candidate) => candidate.name === "litellm_skill_list");

    await tool?.execute?.("call-1", {}, undefined, undefined, {
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: {
            apiKey: "fresh-token",
            baseUrl: "https://fresh.example.com/v1",
            headers: { "x-tenant": "fresh" },
          },
        }),
        getProvider: () => pi.providers[0],
      },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "https://fresh.example.com/claude-code/marketplace.json",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer fresh-token", "x-tenant": "fresh" }),
      }),
    );
  });

  it("injects enabled LiteLLM skills into the system prompt", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, []);
      if (url.endsWith("/claude-code/marketplace.json")) return jsonResponse(404, {});
      if (url.endsWith("/v1/skills")) {
        return jsonResponse(200, {
          data: [{ id: "skill-1", name: "terraform", description: "Terraform conventions", enabled: true }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];
    const result = await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => ({
            auth: { apiKey: "sk-test", baseUrl: "https://litellm.example.com/v1" },
          }),
          getProvider: () => pi.providers[0],
        },
      },
    );

    expect(result.systemPrompt).toContain("Base prompt");
    expect(result.systemPrompt).toContain("<litellm_skills>");
    expect(result.systemPrompt).toContain("Terraform conventions");
  });

  it("clears cached Skills auth when Pi reports revoked credentials", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, []));
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];

    await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => ({
            auth: { apiKey: "active-key", baseUrl: "https://active.example.com/v1" },
          }),
          getProvider: () => pi.providers[0],
        },
      },
    );
    vi.mocked(globalThis.fetch).mockClear();
    await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => undefined,
          getProvider: () => pi.providers[0],
        },
      },
    );

    const listTool = pi.tools.find((tool) => tool.name === "litellm_skill_list");
    await expect(listTool?.execute?.("call-1", {}, undefined, undefined, {})).rejects.toThrow(
      "no credentials for litellm",
    );
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("disables LiteLLM skills through settings", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { skills: { enabled: false } } }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const fetchMock = vi.spyOn(globalThis, "fetch");

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.tools.map((tool) => tool.name)).not.toContain("litellm_skill_list");
    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];
    await expect(beforeAgentStart?.({ systemPrompt: "Base prompt" }, {})).resolves.toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("disables LiteLLM MCP discovery through settings", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    await writeFile(join(agentDir, "settings.json"), JSON.stringify({ litellm: { mcp: { enabled: false } } }), "utf8");
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    expect(requestedUrls).toEqual(["https://litellm.example.com/model/info"]);
    expect(pi.tools.map((tool) => tool.name)).toContain("litellm_skill_list");
    expect(pi.tools.some((tool) => tool.name.startsWith("mcp_"))).toBe(false);
  });

  it("registers cost tracking and session grouping handlers", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.handlers.has("before_provider_request")).toBe(true);
    expect(pi.handlers.has("after_provider_response")).toBe(true);
    expect(pi.handlers.has("message_end")).toBe(true);
  });

  it("does not inject LiteLLM session ids into non-LiteLLM provider requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler(
        { reason: "reload" },
        {
          sessionManager: {
            getSessionFile: () => join(agentDir, "2026-05-11T16-00-00-000Z_123e4567-e89b-12d3-a456-426614174000.jsonl"),
          },
          modelRegistry: { getProviderAuth: async () => undefined, refresh: vi.fn() },
        },
      );
    }

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { messages: [] } },
      { model: { provider: "openai-codex", id: "gpt-5.5" } },
    );
    expect(updated).toBeUndefined();
  });

  it("injects LiteLLM session ids into LiteLLM provider requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler(
        { reason: "reload" },
        {
          sessionManager: {
            getSessionFile: () => join(agentDir, "2026-05-11T16-00-00-000Z_123e4567-e89b-12d3-a456-426614174000.jsonl"),
          },
          modelRegistry: { getProviderAuth: async () => undefined, refresh: vi.fn() },
        },
      );
    }

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.({ payload: { messages: [] } }, { model: { provider: "litellm", id: "kimi-k2.6" } });
    expect(updated).toEqual({
      messages: [],
      litellm_session_id: "123e4567-e89b-12d3-a456-426614174000",
    });
  });

  it("does not inject LiteLLM reasoning fields into Kimi requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k2.6",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.({ payload: { messages: [] } }, { model: { provider: "litellm", id: "kimi-k2.6" } });
    expect(updated).toBeUndefined();
  });

  it("strips unsupported reasoning controls from Bedrock Kimi routes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          messages: [],
          reasoning_effort: "high",
          thinking: { type: "enabled" },
        },
      },
      {
        model: {
          provider: "litellm",
          id: "moonshotai.kimi-k2.5",
          api: "openai-completions",
          compat: { stripReasoningControls: true },
        },
      },
    );

    expect(updated).toEqual({ messages: [] });
  });

  it("preserves reasoning controls when a Moonshot alias is not Bedrock-backed", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { messages: [], thinking: { type: "enabled" } } },
      { model: { provider: "litellm", id: "moonshotai.kimi-k2.5", api: "openai-completions" } },
    );

    expect(updated).toBeUndefined();
  });

  it("leaves Kimi Responses requests unchanged", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { input: [{ type: "message", role: "user", content: "hi" }] } },
      { model: { provider: "litellm", id: "kimi-k2.6", api: "openai-responses" } },
    );

    expect(updated).toBeUndefined();
  });

  it("normalizes Kimi think tags into Pi thinking blocks", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k2.6",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    let message: any = {
      role: "assistant",
      provider: "litellm",
      model: "kimi-k2.6",
      content: [{ type: "text", text: "<think>internal reasoning</think>DONE" }],
      usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
    };
    for (const handler of pi.handlers.get("message_end") ?? []) {
      const result = await handler({ message });
      if (result?.message) message = result.message;
    }

    expect(message.content).toEqual([
      { type: "thinking", thinking: "internal reasoning" },
      { type: "text", text: "DONE" },
    ]);
  });

  it("normalizes think tags for catalog-resolved Kimi route aliases", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    let message: any = {
      role: "assistant",
      provider: "litellm",
      model: "custom-route/kimi-k2.6",
      content: [{ type: "text", text: "<think>internal reasoning</think>DONE" }],
      usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
    };
    for (const handler of pi.handlers.get("message_end") ?? []) {
      const result = await handler({ message });
      if (result?.message) message = result.message;
    }

    expect(message.content).toEqual([
      { type: "thinking", thinking: "internal reasoning" },
      { type: "text", text: "DONE" },
    ]);
  });

  it("keeps final Kimi text visible when a dangling think tag prefixes it", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k2.6",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    let message: any = {
      role: "assistant",
      provider: "litellm",
      model: "kimi-k2.6",
      content: [{ type: "text", text: "<think>DONE" }],
      usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
    };
    for (const handler of pi.handlers.get("message_end") ?? []) {
      const result = await handler({ message });
      if (result?.message) message = result.message;
    }

    expect(message.content).toEqual([{ type: "text", text: "DONE" }]);
  });

  it("overrides assistant cost from LiteLLM response metadata", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler(
        { reason: "start" },
        {
          sessionManager: { getSessionFile: () => undefined },
          modelRegistry: { getProviderAuth: async () => undefined, refresh: vi.fn() },
        },
      );
    }

    const responseHandler = pi.handlers.get("after_provider_response")?.[0];
    responseHandler?.(
      { headers: { "x-litellm-response-cost": "0.42" } },
      { model: { provider: "litellm", id: "anthropic/claude-3-5-sonnet" } },
    );

    const endHandler = pi.handlers.get("message_end")?.[0];
    const result = await endHandler?.({
      message: {
        role: "assistant",
        provider: "litellm",
        model: "anthropic/claude-3-5-sonnet",
        usage: { input: 100, output: 50, cacheRead: 10, cacheWrite: 5 },
      },
    });

    expect(result).toMatchObject({
      message: {
        usage: {
          cost: {
            total: 0.42,
          },
        },
      },
    });
  });

  it("does not apply LiteLLM model costs to other providers' messages", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    // Same model id discovered through LiteLLM, but the message came from
    // a direct provider — LiteLLM pricing must not overwrite its cost.
    const endHandler = pi.handlers.get("message_end")?.[0];
    const result = await endHandler?.({
      message: {
        role: "assistant",
        provider: "anthropic",
        model: "anthropic/claude-3-5-sonnet",
        usage: { input: 100, output: 50, cacheRead: 10, cacheWrite: 5 },
      },
    });

    expect(result).toBeUndefined();
  });

  it("ignores LiteLLM cost headers captured from non-LiteLLM responses", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    const responseHandler = pi.handlers.get("after_provider_response")?.[0];
    responseHandler?.(
      { headers: { "x-litellm-response-cost": "0.42" } },
      { model: { provider: "openai-codex", id: "gpt-5.5" } },
    );

    const endHandler = pi.handlers.get("message_end")?.[0];
    const usage = {
      input: 100,
      output: 50,
      cacheRead: 10,
      cacheWrite: 5,
      cost: { input: 0.0003, output: 0.00075, cacheRead: 0.000003, cacheWrite: 0.00001875, total: 0.00107175 },
    };
    const result = await endHandler?.({
      message: {
        role: "assistant",
        provider: "litellm",
        model: "anthropic/claude-3-5-sonnet",
        usage,
      },
    });

    expect(result).toBeUndefined();
    expect(usage.cost.total).toBeCloseTo(0.00107175, 10);
  });
});
