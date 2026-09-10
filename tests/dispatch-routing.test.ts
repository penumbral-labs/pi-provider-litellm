import { type AuthContext, createModels, InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
// The routing decision belongs to Pi's internal provider composer. Importing the
// installed implementation makes this a canary for changes to that seam instead of
// testing a local reimplementation.
import { composeModelProvider } from "../node_modules/@earendil-works/pi-coding-agent/dist/core/provider-composer.js";
import { createLiteLLMProvider } from "../src/provider.js";
import type { DiscoveredModel } from "../src/types.js";

const CREDENTIAL_ROOT = "https://proxy.example.com";
const FOREIGN_ROOT = "https://foreign.example.com";
const CANARY_CREDENTIAL = ["canary", "credential"].join("-");

type WireRequest = { url: string; authorization: string | null; tenant: string | null };

function model(id: string, api: string, baseUrl: string) {
  return {
    id,
    name: id,
    provider: "litellm",
    api,
    baseUrl,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 4096,
    maxTokens: 1024,
  };
}

function discovered(id: string): DiscoveredModel {
  return {
    id,
    name: id,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 4096,
    maxTokens: 1024,
    api: "openai-completions",
  };
}

function harness(options: { configuredModels: ReturnType<typeof model>[]; discoveredApis?: string[] }) {
  const wire: WireRequest[] = [];
  const payloads: unknown[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const headers = new Headers(init?.headers);
    wire.push({
      url: String(input),
      authorization: headers.get("authorization"),
      tenant: headers.get("x-tenant"),
    });
    if (typeof init?.body === "string") payloads.push(JSON.parse(init.body));
    return new Response('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', {
      headers: { "content-type": "text/event-stream" },
    });
  });

  const listed = (options.discoveredApis ?? ["openai-completions"]).map((api, index) =>
    model(`listed-${index}`, api, `${CREDENTIAL_ROOT}/v1`),
  );
  const base = createLiteLLMProvider({
    id: "litellm",
    name: "LiteLLM",
    baseUrl: `${CREDENTIAL_ROOT}/v1`,
    auth: {
      apiKey: {
        name: "API key",
        // Mirrors resolveApiKeyAuth: auth.baseUrl pins the request host to the credential
        // root so Models.applyAuth overrides any model.baseUrl before dispatch.
        resolve: async () => ({
          auth: { apiKey: CANARY_CREDENTIAL, headers: { "x-tenant": "canary-tenant" }, baseUrl: CREDENTIAL_ROOT },
        }),
      },
    },
    models: listed as never,
    resolveCredentialRoot: () => CREDENTIAL_ROOT,
    discover: async () => ({ models: [discovered("chat")], source: "model_info" }),
  });

  const config = {
    getProvider: () => ({ models: options.configuredModels.map(({ id, api, baseUrl }) => ({ id, api, baseUrl })) }),
    getProviderIds: () => ["litellm"],
  };
  const composed = composeModelProvider("litellm", base, config as never, undefined);
  const authContext: AuthContext = { env: async () => undefined, fileExists: async () => false };
  const models = createModels({
    credentials: new InMemoryCredentialStore(),
    modelsStore: new InMemoryModelsStore(),
    authContext,
  });
  models.setProvider(composed);
  return { wire, payloads, models };
}

describe("dispatch routing through Pi's provider composer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  // The security invariant across every routing outcome: a stale or attacker-supplied
  // model.baseUrl never receives the LiteLLM credential. auth.baseUrl (set by
  // resolveApiKeyAuth / the OAuth toAuth wrapper) pins the request host to the credential
  // root before dispatch, so it holds even when Pi bypasses the provider-owned host guard.
  const assertNoForeignCredential = (wire: WireRequest[]) => {
    for (const request of wire) {
      expect(request.url.startsWith(FOREIGN_ROOT)).toBe(false);
      if (request.authorization === `Bearer ${CANARY_CREDENTIAL}` || request.tenant === "canary-tenant") {
        expect(request.url.startsWith(CREDENTIAL_ROOT)).toBe(true);
      }
    }
  };

  it("pins a foreign-host model to the credential root for a protocol in the catalog", async () => {
    const entry = model("configured-chat", "openai-completions", `${FOREIGN_ROOT}/v1`);
    const { wire, models } = harness({ configuredModels: [entry] });

    const result = await models.complete(entry, { messages: [] });

    expect(result.stopReason).toBe("stop");
    expect(wire.map((request) => request.url)).toEqual([`${CREDENTIAL_ROOT}/v1/chat/completions`]);
    assertNoForeignCredential(wire);
  });

  it("pins a foreign-host Responses model to the credential root when Responses is in the catalog", async () => {
    const entry = model("configured-responses", "openai-responses", `${FOREIGN_ROOT}/v1`);
    const { wire, models } = harness({
      configuredModels: [entry],
      discoveredApis: ["openai-completions", "openai-responses"],
    });

    // The mock only speaks Chat Completions, so the Responses parse fails after the request;
    // the routed URL is what matters here.
    await models.complete(entry, { messages: [] });

    expect(wire.map((request) => request.url)).toEqual([`${CREDENTIAL_ROOT}/v1/responses`]);
    assertNoForeignCredential(wire);
  });

  it("keeps the credential on the credential root when Pi's generic fallback bypasses the guard", async () => {
    const entry = model("configured-responses", "openai-responses", `${FOREIGN_ROOT}/v1`);
    const { wire, models } = harness({ configuredModels: [entry], discoveredApis: ["openai-completions"] });

    await models.complete(entry, { messages: [] });

    // Responses is absent from the catalog, so Pi 0.84 routes through its global Responses
    // implementation instead of the provider-owned guard. auth.baseUrl still pins the host,
    // so the credential reaches the LiteLLM proxy, never the model's stale baseUrl.
    expect(wire).toEqual([
      {
        url: `${CREDENTIAL_ROOT}/responses`,
        authorization: `Bearer ${CANARY_CREDENTIAL}`,
        tenant: "canary-tenant",
      },
    ]);
  });

  it("does not leak the credential to a foreign host for an unsupported configured API", async () => {
    const entry = model("configured-google", "google-generative-ai", FOREIGN_ROOT);
    const { wire, models } = harness({ configuredModels: [entry], discoveredApis: ["openai-completions"] });

    const result = await models.complete(entry, { messages: [] });

    // Generic dispatch for a non-LiteLLM API bypasses the provider guard; the host pin still
    // keeps the request off the model's foreign baseUrl.
    expect(result.stopReason).toBe("error");
    assertNoForeignCredential(wire);
  });

  it("applies cacheControlFormat only to Chat Completions payloads", async () => {
    // Deliberately escape the Responses compat type to exercise the same completions-only flag on both transports.
    const compat = { cacheControlFormat: "anthropic" } as never;
    const completions = {
      ...model("anthropic/claude-sonnet-4-6-completions", "openai-completions", `${CREDENTIAL_ROOT}/v1`),
      compat,
    };
    const responses = {
      ...model("anthropic/claude-sonnet-4-6-responses", "openai-responses", `${CREDENTIAL_ROOT}/v1`),
      compat,
    };
    const { payloads, models } = harness({
      configuredModels: [completions, responses],
      discoveredApis: ["openai-completions", "openai-responses"],
    });
    const context = {
      systemPrompt: "system",
      messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "hello" }], timestamp: 1 }],
    };

    await models.complete(completions, context, { sessionId: "cache-session", cacheRetention: "short" });
    await models.complete(responses, context, { sessionId: "cache-session", cacheRetention: "short" });

    expect(payloads).toHaveLength(2);
    expect(payloads[0]).toMatchObject({
      model: "anthropic/claude-sonnet-4-6-completions",
      messages: expect.any(Array),
    });
    expect(JSON.stringify(payloads[0])).toContain("cache_control");
    expect(payloads[1]).toMatchObject({
      model: "anthropic/claude-sonnet-4-6-responses",
      prompt_cache_key: "cache-session",
      input: expect.any(Array),
    });
    expect(JSON.stringify(payloads[1])).not.toContain("cache_control");
  });
});
