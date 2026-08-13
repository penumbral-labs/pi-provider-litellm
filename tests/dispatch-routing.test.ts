import { type AuthContext, createModels, InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
// Imported by file path, not package specifier: the composer is internal to
// pi-coding-agent and not in its exports map. Reaching for it is deliberate — the
// routing decision under test belongs to Pi, and asserting it against a reimplementation
// would prove nothing. If this import breaks, Pi's composition changed and the routing
// assumptions in this file need re-measuring, which is exactly when we want to know.
import { composeModelProvider } from "../node_modules/@earendil-works/pi-coding-agent/dist/core/provider-composer.js";
import { createLiteLLMProvider } from "../src/provider.js";
import type { DiscoveredModel } from "../src/types.js";

// Exercises the seam this provider does not own: Pi's composer decides whether a
// model is dispatched through us at all. Everything here drives the real
// composeModelProvider and the real createModels, with fetch intercepted, so the
// assertions describe measured routing rather than intended routing.

const ROOT = "https://proxy.example.com";
const FOREIGN = "https://foreign.example.com";
const SECRET = "sk-proxy-secret";

type Wire = { url: string; authorization: string | null; tenant: string | null };

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
    compat: { supportsStore: false },
  };
}

/**
 * Builds the composed provider Pi would build for a user who has both a discovered
 * LiteLLM catalog and hand-written `models.json` entries under `providers.litellm`.
 *
 * `discoveredApis` controls which protocols the base provider currently lists, which
 * is the only signal Pi's `supportsBaseApi` consults.
 */
function harness(options: { configuredModels: ReturnType<typeof model>[]; discoveredApis?: string[] }) {
  const wire: Wire[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const headers = new Headers((init?.headers as Record<string, string>) ?? {});
    wire.push({
      url: String(input),
      authorization: headers.get("authorization"),
      tenant: headers.get("x-tenant"),
    });
    return new Response('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', {
      headers: { "content-type": "text/event-stream" },
    });
  });

  const base = createLiteLLMProvider({
    id: "litellm",
    name: "LiteLLM",
    baseUrl: `${ROOT}/v1`,
    auth: {
      apiKey: {
        name: "API key",
        resolve: async () => ({ auth: { apiKey: SECRET, headers: { "x-tenant": "acme" } } }),
      },
    },
    resolveCredentialRoot: () => ROOT,
    discover: async () => ({ models: [discovered("chat")], source: "model_info" }),
  });
  const listed = (options.discoveredApis ?? ["openai-completions"]).map((api, index) =>
    model(`listed-${index}`, api, `${ROOT}/v1`),
  );
  Object.defineProperty(base, "getModels", { value: () => listed, writable: true });

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
  return { wire, composed, models };
}

describe("dispatch routing through Pi's composer", () => {
  beforeEach(() => {
    for (const key of ["LITELLM_BASE_URL", "LITELLM_API_KEY", "LITELLM_HEADERS", "LITELLM_OFFLINE"]) {
      delete process.env[key];
    }
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("intercepts a foreign-host Chat entry, because Chat is a listed protocol", async () => {
    // Control: this is the protocol discovery produced, so Pi routes through us and the
    // guard runs. No request may leave, and the key must not appear anywhere.
    const entry = model("cfg-chat", "openai-completions", `${FOREIGN}/v1`);
    const { wire, models } = harness({ configuredModels: [entry] });

    const result = await models.complete(entry, { messages: [] });

    expect(result.stopReason).toBe("error");
    expect(wire).toEqual([]);
  });

  it("sends a same-host Chat entry to the credential host", async () => {
    const entry = model("cfg-chat", "openai-completions", `${ROOT}/v1`);
    const { wire, models } = harness({ configuredModels: [entry] });

    const result = await models.complete(entry, { messages: [] });

    expect(result.stopReason).toBe("stop");
    expect(wire.map(({ url }) => url)).toEqual([`${ROOT}/v1/chat/completions`]);
  });

  it("keeps a foreign-host Responses entry off the wire once Responses is a listed protocol", async () => {
    // The exploit precondition is that the protocol has no listed model. When the proxy
    // does expose a responses route, routing reaches us and the guard holds.
    const entry = model("cfg-responses", "openai-responses", `${FOREIGN}/v1`);
    const { wire, models } = harness({
      configuredModels: [entry],
      discoveredApis: ["openai-completions", "openai-responses"],
    });

    const result = await models.complete(entry, { messages: [] });

    expect(result.stopReason).toBe("error");
    expect(wire).toEqual([]);
  });

  // Upstream canary. Pi's `supportsBaseApi` gates delegation on the provider's current
  // model list, and `Provider` exposes no way to declare the protocols a provider
  // implements, so a protocol with zero listed models is dispatched around this
  // provider and its guard never runs. This test pins that measured behavior, including
  // the credential exposure, so it FAILS as soon as Pi routes by declared protocol —
  // at which point the guard covers this case and the assertions below should be
  // inverted to expect interception.
  it("UPSTREAM GAP: dispatches a foreign-host Responses entry around the guard when no Responses model is listed", async () => {
    const entry = model("cfg-responses", "openai-responses", `${FOREIGN}/v1`);
    const { wire, models } = harness({ configuredModels: [entry], discoveredApis: ["openai-completions"] });

    await models.complete(entry, { messages: [] });

    expect(wire.map(({ url }) => url)).toEqual([`${FOREIGN}/v1/responses`]);
    // Documents the exposure precisely: the proxy key and configured headers travel to
    // a host the credential never authorized.
    expect(wire[0]?.authorization).toBe(`Bearer ${SECRET}`);
    expect(wire[0]?.tenant).toBe("acme");
  });

  it("hides every foreign-host entry from the catalog regardless of protocol", async () => {
    // Catalog filtering is ours and is uniform, which is why the gap above is a
    // dispatch-only gap: none of these are offered to the user.
    const entries = [
      model("cfg-chat", "openai-completions", `${FOREIGN}/v1`),
      model("cfg-responses", "openai-responses", `${FOREIGN}/v1`),
      model("cfg-foreign-api", "google-generative-ai", `${FOREIGN}/v1`),
    ];
    const { models } = harness({ configuredModels: entries });

    const available = await models.getAvailable();

    expect(available.map(({ id }) => id)).toEqual(["listed-0"]);
  });
});
