import { readFile } from "node:fs/promises";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import { writeJsonAtomic } from "./cache.js";
import {
  type CatalogResolution,
  catalogResolution,
  closeSerializerPolicy,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  type FamilyEvidence,
  isResponsesMode,
  meetVendorCompat,
  reduceModelGroup,
  type SemanticFamily,
  type SemanticModel,
  wireString,
} from "./model-groups.js";
import type {
  DiscoveredModel,
  DiscoveryOptions,
  DiscoveryResult,
  HealthResponse,
  LiteLLMModelPolicy,
  ModelInfoEntry,
  ModelInfoResponse,
  ModelsListEntry,
  ModelsListResponse,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5000;
const KNOWN_PROVIDER_SET = new Set<string>(getProviders());
const MODELS_DEV_URL = "https://models.dev/api.json";
const MODELS_DEV_CACHE_TTL_MS = 28 * 24 * 60 * 60 * 1000;

interface ModelsDevModel {
  name?: string;
  reasoning?: boolean;
  modalities?: {
    input?: string[];
  };
  limit?: {
    context?: number;
    input?: number;
    output?: number;
  };
  cost?: {
    input?: number;
    output?: number;
    cache_read?: number;
    cache_write?: number;
  };
}

type ModelsDevResponse = Record<string, { models?: Record<string, ModelsDevModel> }>;

interface ModelsDevCacheFile {
  fetchedAt: number;
  catalog: ModelsDevResponse;
}

const modelsDevCaches = new Map<string, ModelsDevCacheFile>();
const modelsDevRefreshes = new Map<string, Promise<ModelsDevResponse | undefined>>();

export function normalizeBaseUrl(input: string): string {
  return input.replace(/\/+$/, "").replace(/\/v1\/?$/i, "");
}

// Matches both the conventional `anthropic/...` prefix and aliases that
// LiteLLM deployments commonly assign to Anthropic-backed routes (e.g.
// `google/claude-sonnet-4-6`, `opus-4.7`, `sonnet-4.6`, `haiku-4.5`). Without
// the `cacheControlFormat: "anthropic"` flag, pi never relays cache_control
// markers through the proxy, so prompt caching silently no-ops on Claude models.
const ANTHROPIC_MODEL_PATTERN = /(?:^|[-_/.:])(?:anthropic\/|(?:claude|opus|sonnet|haiku)(?=$|[-_/.:]))/i;
const MOONSHOT_MODEL_PATTERN = /^(moonshotai\/|moonshot\/|kimi[-/])/i;
const FORCED_THINKING_MODEL_PATTERN = /(?:^|[-/])thinking(?:[-/]|$)/i;
// Deployments expose the gpt-5.5 route under varying names (`llm-gateway/gpt-5.5`,
// bare `gpt-5.5`, dated ids like `gpt-5.5-20260504143601`); match them all so the
// tool+reasoning workaround survives route renames.
const GPT55_MODEL_PATTERN = /(?:^|\/)gpt-5\.5(?:$|[-.])/i;

export function isMoonshotModel(modelId: string): boolean {
  return MOONSHOT_MODEL_PATTERN.test(modelId);
}

export function isGpt55Model(modelId: string): boolean {
  return GPT55_MODEL_PATTERN.test(modelId);
}

// The Moonshot request/display conclusions travel together on the model so the
// request and `message_end` hooks read the same discovered evidence instead of
// each re-deriving a backend from the route name.
// The two repairs are decided separately because they are not equally safe.
//
// `normalizeThinkTags` runs in `message_end`, after the response, and provably
// cannot alter any outbound field, so it may apply on any Moonshot evidence.
//
// `normalizeStrictToolMessages` rewrites outbound assistant and tool messages
// (`content: null` becomes `""`, tool-result arrays become plain strings). Those
// are substitutions, not removals, and compatibility with a candidate that has
// not been identified is unproven — so discovered models enable it only with
// unanimous deployment-family evidence. Route-name fallbacks may still select
// response normalization, compatibility, or limits, but never this repair.
export function moonshotPolicy(modelId: string, strictToolRepair = false): LiteLLMModelPolicy {
  return {
    normalizeStrictToolMessages: strictToolRepair,
    // A route that always reasons streams reasoning content as its own field
    // rather than inlining `<think>` in the answer, so unwrapping is limited to
    // the generations that inline it.
    normalizeThinkTags: !FORCED_THINKING_MODEL_PATTERN.test(modelId),
  };
}

// Ported from pr/route-group-authority (7d6e39b): api-aware compat belongs to
// the multi-protocol core, which owns the per-API compat union. Returning an
// empty object for Responses dropped `supportsDeveloperRole: false` for
// Moonshot-backed routes (pi-ai reads it in openai-responses-shared.js) and left
// the level map with a compat that appears to carry effort, i.e. fail-open.
export function buildCompat(modelId: string, semantic?: FamilyEvidence): DiscoveredModel["compat"] {
  // Contradictory deployment evidence is not an invitation to guess from the
  // route name; such a group only gets vendor-neutral compatibility.
  if (semantic === "conflicting") return { supportsStore: false };
  if (semantic === "kimi" || (semantic === undefined && isMoonshotModel(modelId))) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    };
  }
  if (semantic === "claude" || (semantic === undefined && ANTHROPIC_MODEL_PATTERN.test(modelId))) {
    return { supportsStore: false, cacheControlFormat: "anthropic" };
  }
  return { supportsStore: false };
}

function toKnownProvider(provider: string | undefined): BuiltinProvider | undefined {
  if (!provider) return undefined;
  const normalized = provider.trim().toLowerCase();
  return KNOWN_PROVIDER_SET.has(normalized) ? (normalized as BuiltinProvider) : undefined;
}

// Anthropic recognition is derived from the single `catalogLookupIds` rule so a
// second alias pattern cannot drift away from it. Every Anthropic catalog id and
// every alias that maps onto one is canonicalized to a `claude-` lookup id,
// including single-number names and dated snapshots.
function catalogProviderCandidates(lookupIds: readonly string[], id: string, ownedBy?: string): BuiltinProvider[] {
  const candidates = [toKnownProvider(ownedBy), toKnownProvider(id.split("/")[0])];
  if (lookupIds.some((lookupId) => lookupId.startsWith("claude-"))) candidates.push("anthropic");
  return [...new Set(candidates.filter((provider): provider is BuiltinProvider => provider !== undefined))];
}

function resolveCatalogModel(
  id: string,
  ownedBy?: string,
): { provider: BuiltinProvider; model: Model<Api> } | undefined {
  const lookupIds = catalogLookupIds(id);
  for (const provider of catalogProviderCandidates(lookupIds, id, ownedBy)) {
    const model = findCatalogModelInProvider(provider, lookupIds);
    if (model) return { provider, model };
  }
  return undefined;
}

function findCatalogModel(id: string, ownedBy?: string): Model<Api> | undefined {
  return resolveCatalogModel(id, ownedBy)?.model;
}

// The Moonshot strict-schema compat block is the only one that pins
// `maxTokensField`, so its presence identifies a model that discovery already
// judged to be Moonshot-backed. Releases before request policies existed wrote
// that same block, which makes it usable provenance.
function hasMoonshotCompatEvidence(compat: Model<Api>["compat"]): boolean {
  const openAICompat = compat as Model<"openai-completions">["compat"];
  return openAICompat?.maxTokensField === "max_tokens" && openAICompat.supportsStrictMode === false;
}

// A cached model written before request policies existed carries none. Its
// stored compatibility fingerprint can recover the response-only conclusion,
// but it cannot prove that every deployment identified Moonshot, so the
// outbound repair stays disabled until fresh discovery persists that authority.
function restoreCachedModelPolicy(model: Model<Api>): Model<Api> {
  const cached = model as Model<Api> & { litellmPolicy?: LiteLLMModelPolicy };
  if (cached.litellmPolicy || !hasMoonshotCompatEvidence(model.compat)) return model;
  const restored: typeof cached = { ...cached, litellmPolicy: moonshotPolicy(model.id) };
  return restored;
}

export function enrichCachedModel(input: Model<Api>): Model<Api> {
  const restored = restoreCachedModelPolicy(input);
  // A model stored by a release that predates the transmissibility gate carries
  // whatever level map that release published, so the gate applies to the cached
  // map on the way in — not only to catalog metadata on the way out, which every
  // reasoning model skips via the guard below.
  const model = {
    ...restored,
    ...closeSerializerPolicy({
      api: restored.api === "openai-responses" ? "openai-responses" : "openai-completions",
      reasoning: restored.reasoning,
      vendorCompat: restored.compat,
      catalogLevels: restored.thinkingLevelMap,
    }),
  } as Model<Api>;
  // Reduced deployment groups use a distinct marker; this sentinel remains
  // exclusive to evidence-free fallback models that may be enriched safely.
  if (
    !model.name.endsWith(" (no metadata)") ||
    model.reasoning ||
    model.thinkingLevelMap !== undefined ||
    model.input.length !== 1 ||
    model.input[0] !== "text" ||
    model.cost.input !== 0 ||
    model.cost.output !== 0 ||
    model.cost.cacheRead !== 0 ||
    model.cost.cacheWrite !== 0 ||
    model.cost.tiers !== undefined ||
    model.contextWindow !== DEFAULT_CONTEXT_WINDOW ||
    model.maxTokens !== DEFAULT_MAX_TOKENS
  ) {
    return model;
  }
  const catalogModel = findCatalogModel(model.id);
  if (!catalogModel) return model;
  return {
    ...model,
    name: catalogModel.name,
    // The cached compat stays as stored, so catalog levels are closed against it
    // rather than trusted because the catalog offered them.
    ...closeSerializerPolicy({
      api: model.api === "openai-responses" ? "openai-responses" : "openai-completions",
      reasoning: catalogModel.reasoning,
      vendorCompat: model.compat,
      catalogLevels: catalogModel.thinkingLevelMap,
    }),
    input: catalogModel.input,
    cost: catalogModel.cost,
    contextWindow: catalogModel.contextWindow,
    maxTokens: catalogModel.maxTokens,
  };
}

function catalogLookupIds(id: string): string[] {
  const lookupIds = new Set([id]);
  const unprefixed = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  lookupIds.add(unprefixed);

  const anthropicAlias = unprefixed.toLowerCase().replaceAll(".", "-");
  const match = /^(?:claude-)?(opus|sonnet|haiku)-(\d+)-(\d+)$/.exec(anthropicAlias);
  if (match) lookupIds.add(`claude-${match[1]}-${match[2]}-${match[3]}`);
  if (anthropicAlias === "fable-5" || anthropicAlias === "opus-5") lookupIds.add(`claude-${anthropicAlias}`);

  return [...lookupIds];
}

function findCatalogModelInProvider(provider: BuiltinProvider, lookupIds: string[]): Model<Api> | undefined {
  for (const lookupId of lookupIds) {
    const exact = getModels(provider).find((model) => model.id === lookupId);
    if (exact) return exact;
    const providerQualified = getModels(provider).find((model) => model.id === `${provider}/${lookupId}`);
    if (providerQualified) return providerQualified;
  }
  return undefined;
}

function semanticModel(id: string): SemanticModel | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])kimi[-_/]?k?2[._-]?[56](?:$|[./_:-])/.test(value)) return "kimi-k2.5-k2.6";
  if (/(?:^|[./_-])kimi[-_/]?k?2[._-]?7(?:[-_/]?(?:code|highspeed))?(?:$|[./_:-])/.test(value)) {
    return "kimi-k2.7-code";
  }
  if (/(?:^|[./_-])kimi[-_/]?k?3(?:$|[./_:-])/.test(value)) return "kimi-k3";
  if (/(?:^|[./_-])deepseek[-_/]?v?4(?:$|[./_:-])/.test(value)) return "deepseek-v4";
  return undefined;
}

function semanticFamily(id: string): SemanticFamily | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])(?:anthropic|claude|opus|sonnet|haiku)(?:$|[./_:-])/.test(value)) return "claude";
  if (/(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])/.test(value)) return "kimi";
  if (/(?:^|[./_-])deepseek(?:$|[./_:-])/.test(value)) return "deepseek";
  if (/(?:^|[./_-])gemini(?:$|[./_:-])/.test(value)) return "gemini";
  if (/(?:^|[./_-])(?:openai|gpt|o\d)(?:$|[./_:-])/.test(value)) return "openai";
  return undefined;
}

const ADAPTER_CATALOG_PROVIDERS: Readonly<Record<string, BuiltinProvider>> = {
  anthropic: "anthropic",
  azure: "azure-openai-responses",
  azure_ai: "azure-openai-responses",
  bedrock: "amazon-bedrock",
  bedrock_converse: "amazon-bedrock",
  deepseek: "deepseek",
  fireworks_ai: "fireworks",
  gemini: "google",
  moonshot: "moonshotai",
  nvidia_nim: "nvidia",
  openai: "openai",
  together_ai: "together",
  vertex_ai: "google-vertex",
};

function adapterCatalogProvider(adapter: unknown): BuiltinProvider | undefined {
  const normalized = wireString(adapter)?.trim().toLowerCase();
  return normalized ? (ADAPTER_CATALOG_PROVIDERS[normalized] ?? toKnownProvider(normalized)) : undefined;
}

export function resolveModelInfoCatalog(entry: ModelInfoEntry): CatalogResolution | undefined {
  const adapterProvider = adapterCatalogProvider(entry.model_info?.litellm_provider);
  const routingModel = wireString(entry.litellm_params?.model)?.trim() || undefined;
  const baseModel = wireString(entry.model_info?.base_model)?.trim() || undefined;
  const routingFamily = routingModel ? semanticFamily(routingModel) : undefined;
  const baseFamily = baseModel ? semanticFamily(baseModel) : undefined;
  // A routing model and a declared base model that name different families
  // describe different backends. Neither one is trustworthy metadata authority
  // for this deployment, so report the contradiction and resolve nothing.
  if (routingFamily !== undefined && baseFamily !== undefined && routingFamily !== baseFamily) {
    return { semanticFamily: "conflicting" };
  }
  // Two generations of one family are as contradictory as two families: applying
  // the routing model's contract to a declared K3 backend would send the wrong
  // control. Withhold the generation policy instead of taking the first.
  const routingGeneration = routingModel ? semanticModel(routingModel) : undefined;
  const baseGeneration = baseModel ? semanticModel(baseModel) : undefined;
  const contradictoryGenerations =
    routingGeneration !== undefined && baseGeneration !== undefined && routingGeneration !== baseGeneration;
  const model = contradictoryGenerations ? undefined : (routingGeneration ?? baseGeneration);
  const candidates = [routingModel, baseModel].filter((candidate): candidate is string => candidate !== undefined);
  // Provider identity and semantic family must describe the same backend, so a
  // resolution reports the family of the candidate that resolved it (or of the
  // catalog model itself) rather than a family borrowed from a sibling candidate.
  let unresolvedFamily: SemanticFamily | undefined;
  for (const candidate of candidates) {
    const family = semanticFamily(candidate);
    const resolved = resolveCatalogModel(candidate, adapterProvider);
    if (resolved) {
      return {
        ...catalogResolution(resolved.provider, family ?? semanticFamily(resolved.model.id), resolved.model),
        ...(model ? { semanticModel: model } : {}),
      };
    }
    unresolvedFamily ??= family;
  }
  return unresolvedFamily
    ? { semanticFamily: unresolvedFamily, ...(model ? { semanticModel: model } : {}) }
    : undefined;
}

function getFallbackProviderAndModel(id: string, ownedBy?: string): { provider?: string; modelId: string } {
  const [prefix, ...rest] = id.split("/");
  const prefixProvider = toKnownProvider(prefix);
  if (prefixProvider && rest.length > 0) {
    return { provider: prefixProvider, modelId: rest.join("/") };
  }
  return { provider: toKnownProvider(ownedBy), modelId: id };
}

function findModelsDevModel(
  catalog: ModelsDevResponse | undefined,
  id: string,
  ownedBy?: string,
): ModelsDevModel | undefined {
  const { provider, modelId } = getFallbackProviderAndModel(id, ownedBy);
  if (!provider) return undefined;
  const models = catalog?.[provider]?.models;
  return models && Object.hasOwn(models, modelId) ? models[modelId] : undefined;
}

function awaitWithSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise<T>((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      reject(signal.reason);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
  });
}
async function fetchJson<T>(
  url: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<{ ok: true; data: T } | { ok: false; status: number }> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const response = await fetch(url, {
    headers: { ...options.headers, Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
    signal: options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
      : AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) return { ok: false, status: response.status };
  const data = (await response.json()) as T;
  return { ok: true, data };
}

async function fetchPublicJson<T>(url: string, options: DiscoveryOptions): Promise<T> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    signal: options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
      : AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return (await response.json()) as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeModelsDevCatalog(value: unknown): ModelsDevResponse | undefined {
  if (!isRecord(value)) return undefined;
  const catalog: ModelsDevResponse = {};
  for (const [providerId, providerValue] of Object.entries(value)) {
    if (!isRecord(providerValue) || !isRecord(providerValue.models)) continue;
    const models: Record<string, ModelsDevModel> = {};
    for (const [modelId, modelValue] of Object.entries(providerValue.models)) {
      if (!isRecord(modelValue)) continue;
      const model: ModelsDevModel = {};
      if (typeof modelValue.name === "string") model.name = modelValue.name;
      if (typeof modelValue.reasoning === "boolean") model.reasoning = modelValue.reasoning;
      if (isRecord(modelValue.modalities) && Array.isArray(modelValue.modalities.input)) {
        const input = modelValue.modalities.input.filter((entry): entry is string => typeof entry === "string");
        if (input.length > 0) model.modalities = { input };
      }
      if (isRecord(modelValue.limit)) {
        const limit: NonNullable<ModelsDevModel["limit"]> = {};
        const context = finiteNumber(modelValue.limit.context);
        const input = finiteNumber(modelValue.limit.input);
        const output = finiteNumber(modelValue.limit.output);
        if (context !== undefined) limit.context = context;
        if (input !== undefined) limit.input = input;
        if (output !== undefined) limit.output = output;
        if (Object.keys(limit).length > 0) model.limit = limit;
      }
      if (isRecord(modelValue.cost)) {
        const cost: NonNullable<ModelsDevModel["cost"]> = {};
        const input = finiteNumber(modelValue.cost.input);
        const output = finiteNumber(modelValue.cost.output);
        const cacheRead = finiteNumber(modelValue.cost.cache_read);
        const cacheWrite = finiteNumber(modelValue.cost.cache_write);
        if (input !== undefined) cost.input = input;
        if (output !== undefined) cost.output = output;
        if (cacheRead !== undefined) cost.cache_read = cacheRead;
        if (cacheWrite !== undefined) cost.cache_write = cacheWrite;
        if (Object.keys(cost).length > 0) model.cost = cost;
      }
      models[modelId] = model;
    }
    catalog[providerId] = { models };
  }
  return catalog;
}

async function readModelsDevCache(path: string): Promise<ModelsDevCacheFile | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
    if (
      !isRecord(parsed) ||
      typeof parsed.fetchedAt !== "number" ||
      !Number.isFinite(parsed.fetchedAt) ||
      parsed.fetchedAt < 0 ||
      parsed.fetchedAt > Date.now()
    ) {
      return undefined;
    }
    const catalog = normalizeModelsDevCatalog(parsed.catalog);
    return catalog ? { fetchedAt: parsed.fetchedAt, catalog } : undefined;
  } catch {
    return undefined;
  }
}

function refreshModelsDevCatalog(key: string, options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  const active = modelsDevRefreshes.get(key);
  if (active) return active;
  const refresh = (async () => {
    try {
      const catalog = normalizeModelsDevCatalog(await fetchPublicJson<unknown>(MODELS_DEV_URL, options));
      if (!catalog) return undefined;
      const cache = { fetchedAt: Date.now(), catalog };
      modelsDevCaches.set(key, cache);
      if (options.modelsDevCachePath) {
        await writeJsonAtomic(options.modelsDevCachePath, cache).catch(() => undefined);
      }
      return catalog;
    } catch {
      return undefined;
    } finally {
      modelsDevRefreshes.delete(key);
    }
  })();
  modelsDevRefreshes.set(key, refresh);
  return refresh;
}

async function getModelsDevCatalog(options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  const key = options.modelsDevCachePath ?? MODELS_DEV_URL;
  const refreshOptions = { ...options, timeoutMs: DEFAULT_TIMEOUT_MS, signal: undefined };
  let cache = modelsDevCaches.get(key);
  if (!cache && options.modelsDevCachePath) {
    cache = await readModelsDevCache(options.modelsDevCachePath);
    if (cache) modelsDevCaches.set(key, cache);
  }
  if (!cache) {
    const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    return awaitWithSignal(
      refreshModelsDevCatalog(key, refreshOptions),
      options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
    );
  }
  if (Date.now() - cache.fetchedAt < MODELS_DEV_CACHE_TTL_MS) return cache.catalog;
  void refreshModelsDevCatalog(key, refreshOptions);
  return cache.catalog;
}

function mapModelsDevMetadata(model: ModelsDevModel | undefined): Partial<DiscoveredModel> {
  if (!model) return {};
  const metadata: Partial<DiscoveredModel> = {};
  if (model.name) metadata.name = model.name;
  if (model.reasoning !== undefined) metadata.reasoning = model.reasoning;
  if (model.modalities?.input) {
    metadata.input = model.modalities.input.includes("image") ? ["text", "image"] : ["text"];
  }
  const contextWindow = model.limit?.context ?? model.limit?.input;
  if (contextWindow !== undefined) metadata.contextWindow = contextWindow;
  if (model.limit?.output !== undefined) metadata.maxTokens = model.limit.output;
  if (model.cost) {
    metadata.cost = {
      input: model.cost.input ?? 0,
      output: model.cost.output ?? 0,
      cacheRead: model.cost.cache_read ?? 0,
      cacheWrite: model.cost.cache_write ?? 0,
    };
  }
  return metadata;
}

const AMBIGUOUS_AUTHORITY_SAMPLE = 3;

// Keyed by route so persistent ambiguity is reported once per process while a
// newly observed route still receives a bounded diagnostic.
const reportedAmbiguousRoutes = new Set<string>();

function reportAmbiguousCatalogAuthority(routes: readonly string[]): void {
  const unreported = routes.filter((route) => !reportedAmbiguousRoutes.has(route));
  if (unreported.length === 0) return;
  for (const route of unreported) reportedAmbiguousRoutes.add(route);
  const hidden = unreported.length - AMBIGUOUS_AUTHORITY_SAMPLE;
  const sample = unreported.slice(0, AMBIGUOUS_AUTHORITY_SAMPLE).join(", ");
  process.stderr.write(
    `LiteLLM discovery: ${unreported.length} route group(s) have missing or conflicting deployment provider ` +
      `evidence; catalog limits, pricing, and reasoning metadata are withheld: ${sample}` +
      `${hidden > 0 ? ` (+${hidden} more)` : ""}\n`,
  );
}

// Withholding a repair a Moonshot deployment demonstrably needs is the safe
// choice for its unidentified sibling, but it is not a free one: tool calls to
// the Moonshot deployment can fail. Say so rather than letting the group look
// fully supported. As with catalog-authority diagnostics, persistent routes are
// reported once per process while newly observed routes still get one bounded
// line. Deduplicate the input too, since fallback expansion may repeat a route.
const reportedWithheldRepairRoutes = new Set<string>();

function reportWithheldToolRepair(routes: readonly string[]): void {
  const unreported = [...new Set(routes)].filter((route) => !reportedWithheldRepairRoutes.has(route));
  if (unreported.length === 0) return;
  for (const route of unreported) reportedWithheldRepairRoutes.add(route);
  const hidden = unreported.length - AMBIGUOUS_AUTHORITY_SAMPLE;
  const sample = unreported.slice(0, AMBIGUOUS_AUTHORITY_SAMPLE).join(", ");
  process.stderr.write(
    `LiteLLM discovery: ${unreported.length} route group(s) look Moonshot-backed but not every deployment evidences ` +
      `it; strict tool-message repair is withheld because it rewrites outbound messages and is unproven for a ` +
      `deployment that has not identified its backend. Moonshot tool calls on these routes may fail until every ` +
      `deployment declares its backend: ${sample}${hidden > 0 ? ` (+${hidden} more)` : ""}\n`,
  );
}

function reportWithheldToolRepairForModels(models: readonly DiscoveredModel[]): void {
  reportWithheldToolRepair(
    models.filter((model) => model.litellmPolicy?.normalizeStrictToolMessages === false).map((model) => model.id),
  );
}

function mapFromModelInfoGroup(
  entries: readonly ModelInfoEntry[],
  ambiguousRoutes?: string[],
  withheldRepairRoutes?: string[],
): DiscoveredModel | undefined {
  const reduced = reduceModelGroup(entries, (entry, singleton) => {
    const resolved = resolveModelInfoCatalog(entry);
    if (resolved || !singleton) return resolved;
    // Exactly one routable deployment: the public route name is the only
    // remaining hint, and using it preserves upstream singleton behavior.
    const id = entry.model_name;
    const catalog = id ? resolveCatalogModel(id) : undefined;
    return catalog
      ? { ...catalogResolution(catalog.provider, semanticFamily(catalog.model.id), catalog.model), routeNameOnly: true }
      : undefined;
  });
  if (!reduced) return undefined;
  if (reduced.catalogAuthorityAmbiguous) ambiguousRoutes?.push(reduced.id);
  const reasoningPolicy = reduced.semanticModel ? reduced.reasoningPolicy : undefined;
  const reasoning = reasoningPolicy?.reasoning ?? reduced.reasoning;
  // The policy's compat only applies on Chat, so its level map must be gated the
  // same way; otherwise a Responses group advertises Chat-shaped levels that the
  // Responses wire emits as bare `reasoning.effort` values.
  // Vendor compatibility is the per-field conservative meet of what each routable
  // deployment evidences. An unlabeled deployment contributes no conclusion, which
  // blocks every shape-changing field while still letting a deployment-evidenced
  // safety restriction survive. A unanimous group meets to exactly that vendor's
  // block, so the common case is unchanged. Route text is consulted only when no
  // deployment identified a family at all.
  const unlabeled = reduced.deploymentFamilies.every((family) => family === undefined);
  // Every routable deployment evidences Moonshot: the outbound rewrite is needed
  // by all of them, so there is no candidate it could be wrong for.
  const unanimousMoonshot =
    reduced.deploymentFamilies.length > 0 && reduced.deploymentFamilies.every((family) => family === "kimi");
  const moonshotEvidence = reduced.deploymentFamilies.includes("kimi") || (unlabeled && isMoonshotModel(reduced.id));
  // Withheld whenever the need is not unanimous — including a group where no
  // deployment identifies a backend at all, where the route name alone is not
  // explicit need for an outbound rewrite.
  if (moonshotEvidence && !unanimousMoonshot) withheldRepairRoutes?.push(reduced.id);
  const vendorCompat = unlabeled
    ? buildCompat(reduced.id, reduced.semanticFamily)
    : meetVendorCompat(
        // A deployment that identified no family contributes NO conclusion, not a
        // route-name-derived one: passing `undefined` family to `buildCompat` would
        // infer the vendor from the route id and hand the meet a unanimous vote it
        // never earned, which is exactly the inference this withholds.
        reduced.deploymentFamilies.map((family) =>
          family === undefined ? undefined : buildCompat(reduced.id, family),
        ),
      );
  const policy = closeSerializerPolicy({
    api: reduced.api,
    reasoning,
    vendorCompat,
    semanticCompat: reasoningPolicy?.compat,
    semanticLevels: reasoningPolicy?.thinkingLevelMap,
    catalogLevels: reduced.thinkingLevelMap,
  });
  return {
    id: reduced.id,
    // Reduced groups never borrow the ` (no metadata)` sentinel, which authorizes
    // catalog re-derivation from the model id during offline cache reads.
    name: reduced.hasCompleteCost ? reduced.id : `${reduced.id} (incomplete metadata)`,
    ...policy,
    input: reduced.vision ? ["text", "image"] : ["text"],
    cost: reduced.cost,
    contextWindow: reduced.contextWindow,
    maxTokens: reduced.maxTokens,
    api: reduced.api,
    // The route id, not the semantic label: `moonshotPolicy`'s forced-thinking
    // check is a route-id pattern, and no semantic label can ever match it, so
    // passing a label made the exemption unreachable and split the display
    // conclusion by discovery source for one route.
    // The two repairs are decided separately. `<think>` unwrapping is response-only
    // and cannot change the request, so any Moonshot evidence is enough. Strict
    // tool-message repair rewrites outbound messages, so it needs every routable
    // deployment to evidence the need; a mixed group withholds it and says so.
    ...(moonshotEvidence ? { litellmPolicy: moonshotPolicy(reduced.id, unanimousMoonshot) } : {}),
  };
}

function mapFromModelInfo(entry: ModelInfoEntry): DiscoveredModel | undefined {
  return mapFromModelInfoGroup([entry]);
}

function mapFromHealthModelInfo(entry: ModelInfoEntry, fallbackId: string | undefined): DiscoveredModel | undefined {
  const named = wireString(entry.model_name) || !fallbackId ? entry : { ...entry, model_name: fallbackId };
  // `/health` detail lookups are per deployment, not complete route groups, so
  // they stay on Chat. Rewriting the mode before reduction keeps the deployment
  // on the ordinary Chat path, so vendor and reasoning-replay compatibility are
  // built for it instead of being patched back on afterwards. Modes this
  // discovery cannot serve at all are left alone so they still drop out.
  const chatOnly = isResponsesMode(named.model_info?.mode)
    ? { ...named, model_info: { ...named.model_info, mode: "chat" } }
    : named;
  const model = mapFromModelInfo(chatOnly);
  if (!model) return undefined;
  // Without a deployment `model_name` the only identifier is the `/health` route
  // text, which is not evidence for request controls. Deleting the map would hand
  // pi-ai an absent map, i.e. every standard level, so the conclusion is closed
  // again with no candidate levels rather than mutated in place.
  if (entry.model_name) return model;
  return {
    ...model,
    ...closeSerializerPolicy({
      api: model.api === "openai-responses" ? "openai-responses" : "openai-completions",
      reasoning: model.reasoning,
      vendorCompat: model.compat,
      denyLevels: true,
    }),
  };
}

function mapFromHealthEndpoint(entry: { model?: string }): DiscoveredModel | undefined {
  const id = wireString(entry.model);
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id);
  return {
    id,
    // Health route text is not later cache authority. Unknown routes therefore
    // use the permanent reduced-evidence marker, never the fallback sentinel.
    name: catalogModel?.name ?? `${id} (incomplete metadata)`,
    // Same evidence quality as the `/v1/models` fallback — a route name and
    // nothing else — so it closes the same catalog map the same way.
    ...closeSerializerPolicy({
      api: "openai-completions",
      reasoning: catalogModel?.reasoning ?? false,
      vendorCompat: buildCompat(id),
      catalogLevels: catalogModel?.thinkingLevelMap,
    }),
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: "openai-completions",
    ...(isMoonshotModel(id) ? { litellmPolicy: moonshotPolicy(id) } : {}),
  };
}

function mapFromModelsList(
  entry: ModelsListEntry,
  modelsDev: ModelsDevResponse | undefined,
): DiscoveredModel | undefined {
  const id = wireString(entry.id);
  if (!id) return undefined;
  const ownedBy = wireString(entry.owned_by);
  const catalogModel = findCatalogModel(id, ownedBy);
  const modelsDevMetadata = mapModelsDevMetadata(findModelsDevModel(modelsDev, id, ownedBy));
  return {
    id,
    name: modelsDevMetadata.name ?? catalogModel?.name ?? `${id} (no metadata)`,
    ...closeSerializerPolicy({
      api: "openai-completions",
      reasoning: modelsDevMetadata.reasoning ?? catalogModel?.reasoning ?? false,
      vendorCompat: buildCompat(id),
      catalogLevels: catalogModel?.thinkingLevelMap,
    }),
    input: modelsDevMetadata.input ?? catalogModel?.input ?? ["text"],
    cost: modelsDevMetadata.cost ?? catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: modelsDevMetadata.contextWindow ?? catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: modelsDevMetadata.maxTokens ?? catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: "openai-completions",
    ...(isMoonshotModel(id) ? { litellmPolicy: moonshotPolicy(id) } : {}),
  };
}

async function discoverFromHealth(
  base: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void; silent?: boolean },
): Promise<DiscoveredModel[]> {
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /health endpoint...");
  // KNOWN LIMITATION: `/health` lists deployments, not routes, and each one is
  // mapped on its own, so deployments sharing a `model_name` are not reduced
  // against each other the way `/model/info` groups are. `Promise.all` preserves
  // list order and `deduplicateModels` keeps the first entry, so the survivor is
  // whichever deployment `/health` listed first — not whichever answered first.
  // Reducing here means aggregating the per-deployment detail fetches by
  // `model_name` and feeding whole groups to `mapFromModelInfoGroup`, which also
  // changes naming and the singleton catalog fallback on this path. The approved
  // PRD names that a non-goal; see the `/health` notes in AGENTS.md and README.md.
  const healthResult = await fetchJson<HealthResponse>(`${base}/health`, apiKey, options);
  if (!healthResult.ok) return [];
  const endpoints = (healthResult.data.healthy_endpoints ?? []).filter((entry) => entry.model || entry.model_id);
  progress?.(`Discovered ${endpoints.length} model endpoints, fetching details...`);
  let completed = 0;
  const models = await Promise.all(
    endpoints.map(async (endpoint) => {
      let model = mapFromHealthEndpoint(endpoint);
      if (endpoint.model_id) {
        const infoResult = await fetchJson<ModelInfoResponse>(
          `${base}/model/info?litellm_model_id=${encodeURIComponent(endpoint.model_id)}`,
          apiKey,
          options,
        );
        const entry = infoResult.ok ? infoResult.data.data?.[0] : undefined;
        if (entry) model = mapFromHealthModelInfo(entry, endpoint.model);
      }
      completed++;
      if (completed % 10 === 0 || completed === endpoints.length) {
        progress?.(`Fetched ${completed}/${endpoints.length} models...`);
      }
      return model;
    }),
  );
  const discovered = models.filter((model): model is DiscoveredModel => model !== undefined);
  reportWithheldToolRepairForModels(discovered);
  return discovered;
}

function deduplicateModels(models: DiscoveredModel[]): DiscoveredModel[] {
  const seen = new Set<string>();
  return models.filter((m) => {
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void; silent?: boolean } = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl);
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /model/info endpoint...");
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const groups = new Map<string, ModelInfoEntry[]>();
    for (const entry of infoResult.data.data ?? []) {
      const route = wireString(entry.model_name);
      if (!route) continue;
      const group = groups.get(route) ?? [];
      group.push(entry);
      groups.set(route, group);
    }
    const ambiguousRoutes: string[] = [];
    const withheldRepairRoutes: string[] = [];
    let models = [...groups.values()]
      .map((group) => mapFromModelInfoGroup(group, ambiguousRoutes, withheldRepairRoutes))
      .filter((m): m is DiscoveredModel => m !== undefined);
    reportAmbiguousCatalogAuthority(ambiguousRoutes);
    reportWithheldToolRepair(withheldRepairRoutes);
    // LiteLLM's /model/info does NOT expand wildcard model_name entries (e.g.
    // "lemonade/*" backed by model: openai/* + check_provider_endpoint: true)
    // — it returns the literal wildcard only. The discovered ids live in
    // /v1/models instead. When /model/info contains any wildcard id, also query
    // /v1/models and merge the expanded (non-wildcard) entries in, dropping the
    // raw wildcard row so it doesn't surface as a phantom model choice.
    // Ref: docs.litellm.ai/docs/proxy/model_discovery
    if (models.some((m) => m.id.includes("*"))) {
      progress?.("/model/info has wildcard entries, expanding via /v1/models...");
      let modelsDev: ModelsDevResponse | undefined;
      if (options.modelsDev !== false) {
        progress?.("Loading models.dev catalog for metadata enrichment...");
        modelsDev = await getModelsDevCatalog(options);
      }
      const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
      if (listResult.ok) {
        const expanded = (listResult.data.data ?? [])
          .map((entry) => mapFromModelsList(entry, modelsDev))
          .filter((m): m is DiscoveredModel => m !== undefined && !m.id.includes("*"));
        const seen = new Set<string>(models.map((m) => m.id));
        models = [...models.filter((m) => !m.id.includes("*")), ...expanded.filter((m) => !seen.has(m.id))];
      }
    }
    const deduplicated = deduplicateModels(models);
    reportWithheldToolRepairForModels(deduplicated);
    return { source: "model_info", models: deduplicated };
  }
  if (![401, 403, 404].includes(infoResult.status)) {
    throw new Error(`/model/info returned ${infoResult.status}`);
  }
  progress?.("/model/info unavailable, trying /v1/models...");
  const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
  if (!listResult.ok) {
    if ([401, 403, 404].includes(listResult.status)) {
      progress?.("/v1/models unavailable, falling back to /health endpoint...");
      const models = await discoverFromHealth(base, apiKey, options);
      if (models.length > 0) return { source: "health", models: deduplicateModels(models) };
    }
    throw new Error(`/v1/models returned ${listResult.status}`);
  }
  let modelsDev: ModelsDevResponse | undefined;
  if (options.modelsDev !== false) {
    progress?.("Loading models.dev catalog for metadata enrichment...");
    modelsDev = await getModelsDevCatalog(options);
  }
  const models = (listResult.data.data ?? [])
    .map((entry) => mapFromModelsList(entry, modelsDev))
    .filter((m): m is DiscoveredModel => m !== undefined);
  const deduplicated = deduplicateModels(models);
  reportWithheldToolRepairForModels(deduplicated);
  return { source: "models_list", models: deduplicated };
}
