import { readFile } from "node:fs/promises";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import { writeJsonAtomic } from "./cache.js";
import {
  type CatalogResolution,
  catalogResolution,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  effectiveDeploymentCount,
  type FamilyEvidence,
  isResponsesMode,
  reduceModelGroup,
  type SemanticFamily,
  type SemanticModel,
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

export function shouldSuppressReasoningContent(modelId: string): boolean {
  return isMoonshotModel(modelId) && !FORCED_THINKING_MODEL_PATTERN.test(modelId);
}

export function buildCompat(
  modelId: string,
  api: DiscoveredModel["api"] = "openai-completions",
  semantic?: FamilyEvidence,
): DiscoveredModel["compat"] {
  if (api === "openai-responses") return {};
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

function catalogProviderCandidates(id: string, ownedBy?: string): BuiltinProvider[] {
  const candidates = [toKnownProvider(ownedBy), toKnownProvider(id.split("/")[0])];
  const unprefixed = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  const anthropicAlias = unprefixed.toLowerCase().replaceAll(".", "-");
  if (/^(?:claude-)?(?:opus|sonnet|haiku)-\d+-\d+$/.test(anthropicAlias)) candidates.push("anthropic");
  if (anthropicAlias === "fable-5" || anthropicAlias === "opus-5") candidates.push("anthropic");
  return [...new Set(candidates.filter((provider): provider is BuiltinProvider => provider !== undefined))];
}

function resolveCatalogModel(
  id: string,
  ownedBy?: string,
): { provider: BuiltinProvider; model: Model<Api> } | undefined {
  const lookupIds = catalogLookupIds(id);
  for (const provider of catalogProviderCandidates(id, ownedBy)) {
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

// A cached model written before request policies existed carries none, and
// startup refreshes offline, so strict tool-message repair would silently stop
// for it. Re-derive the policy from the compatibility evidence already stored
// on the model rather than from its route name: the name is not evidence of a
// backend, and a route that only looks like Kimi never carried this block.
function restoreCachedModelPolicy(model: Model<Api>): Model<Api> {
  const cached = model as Model<Api> & { litellmPolicy?: LiteLLMModelPolicy };
  if (cached.litellmPolicy || !hasMoonshotCompatEvidence(model.compat)) return model;
  const restored: typeof cached = { ...cached, litellmPolicy: { normalizeStrictToolMessages: true } };
  return restored;
}

export function enrichCachedModel(input: Model<Api>): Model<Api> {
  const model = restoreCachedModelPolicy(input);
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
    reasoning: catalogModel.reasoning,
    thinkingLevelMap: catalogModel.thinkingLevelMap,
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

function adapterCatalogProvider(adapter: string | undefined): BuiltinProvider | undefined {
  const normalized = adapter?.trim().toLowerCase();
  return normalized ? (ADAPTER_CATALOG_PROVIDERS[normalized] ?? toKnownProvider(normalized)) : undefined;
}

export function resolveModelInfoCatalog(entry: ModelInfoEntry): CatalogResolution | undefined {
  const adapterProvider = adapterCatalogProvider(entry.model_info?.litellm_provider);
  const routingModel = entry.litellm_params?.model?.trim() || undefined;
  const baseModel = entry.model_info?.base_model?.trim() || undefined;
  const routingFamily = routingModel ? semanticFamily(routingModel) : undefined;
  const baseFamily = baseModel ? semanticFamily(baseModel) : undefined;
  // A routing model and a declared base model that name different families
  // describe different backends. Neither one is trustworthy metadata authority
  // for this deployment, so report the contradiction and resolve nothing.
  if (routingFamily !== undefined && baseFamily !== undefined && routingFamily !== baseFamily) {
    return { semanticFamily: "conflicting" };
  }
  const model =
    (routingModel ? semanticModel(routingModel) : undefined) ?? (baseModel ? semanticModel(baseModel) : undefined);
  const candidates = [routingModel, baseModel].filter((candidate): candidate is string => candidate !== undefined);
  for (const candidate of candidates) {
    const resolved = resolveCatalogModel(candidate, adapterProvider);
    // Catalog metadata and family must describe the same candidate; pairing one
    // candidate's family with another's cost silently drops vendor compat.
    if (resolved)
      return {
        ...catalogResolution(resolved.provider, semanticFamily(candidate), resolved.model),
        ...(model ? { semanticModel: model } : {}),
      };
  }
  const family = routingFamily ?? baseFamily;
  return family ? { semanticFamily: family, ...(model ? { semanticModel: model } : {}) } : undefined;
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

function mapFromModelInfoGroup(entries: readonly ModelInfoEntry[]): DiscoveredModel | undefined {
  // Exact repeats of an identified deployment are one effective deployment;
  // anonymous rows remain distinct because LiteLLM provides no identity to dedupe.
  const deploymentCount = effectiveDeploymentCount(entries);
  const singleton = deploymentCount === 1;
  const reduced = reduceModelGroup(entries, (entry) => {
    const resolved = resolveModelInfoCatalog(entry);
    if (resolved || !singleton) return resolved;
    const id = entry.model_name;
    const catalog = id ? resolveCatalogModel(id) : undefined;
    return catalog ? catalogResolution(catalog.provider, semanticFamily(catalog.model.id), catalog.model) : undefined;
  });
  if (!reduced) return undefined;
  const incompleteMetadataName =
    deploymentCount > 1 ? `${reduced.id} (incomplete metadata)` : `${reduced.id} (no metadata)`;
  const reasoningPolicy = reduced.semanticModel ? reduced.reasoningPolicy : undefined;
  return {
    id: reduced.id,
    name: reduced.hasCompleteCost ? reduced.id : incompleteMetadataName,
    reasoning: reasoningPolicy?.reasoning ?? reduced.reasoning,
    ...(reasoningPolicy
      ? reasoningPolicy.thinkingLevelMap
        ? { thinkingLevelMap: reasoningPolicy.thinkingLevelMap }
        : {}
      : reduced.thinkingLevelMap
        ? { thinkingLevelMap: reduced.thinkingLevelMap }
        : {}),
    input: reduced.vision ? ["text", "image"] : ["text"],
    cost: reduced.cost,
    contextWindow: reduced.contextWindow,
    maxTokens: reduced.maxTokens,
    api: reduced.api,
    compat:
      reduced.api === "openai-completions"
        ? { ...buildCompat(reduced.id, reduced.api, reduced.semanticFamily), ...reasoningPolicy?.compat }
        : buildCompat(reduced.id, reduced.api, reduced.semanticFamily),
    ...(reduced.semanticFamily === "kimi" || (reduced.semanticFamily === undefined && isMoonshotModel(reduced.id))
      ? { litellmPolicy: { normalizeStrictToolMessages: true } }
      : {}),
  };
}

function mapFromModelInfo(entry: ModelInfoEntry): DiscoveredModel | undefined {
  return mapFromModelInfoGroup([entry]);
}

function mapFromHealthModelInfo(entry: ModelInfoEntry, fallbackId: string | undefined): DiscoveredModel | undefined {
  const named = entry.model_name || !fallbackId ? entry : { ...entry, model_name: fallbackId };
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
  if (!entry.model_name) delete model.thinkingLevelMap;
  return model;
}

function mapFromHealthEndpoint(entry: { model?: string }): DiscoveredModel | undefined {
  const id = entry.model;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id);
  return {
    id,
    name: catalogModel?.name ?? id,
    reasoning: catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: "openai-completions",
    compat: buildCompat(id),
    ...(isMoonshotModel(id) ? { litellmPolicy: { normalizeStrictToolMessages: true } } : {}),
  };
}

function mapFromModelsList(
  entry: ModelsListEntry,
  modelsDev: ModelsDevResponse | undefined,
): DiscoveredModel | undefined {
  const id = entry.id;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id, entry.owned_by);
  const modelsDevMetadata = mapModelsDevMetadata(findModelsDevModel(modelsDev, id, entry.owned_by));
  return {
    id,
    name: modelsDevMetadata.name ?? catalogModel?.name ?? `${id} (no metadata)`,
    reasoning: modelsDevMetadata.reasoning ?? catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: modelsDevMetadata.input ?? catalogModel?.input ?? ["text"],
    cost: modelsDevMetadata.cost ?? catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: modelsDevMetadata.contextWindow ?? catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: modelsDevMetadata.maxTokens ?? catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: "openai-completions",
    compat: buildCompat(id),
    ...(isMoonshotModel(id) ? { litellmPolicy: { normalizeStrictToolMessages: true } } : {}),
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
  // against each other the way `/model/info` groups are — `deduplicateModels`
  // keeps whichever finished first. Reducing here means aggregating the
  // per-deployment detail fetches by `model_name` and feeding whole groups to
  // `mapFromModelInfoGroup`, which also changes naming and the singleton
  // catalog fallback on this path. That belongs in its own change; see the
  // `/health` note in AGENTS.md and README.md.
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
  return models.filter((model): model is DiscoveredModel => model !== undefined);
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
      if (!entry.model_name) continue;
      const group = groups.get(entry.model_name) ?? [];
      group.push(entry);
      groups.set(entry.model_name, group);
    }
    let models = [...groups.values()].map(mapFromModelInfoGroup).filter((m): m is DiscoveredModel => m !== undefined);
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
    return { source: "model_info", models: deduplicateModels(models) };
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
  return { source: "models_list", models: deduplicateModels(models) };
}
