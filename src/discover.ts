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
  type MessagesBackendCompat,
  reduceModelGroup,
  type SemanticFamily,
  wireString,
} from "./model-groups.js";
import type {
  DiscoveredModel,
  DiscoveryOptions,
  DiscoveryResult,
  HealthResponse,
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
const ANTHROPIC_MODEL_PATTERN = /(?:^|[-_/.:])(?:anthropic\/|(?:claude|opus|sonnet|haiku|fable)(?=$|[-_/.:]))/i;
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
  semanticFamily?: SemanticFamily,
): DiscoveredModel["compat"] {
  if (api === "anthropic-messages") return undefined;
  if (api === "openai-responses") return {};
  if (isMoonshotModel(modelId)) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    };
  }
  if (semanticFamily === "claude" || ANTHROPIC_MODEL_PATTERN.test(modelId)) {
    return { supportsStore: false, cacheControlFormat: "anthropic" };
  }
  return { supportsStore: false };
}

function toKnownProvider(provider: string | undefined): BuiltinProvider | undefined {
  if (!provider) return undefined;
  const normalized = provider.trim().toLowerCase();
  return KNOWN_PROVIDER_SET.has(normalized) ? (normalized as BuiltinProvider) : undefined;
}

function catalogProviderCandidates(id: string, ownedBy?: string, adapterProvider?: BuiltinProvider): BuiltinProvider[] {
  const candidates = [adapterProvider, toKnownProvider(ownedBy), toKnownProvider(id.split("/")[0])].filter(
    (provider): provider is BuiltinProvider => provider !== undefined,
  );
  // A declared `litellm_provider` adapter is authoritative and outranks the bare-alias
  // exception: a Vertex- or Bedrock-served Claude must not be priced from the first-party
  // Anthropic catalog just because its alias reads like a Claude model. Messages
  // compatibility is resolved separately and still recognizes those routes.
  //
  // `owned_by` from `/v1/models` is NOT such evidence — LiteLLM reports "openai" for every
  // route there — so the exception still applies on the fallback and cache-enrichment
  // paths, which is the behavior D-005 approves for unqualified route names.
  if (adapterProvider) return [...new Set(candidates)];
  const unprefixed = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  const anthropicAlias = unprefixed.toLowerCase().replaceAll(".", "-");
  if (
    /^(?:claude-)?(?:opus|sonnet|haiku)-\d+-\d+$/.test(anthropicAlias) ||
    anthropicAlias === "fable-5" ||
    anthropicAlias === "opus-5"
  ) {
    candidates.push("anthropic");
  }
  return [...new Set(candidates)];
}

function resolveCatalogModel(
  id: string,
  ownedBy?: string,
  adapterProvider?: BuiltinProvider,
): { provider: BuiltinProvider; model: Model<Api> } | undefined {
  const lookupIds = catalogLookupIds(id);
  for (const provider of catalogProviderCandidates(id, ownedBy, adapterProvider)) {
    const model = findCatalogModelInProvider(provider, lookupIds);
    if (model) return { provider, model };
  }
  return undefined;
}

function findCatalogModel(id: string, ownedBy?: string): Model<Api> | undefined {
  return resolveCatalogModel(id, ownedBy)?.model;
}

export function enrichCachedModel(model: Model<Api>): Model<Api> {
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

// Trailing deployment decoration a router adds to a backend id: a Bedrock model
// version (`-v1:0`, sometimes catalogued as `-v1`), a Vertex serving suffix
// (`@20260101`), or a dated release snapshot. Region and adapter prefixes are
// deliberately NOT stripped — provider catalogs key Bedrock models by inference
// profile (`us.anthropic.…`), so removing the region would lose the identity this
// exists to find. Most specific first, so an exactly-catalogued id always wins.
function undecoratedBackendIds(id: string): string[] {
  const routed = (id.split("/").pop() ?? id).toLowerCase();
  const undecorated = routed.replace(/-v\d+(?::\d+)?$/, "").replace(/@[a-z0-9-]+$/, "");
  return [
    ...new Set([routed, routed.replace(/:\d+$/, ""), undecorated, undecorated.replace(/-\d{8}$/, "")].filter(Boolean)),
  ];
}

function catalogLookupIds(id: string): string[] {
  const lookupIds = new Set([id]);
  const unprefixed = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  lookupIds.add(unprefixed);
  // Adapter-scoped only: this widens the id spellings tried inside a provider the
  // caller already authorized, never the set of providers searched.
  for (const candidate of undecoratedBackendIds(id)) lookupIds.add(candidate);

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

function messagesCompatOf(model: Model<Api>): MessagesBackendCompat | undefined {
  if (model.api !== "anthropic-messages") return undefined;
  const compat = (model as Model<"anthropic-messages">).compat;
  const carried: MessagesBackendCompat = {};
  if (compat?.forceAdaptiveThinking !== undefined) carried.forceAdaptiveThinking = compat.forceAdaptiveThinking;
  if (compat?.supportsTemperature !== undefined) carried.supportsTemperature = compat.supportsTemperature;
  if (compat?.supportsStrictTools !== undefined) carried.supportsStrictTools = compat.supportsStrictTools;
  return carried;
}

// The first-party Anthropic catalog keys models bare (`claude-opus-4-7`), so a
// compatibility lookup additionally strips the adapter path and any cross-region or
// cross-partition inference profile (`us.`, `eu.`, `apac.`, `us-gov.`, `global.`)
// before the shared decoration stripping.
function anthropicBackendLookupIds(id: string): string[] {
  const routed = (id.split("/").pop() ?? id).toLowerCase();
  const base = routed.replace(/^(?:[a-z0-9-]+\.)*anthropic[./]/, "");
  return undecoratedBackendIds(base);
}

// Anthropic compatibility for a LiteLLM backend routing id, derived only from the
// backend identity LiteLLM reports. `undefined` means the backend model is unknown
// to the catalog; `{}` means it is known and carries no special requirements.
function messagesCompatFromBackend(id: string): MessagesBackendCompat | undefined {
  const model = findCatalogModelInProvider("anthropic", anthropicBackendLookupIds(id));
  return model ? messagesCompatOf(model) : undefined;
}

function semanticFamily(id: string): SemanticFamily | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])(?:anthropic|claude|opus|sonnet|haiku|fable)(?:$|[./_:-])/.test(value)) return "claude";
  if (/(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])/.test(value)) return "kimi";
  if (/(?:^|[./_-])deepseek(?:$|[./_:-])/.test(value)) return "deepseek";
  if (/(?:^|[./_-])gemini(?:$|[./_:-])/.test(value)) return "gemini";
  if (/(?:^|[./_-])(?:openai|gpt|o\d)(?:$|[./_:-])/.test(value)) return "openai";
  return undefined;
}

const CLAUDE_CAPABLE_ADAPTERS = new Set(["anthropic", "bedrock", "bedrock_converse", "vertex_ai"]);
// `fable` is included on the same bounded evidence as the other bare aliases: the
// catalog carries `claude-fable-5` and `*.anthropic.claude-fable-5`, and both the
// provider-candidate and lookup-id tables already recognize the bare `fable-5` alias.
const CLAUDE_MODEL_PATTERN = /(?:^|[./_-])(?:claude|opus|sonnet|haiku|fable)(?:$|[./_:-])/i;

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

function isClaudeWitness(candidate: string | undefined): candidate is string {
  return candidate !== undefined && semanticFamily(candidate) === "claude" && CLAUDE_MODEL_PATTERN.test(candidate);
}

function resolveClaudeBackendIdentity(
  adapter: string | undefined,
  routingModel: string | undefined,
  baseModel: string | undefined,
  adapterProvider: BuiltinProvider | undefined,
): CatalogResolution["backendIdentity"] {
  if (!adapter || !CLAUDE_CAPABLE_ADAPTERS.has(adapter)) return undefined;
  if (isClaudeWitness(routingModel)) return { semanticFamily: "claude" };
  if (routingModel) {
    // The routing target does not name Claude. Contrary family text fails closed.
    if (semanticFamily(routingModel) !== undefined) return undefined;
    // A routing target the adapter's own catalog recognizes is positive contrary
    // evidence even when its text names no family, so a Nova route cannot borrow a
    // Claude `base_model`.
    if (resolveCatalogModel(routingModel, undefined, adapterProvider)) return undefined;
    // Otherwise the target is an opaque handle — a provisioned-throughput or
    // application-inference-profile ARN — and `base_model` is the only backend evidence
    // the router offers. It may establish semantic family and Messages compatibility;
    // catalog identity and pricing still come from the adapter's own provider, so this
    // cannot relabel a Bedrock route as first-party Anthropic.
  }
  return isClaudeWitness(baseModel) ? { semanticFamily: "claude" } : undefined;
}

export function resolveModelInfoCatalog(entry: ModelInfoEntry): CatalogResolution | undefined {
  const adapter = wireString(entry.model_info?.litellm_provider)?.trim().toLowerCase();
  const adapterProvider = adapterCatalogProvider(adapter);
  const routingModel = wireString(entry.litellm_params?.model)?.trim() || undefined;
  const baseModel = wireString(entry.model_info?.base_model)?.trim() || undefined;
  const routingFamily = routingModel ? semanticFamily(routingModel) : undefined;
  const baseFamily = baseModel ? semanticFamily(baseModel) : undefined;
  // A recognized adapter catalog entry is contrary evidence even when the family
  // classifier does not name it (for example Bedrock Nova). It must outrank a
  // Claude-looking base_model rather than letting the base relabel the backend.
  const routingCatalog = routingModel ? resolveCatalogModel(routingModel, undefined, adapterProvider) : undefined;
  const conflictingFamilies =
    routingModel !== undefined &&
    baseFamily !== undefined &&
    ((routingFamily !== undefined && routingFamily !== baseFamily) ||
      (routingCatalog !== undefined && semanticFamily(routingCatalog.model.id) !== baseFamily));
  const family = routingFamily ?? (routingCatalog ? semanticFamily(routingCatalog.model.id) : undefined) ?? baseFamily;
  const candidates = [routingModel, baseModel].filter((candidate): candidate is string => candidate !== undefined);
  const backendIdentity = resolveClaudeBackendIdentity(
    adapter,
    routingModel,
    conflictingFamilies ? undefined : baseModel,
    adapterProvider,
  );
  const resolvedFamily = family ?? (backendIdentity ? "claude" : undefined);
  // Catalog metadata and Anthropic compatibility policy may come only from a candidate
  // whose own semantic family POSITIVELY matches the resolved family. An unrecognized
  // family is not agreement: admitting it let a `base_model` the family regex does not
  // name (Qwen, Mistral, Nova, Llama) hand its pricing and limits to a Claude route.
  const agreeing = candidates.filter(
    (candidate) => resolvedFamily === undefined || semanticFamily(candidate) === resolvedFamily,
  );
  const messagesCompat = agreeing.reduce<MessagesBackendCompat | undefined>(
    (carried, candidate) => carried ?? messagesCompatFromBackend(candidate),
    undefined,
  );

  for (const candidate of agreeing) {
    const resolved = resolveCatalogModel(candidate, undefined, adapterProvider);
    if (resolved) {
      return {
        ...catalogResolution(resolved.provider, resolvedFamily, resolved.model),
        backendIdentity,
        messagesCompat: messagesCompat ?? messagesCompatOf(resolved.model),
      };
    }
  }
  // Identifiable backend, opaque metadata: carry compatibility without granting the
  // catalog authority over provider identity, pricing, or limits.
  if (resolvedFamily)
    return { semanticFamily: resolvedFamily, backendIdentity, ...(messagesCompat && { messagesCompat }) };
  return undefined;
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

function mapFromModelInfoGroup(
  entries: readonly ModelInfoEntry[],
  ambiguousRoutes?: string[],
): DiscoveredModel | undefined {
  const reduced = reduceModelGroup(entries, (entry, singleton) => {
    const resolved = resolveModelInfoCatalog(entry);
    if (resolved || !singleton) return resolved;
    const id = wireString(entry.model_name);
    const catalog = id ? resolveCatalogModel(id) : undefined;
    if (!catalog) return undefined;
    return catalogResolution(catalog.provider, semanticFamily(catalog.model.id), catalog.model);
  });
  if (!reduced) return undefined;
  if (reduced.catalogAuthorityAmbiguous) ambiguousRoutes?.push(reduced.id);
  const shared = {
    id: reduced.id,
    // Reduced groups never borrow the fallback ` (no metadata)` sentinel, which
    // authorizes catalog re-derivation from route text during offline cache reads.
    name: reduced.hasCompleteCost ? reduced.id : `${reduced.id} (incomplete metadata)`,
    reasoning: reduced.reasoning,
    ...(reduced.thinkingLevelMap ? { thinkingLevelMap: reduced.thinkingLevelMap } : {}),
    input: (reduced.vision ? ["text", "image"] : ["text"]) as ("text" | "image")[],
    cost: reduced.cost,
    contextWindow: reduced.contextWindow,
    maxTokens: reduced.maxTokens,
  };
  if (reduced.api === "anthropic-messages") {
    return { ...shared, api: "anthropic-messages", compat: carriedMessagesCompat(reduced.messagesCompat) };
  }
  if (reduced.api === "openai-responses") {
    return { ...shared, api: "openai-responses", compat: buildCompat(reduced.id, "openai-responses") };
  }
  return {
    ...shared,
    api: "openai-completions",
    compat: buildCompat(reduced.id, "openai-completions", reduced.semanticFamily),
  };
}

// An empty carried compat is indistinguishable from "no requirements", so omit it.
function carriedMessagesCompat(compat: MessagesBackendCompat | undefined): MessagesBackendCompat | undefined {
  return compat && Object.keys(compat).length > 0 ? compat : undefined;
}

function mapFromModelInfo(entry: ModelInfoEntry): DiscoveredModel | undefined {
  return mapFromModelInfoGroup([entry]);
}

function mapFromHealthModelInfo(entry: ModelInfoEntry, fallbackId: string | undefined): DiscoveredModel | undefined {
  const model = mapFromModelInfo(
    wireString(entry.model_name) || !fallbackId ? entry : { ...entry, model_name: fallbackId },
  );
  if (!model) return undefined;
  // A `/health` detail row is one deployment, not a complete route group, so it is not
  // sufficient evidence to expose route-wide thinking controls or route natively.
  // Downgrade to Chat Completions and rebuild compat for that transport, dropping every
  // Messages-only field.
  delete model.thinkingLevelMap;
  if (model.api === "openai-completions") return model;
  const sourceApi = model.api;
  const downgradedCompat = sourceApi === "anthropic-messages" ? undefined : model.compat;
  return {
    ...model,
    api: "openai-completions",
    compat: {
      ...buildCompat(model.id, "openai-completions", sourceApi === "anthropic-messages" ? "claude" : undefined),
      ...downgradedCompat,
    },
  };
}

function mapFromHealthEndpoint(entry: { model?: string }): DiscoveredModel | undefined {
  const id = wireString(entry.model);
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id);
  return {
    id,
    name: catalogModel?.name ?? `${id} (incomplete metadata)`,
    reasoning: catalogModel?.reasoning ?? false,
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: "openai-completions",
    compat: buildCompat(id, "openai-completions"),
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
  };
}

async function discoverFromHealth(
  base: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void; silent?: boolean },
): Promise<DiscoveredModel[]> {
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /health endpoint...");
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
      const route = wireString(entry.model_name);
      if (!route) continue;
      const group = groups.get(route) ?? [];
      group.push(entry);
      groups.set(route, group);
    }
    const ambiguousRoutes: string[] = [];
    let models = [...groups.values()]
      .map((group) => mapFromModelInfoGroup(group, ambiguousRoutes))
      .filter((m): m is DiscoveredModel => m !== undefined);
    reportAmbiguousCatalogAuthority(ambiguousRoutes);
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
