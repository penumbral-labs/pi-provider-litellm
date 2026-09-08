import { isIP } from "node:net";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import { resolveBackendIdentity } from "./backend-identity.js";
import {
  type CatalogResolution,
  catalogResolution,
  conservativeCostTiers,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  hasMixedIncompatibleDeploymentModes,
  reduceModelGroup,
  wireString,
} from "./model-groups.js";
import { loadPublicCatalog, type PublicCatalog, type PublicCatalogRecord } from "./public-catalog.js";
import { intersectThinkingLevelMaps } from "./thinking-levels.js";
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
const PUBLIC_CATALOG_BUDGET_MS = 1000;
const PUBLIC_CATALOG_PROVIDER_BY_FAMILY = {
  claude: "anthropic",
  deepseek: "deepseek",
  gemini: "gemini",
  kimi: "moonshot",
  openai: "openai",
} as const;
const KNOWN_PROVIDER_SET = new Set<string>(getProviders());
export function normalizeBaseUrl(input: string, allowInsecureHttp = false): string {
  const url = new URL(input);
  const hostname = url.hostname.toLowerCase();
  const loopback =
    hostname === "localhost" || hostname === "[::1]" || (isIP(hostname) === 4 && hostname.startsWith("127."));
  if (url.protocol !== "https:" && !(url.protocol === "http:" && (loopback || allowInsecureHttp))) {
    throw new Error("LiteLLM base URL must use HTTPS except for loopback hosts");
  }
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

const MOONSHOT_ROUTE_PROVIDERS = new Set(["moonshot", "moonshotai"]);

function isMoonshotRoute(entry: ModelInfoEntry): boolean {
  const params = entry.litellm_params;
  if (!params) return false;
  const model = wireString(params.model)?.trim();
  if (!model) return false;
  const providers = [
    wireString(params.custom_llm_provider)?.trim(),
    model.includes("/") ? model.split("/", 1)[0] : undefined,
  ].filter((provider): provider is string => Boolean(provider));
  return providers.length > 0 && providers.every((provider) => MOONSHOT_ROUTE_PROVIDERS.has(provider.toLowerCase()));
}

function shouldSuppressRouteReasoningContent(modelId: string, entry: ModelInfoEntry): boolean {
  const routeModelId = wireString(entry.litellm_params?.model);
  return (
    isMoonshotRoute(entry) &&
    !FORCED_THINKING_MODEL_PATTERN.test(modelId) &&
    !(routeModelId && FORCED_THINKING_MODEL_PATTERN.test(routeModelId))
  );
}

function aggregateSuppressionEvidence(evidence: Iterable<boolean>): boolean {
  let hasEvidence = false;
  for (const suppress of evidence) {
    hasEvidence = true;
    if (!suppress) return false;
  }
  return hasEvidence;
}

function shouldSuppressReasoningContent(modelId: string): boolean {
  return isMoonshotModel(modelId) && !FORCED_THINKING_MODEL_PATTERN.test(modelId);
}

export function emitsThinkTags(modelId: string): boolean {
  return shouldSuppressReasoningContent(modelId);
}

export function buildCompat(modelId: string): DiscoveredModel["compat"] {
  if (isMoonshotModel(modelId)) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    };
  }
  if (ANTHROPIC_MODEL_PATTERN.test(modelId)) {
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

export function enrichCachedModel(model: Model<Api>): Model<Api> {
  // This sentinel is emitted only by the evidence-free `/v1/models` fallback, so
  // re-deriving catalog metadata from the model id here cannot re-authorize a
  // reduced `/model/info` group whose catalog authority was withheld. Reduced
  // groups carry the distinct ` (incomplete metadata)` marker instead.
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
    api: catalogModel.api === "openai-responses" ? "openai-responses" : "openai-completions",
    compat: buildCompat(model.id),
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

const ADAPTER_CATALOG_PROVIDERS: Readonly<Record<string, BuiltinProvider>> = {
  anthropic: "anthropic",
  claude: "anthropic",
  azure: "azure-openai-responses",
  azure_ai: "azure-openai-responses",
  bedrock: "amazon-bedrock",
  bedrock_converse: "amazon-bedrock",
  deepseek: "deepseek",
  "fireworks-ai": "fireworks",
  fireworks_ai: "fireworks",
  gemini: "google",
  kimi: "moonshotai",
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

function adaptPublicCatalogRecord(
  provider: string,
  catalogModelId: string,
  record: PublicCatalogRecord | undefined,
): CatalogResolution {
  const piProvider = adapterCatalogProvider(record?.provider ?? provider);
  const resolved = piProvider
    ? (resolveCatalogModel(record?.modelId ?? catalogModelId, piProvider) ??
      resolveCatalogModel(catalogModelId, piProvider))
    : undefined;
  const piCatalog = resolved ? catalogResolution(resolved.provider, resolved.model) : undefined;
  const piCost = piCatalog?.cost;
  return {
    provider,
    // A models.dev hit names the model canonically even when Pi's catalog does not know it yet,
    // so two spellings of one backend model reduce to one identity instead of a conflict.
    catalogModelId: resolved?.model.id ?? record?.modelId ?? catalogModelId,
    ...(piCatalog?.reasoning !== undefined ? { reasoning: piCatalog.reasoning } : {}),
    ...(piCatalog?.thinkingLevelMap ? { thinkingLevelMap: piCatalog.thinkingLevelMap } : {}),
    ...(record?.modalities
      ? { vision: record.modalities.includes("image") }
      : piCatalog?.vision !== undefined
        ? { vision: piCatalog.vision }
        : {}),
    ...(record?.limits?.context !== undefined
      ? { contextWindow: record.limits.context }
      : piCatalog?.contextWindow !== undefined
        ? { contextWindow: piCatalog.contextWindow }
        : {}),
    ...(record?.limits?.output !== undefined
      ? { maxTokens: record.limits.output }
      : piCatalog?.maxTokens !== undefined
        ? { maxTokens: piCatalog.maxTokens }
        : {}),
    ...(record?.cost || piCost
      ? {
          cost: {
            ...((record?.cost?.input ?? piCost?.input) !== undefined
              ? { input: record?.cost?.input ?? piCost?.input }
              : {}),
            ...((record?.cost?.output ?? piCost?.output) !== undefined
              ? { output: record?.cost?.output ?? piCost?.output }
              : {}),
            ...((record?.cost?.cacheRead ?? piCost?.cacheRead) !== undefined
              ? { cacheRead: record?.cost?.cacheRead ?? piCost?.cacheRead }
              : {}),
            ...((record?.cost?.cacheWrite ?? piCost?.cacheWrite) !== undefined
              ? { cacheWrite: record?.cost?.cacheWrite ?? piCost?.cacheWrite }
              : {}),
            ...(piCost?.tiers ? { tiers: piCost.tiers } : {}),
          },
        }
      : {}),
  };
}

export function resolveModelInfoCatalog(
  entry: ModelInfoEntry,
  publicCatalog?: PublicCatalog,
): CatalogResolution | undefined {
  const hasBackendIdentity = Boolean(
    wireString(entry.model_info?.base_model)?.trim() || wireString(entry.litellm_params?.model)?.trim(),
  );
  if (!hasBackendIdentity) return undefined;
  const identity = resolveBackendIdentity(entry);
  if (!identity) return undefined;
  if (identity.provider) {
    const record = publicCatalog?.lookup(identity.provider, identity.modelId);
    return adaptPublicCatalogRecord(identity.provider, identity.qualifiedId, record);
  }
  if (!identity.family) return undefined;

  const publicProvider = PUBLIC_CATALOG_PROVIDER_BY_FAMILY[identity.family];
  const record = publicCatalog?.lookup(publicProvider, identity.modelId);
  return adaptPublicCatalogRecord(publicProvider, identity.qualifiedId, record);
}

function awaitWithSignal<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
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

async function awaitEnrichmentWithinBudget<T>(
  promise: Promise<T>,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<T | undefined> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const budget = new Promise<undefined>((resolve) => {
    timeout = setTimeout(() => resolve(undefined), timeoutMs);
  });
  try {
    return await Promise.race([signal ? awaitWithSignal(promise, signal) : promise, budget]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

function loadDiscoveryPublicCatalog(options: DiscoveryOptions): Promise<PublicCatalog | undefined> {
  const publicCatalogPromise = loadPublicCatalog({
    cachePath: options.modelsDevCachePath,
    offline: options.modelsDev === false ? true : undefined,
    timeoutMs: DEFAULT_TIMEOUT_MS,
  });
  return awaitEnrichmentWithinBudget(
    publicCatalogPromise,
    options.signal,
    Math.min(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, PUBLIC_CATALOG_BUDGET_MS),
  );
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

const DIAGNOSTIC_ROUTE_SAMPLE = 3;

function reportBoundedRoutes(
  reported: Set<string>,
  routes: readonly string[],
  describe: (count: number) => string,
): void {
  const unreported = [...new Set(routes)].filter((route) => !reported.has(route));
  if (unreported.length === 0) return;
  for (const route of unreported) reported.add(route);
  const hidden = unreported.length - DIAGNOSTIC_ROUTE_SAMPLE;
  const sample = unreported.slice(0, DIAGNOSTIC_ROUTE_SAMPLE).join(", ");
  process.stderr.write(`${describe(unreported.length)}: ${sample}${hidden > 0 ? ` (+${hidden} more)` : ""}\n`);
}

// Reported routes, so a persistent misconfiguration is announced once rather than on
// every background refresh and every `/model` open. Keyed by route rather than a
// single flag so a newly ambiguous route is still reported. Mirrors the once-per-
// process diagnostic set in src/index.ts.
const reportedAmbiguousRoutes = new Set<string>();

// Withholding catalog authority can be invisible in the model name when the router
// supplies complete prices; limits and other catalog-derived metadata may still use
// conservative defaults. Report that degradation regardless of
// LITELLM_VERBOSE_DISCOVERY, carrying only a count and bounded public route ids.
function reportAmbiguousCatalogAuthority(routes: readonly string[]): void {
  reportBoundedRoutes(
    reportedAmbiguousRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) have missing or conflicting deployment provider evidence; ` +
      "catalog limits, pricing, and reasoning metadata are withheld",
  );
}

const reportedIncompatibleModeRoutes = new Set<string>();

function reportIncompatibleDeploymentModes(routes: readonly string[]): void {
  reportBoundedRoutes(
    reportedIncompatibleModeRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) mix chat-style and explicitly incompatible deployment modes; ` +
      "the routes are withheld because not every deployment can accept chat requests",
  );
}

function mapFromModelInfoGroup(
  entries: readonly ModelInfoEntry[],
  publicCatalog?: PublicCatalog,
  ambiguousRoutes?: string[],
): DiscoveredModel | undefined {
  const reduced = reduceModelGroup(entries, (entry) => resolveModelInfoCatalog(entry, publicCatalog));
  if (!reduced) return undefined;
  if (reduced.catalogAuthorityAmbiguous) ambiguousRoutes?.push(reduced.id);
  // A surviving group contains only chat-style rows, because the reducer withholds
  // every group containing an explicitly incompatible deployment mode.
  const suppressReasoningContent = aggregateSuppressionEvidence(
    entries.map((entry) => shouldSuppressRouteReasoningContent(reduced.id, entry)),
  );
  return {
    id: reduced.id,
    // Reduced groups never borrow the ` (no metadata)` sentinel, which authorizes
    // catalog re-derivation from the model id during offline cache reads. Complete
    // router pricing does not make catalog-derived capabilities and limits authoritative.
    name: reduced.hasCompleteMetadata ? reduced.id : `${reduced.id} (incomplete metadata)`,
    reasoning: reduced.reasoning,
    ...(reduced.thinkingLevelMap ? { thinkingLevelMap: reduced.thinkingLevelMap } : {}),
    input: reduced.vision ? ["text", "image"] : ["text"],
    cost: reduced.cost,
    contextWindow: reduced.contextWindow,
    maxTokens: reduced.maxTokens,
    api: reduced.api,
    compat: buildCompat(reduced.id),
    ...(suppressReasoningContent ? { suppressReasoningContent: true } : {}),
  };
}

function syntheticHealthRow(route: string): ModelInfoEntry {
  const catalogModel = findCatalogModel(route);
  return {
    model_name: route,
    // The route name authorizes nothing but the transport, which the Pi catalog
    // supplies for an evidence-free entry; levels stay denied and no catalog
    // metadata is granted.
    model_info: { mode: catalogModel?.api === "openai-responses" ? "responses" : "chat" },
  };
}

function mapFromModelsList(entry: ModelsListEntry): DiscoveredModel | undefined {
  const id = wireString(entry.id);
  if (!id) return undefined;
  const ownedBy = wireString(entry.owned_by);
  const catalogModel = findCatalogModel(id, ownedBy);
  return {
    id,
    name: catalogModel?.name ?? `${id} (no metadata)`,
    reasoning: catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: catalogModel?.api === "openai-responses" ? "openai-responses" : "openai-completions",
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
  const denyThinkingLevels = new Set<ModelInfoEntry>();
  const rows = await Promise.all(
    endpoints.map(async (endpoint) => {
      const healthRoute = wireString(endpoint.model);
      let entry: ModelInfoEntry | undefined;
      if (endpoint.model_id) {
        const infoResult = await fetchJson<ModelInfoResponse>(
          `${base}/model/info?litellm_model_id=${encodeURIComponent(endpoint.model_id)}`,
          apiKey,
          options,
        );
        const detail = infoResult.ok ? infoResult.data.data?.[0] : undefined;
        if (detail) {
          if (wireString(detail.model_name)) {
            entry = detail;
          } else if (healthRoute) {
            entry = { ...detail, model_name: healthRoute };
            denyThinkingLevels.add(entry);
          }
        }
      }
      entry ??= healthRoute ? syntheticHealthRow(healthRoute) : undefined;
      completed++;
      if (completed % 10 === 0 || completed === endpoints.length) {
        progress?.(`Fetched ${completed}/${endpoints.length} models...`);
      }
      return entry;
    }),
  );
  const groups = new Map<string, ModelInfoEntry[]>();
  for (const entry of rows) {
    const route = wireString(entry?.model_name);
    if (!entry || !route) continue;
    const group = groups.get(route) ?? [];
    group.push(entry);
    groups.set(route, group);
  }
  const publicCatalog = await loadDiscoveryPublicCatalog(options);
  const incompatibleModeRoutes: string[] = [];
  const models = [...groups.entries()]
    .map(([route, group]) => {
      if (hasMixedIncompatibleDeploymentModes(group)) incompatibleModeRoutes.push(route);
      const model = mapFromModelInfoGroup(group, publicCatalog);
      if (model && group.some((entry) => denyThinkingLevels.has(entry))) delete model.thinkingLevelMap;
      return model;
    })
    .filter((model): model is DiscoveredModel => model !== undefined);
  reportIncompatibleDeploymentModes(incompatibleModeRoutes);
  return models;
}

function deduplicateModels(models: DiscoveredModel[]): DiscoveredModel[] {
  const entries = new Map<string, { model: DiscoveredModel; suppressions: boolean[] }>();
  for (const model of models) {
    const existing = entries.get(model.id);
    if (existing) {
      existing.suppressions.push(model.suppressReasoningContent === true);
    } else {
      entries.set(model.id, { model, suppressions: [model.suppressReasoningContent === true] });
    }
  }
  return [...entries.values()].map(({ model, suppressions }) => {
    const deduplicated = { ...model };
    if (aggregateSuppressionEvidence(suppressions)) deduplicated.suppressReasoningContent = true;
    else delete deduplicated.suppressReasoningContent;
    return deduplicated;
  });
}

function wildcardMatches(route: string, modelId: string): boolean {
  const segments = route.split("*");
  let offset = 0;
  for (const [index, segment] of segments.entries()) {
    if (segment === "") continue;
    const found = modelId.indexOf(segment, offset);
    if (found < 0 || (index === 0 && found !== 0)) return false;
    offset = found + segment.length;
  }
  const suffix = segments.at(-1);
  return suffix === "" || modelId.endsWith(suffix ?? "");
}

function mapFromWildcardExpansion(
  entry: ModelsListEntry,
  wildcards: readonly DiscoveredModel[],
): DiscoveredModel | undefined {
  const id = wireString(entry.id);
  if (!id || id.includes("*")) return undefined;
  const matches = wildcards.filter((model) => wildcardMatches(model.id, id));
  if (matches.length === 0) return undefined;

  const api = matches.every((model) => model.api === "openai-responses") ? "openai-responses" : "openai-completions";
  const reasoning = matches.every((model) => model.reasoning);
  const vision = matches.every((model) => model.input.includes("image"));
  const contextWindow = Math.min(...matches.map((model) => model.contextWindow));
  const maxTokens = Math.min(...matches.map((model) => model.maxTokens));
  const thinkingLevelMap = reasoning
    ? intersectThinkingLevelMaps(matches.map((model) => model.thinkingLevelMap))
    : undefined;
  const incomplete = matches.some((model) => model.name.endsWith(" (incomplete metadata)"));
  const suppressReasoningContent =
    !FORCED_THINKING_MODEL_PATTERN.test(id) &&
    aggregateSuppressionEvidence(matches.map((model) => model.suppressReasoningContent === true));
  // Preserve every known tier even when a sibling has incomplete metadata. Omitting
  // a complete sibling's higher tier would understate the known worst-case rate;
  // the incomplete marker continues to signal that the resulting envelope is partial.
  const costTiers = conservativeCostTiers(matches.map((model) => model.cost));
  return {
    id,
    name: incomplete ? `${id} (incomplete metadata)` : id,
    reasoning,
    ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
    input: vision ? ["text", "image"] : ["text"],
    cost: {
      input: Math.max(...matches.map((model) => model.cost.input)),
      output: Math.max(...matches.map((model) => model.cost.output)),
      cacheRead: Math.max(...matches.map((model) => model.cost.cacheRead)),
      cacheWrite: Math.max(...matches.map((model) => model.cost.cacheWrite)),
      ...(costTiers ? { tiers: costTiers } : {}),
    },
    contextWindow,
    maxTokens,
    api,
    // Compatibility describes the concrete public id, while every authority-
    // bearing field above remains bounded by all matching wildcard groups.
    compat: buildCompat(id),
    ...(suppressReasoningContent ? { suppressReasoningContent: true } : {}),
  };
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void; silent?: boolean } = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl, options.allowInsecureHttp);
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /model/info endpoint...");
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const groups = new Map<string, ModelInfoEntry[]>();
    for (const entry of infoResult.data.data ?? []) {
      // A route without a readable public name cannot be grouped or addressed.
      const route = wireString(entry.model_name);
      if (!route) continue;
      const group = groups.get(route) ?? [];
      group.push(entry);
      groups.set(route, group);
    }
    const publicCatalog = await loadDiscoveryPublicCatalog(options);
    const ambiguousRoutes: string[] = [];
    const incompatibleModeRoutes: string[] = [];
    const reducedGroups = [...groups.entries()].map(([route, group]) => {
      if (hasMixedIncompatibleDeploymentModes(group)) incompatibleModeRoutes.push(route);
      return {
        route,
        model: mapFromModelInfoGroup(group, publicCatalog, ambiguousRoutes),
      };
    });
    let models = reducedGroups
      .map(({ model }) => model)
      .filter((model): model is DiscoveredModel => model !== undefined);
    reportIncompatibleDeploymentModes(incompatibleModeRoutes);
    reportAmbiguousCatalogAuthority(ambiguousRoutes);
    // LiteLLM's /model/info does NOT expand wildcard model_name entries (e.g.
    // "lemonade/*" backed by model: openai/* + check_provider_endpoint: true)
    // — it returns the literal wildcard only. The discovered ids live in
    // /v1/models instead. When /model/info contains any wildcard id, also query
    // /v1/models and merge the expanded (non-wildcard) entries in, dropping the
    // raw wildcard row so it doesn't surface as a phantom model choice.
    // Ref: docs.litellm.ai/docs/proxy/model_discovery
    const wildcardRoutes = reducedGroups.filter(({ route }) => route.includes("*"));
    if (wildcardRoutes.length > 0) {
      const wildcards = wildcardRoutes
        .map(({ model }) => model)
        .filter((model): model is DiscoveredModel => model !== undefined);
      const droppedRoutes = reducedGroups.filter(({ model }) => model === undefined).map(({ route }) => route);
      // Exact exclusions are bounded to the same public id: `/v1/models` lacks
      // deployment identity, so a differently named id for that deployment is unknowable.
      const droppedExactIds = new Set(droppedRoutes.filter((route) => !route.includes("*")));
      const droppedWildcards = droppedRoutes.filter((route) => route.includes("*"));
      // A wildcard row is not addressable. Remove it before expansion so a failed
      // `/v1/models` request cannot leak the literal wildcard into the selector.
      models = models.filter((model) => !model.id.includes("*"));
      progress?.("/model/info has wildcard entries, expanding via /v1/models...");
      const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
      if (listResult.ok && wildcards.length > 0) {
        const expanded = (listResult.data.data ?? [])
          .filter((entry) => {
            const id = wireString(entry.id);
            return (
              id === undefined ||
              (!droppedExactIds.has(id) && !droppedWildcards.some((route) => wildcardMatches(route, id)))
            );
          })
          .map((entry) => mapFromWildcardExpansion(entry, wildcards))
          .filter((model): model is DiscoveredModel => model !== undefined);
        const seen = new Set<string>(models.map((m) => m.id));
        models = [...models, ...expanded.filter((m) => !seen.has(m.id))];
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
  const models = (listResult.data.data ?? [])
    .map(mapFromModelsList)
    .filter((m): m is DiscoveredModel => m !== undefined);
  return { source: "models_list", models: deduplicateModels(models) };
}
