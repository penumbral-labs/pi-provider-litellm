import type { Api, Model } from "@earendil-works/pi-ai";
import type { DiscoveredModel, ModelInfoEntry } from "./types.js";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

export type SemanticFamily = "claude" | "deepseek" | "gemini" | "kimi" | "openai";

export type MessagesBackendCompat = Pick<
  NonNullable<Model<"anthropic-messages">["compat"]>,
  "forceAdaptiveThinking" | "supportsTemperature" | "supportsStrictTools"
>;

export interface CatalogResolution {
  provider?: string;
  // Concrete catalog identity; provider unanimity alone cannot authorize one
  // model's limits or pricing for a different model from the same provider.
  catalogModelId?: string;
  semanticFamily?: SemanticFamily;
  messagesCompat?: MessagesBackendCompat;
  messagesThinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  reasoning?: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision?: boolean;
  contextWindow?: number;
  maxTokens?: number;
  cost?: DiscoveredModel["cost"];
}

export type CatalogResolver = (entry: ModelInfoEntry) => CatalogResolution | undefined;

export interface ReducedModelGroup {
  id: string;
  api: "anthropic-messages" | "openai-completions" | "openai-responses";
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision: boolean;
  contextWindow: number;
  maxTokens: number;
  cost: DiscoveredModel["cost"];
  hasCompleteMetadata: boolean;
  catalogProvider?: string;
  semanticFamily?: SemanticFamily;
  messagesCompat?: MessagesBackendCompat;
  // Set when deployments disagreed on catalog provider identity, so catalog
  // limits, pricing, and reasoning metadata were withheld for the whole group.
  catalogAuthorityAmbiguous?: boolean;
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;
const CHAT_STYLE_MODE_PATTERN = /^chat$/i;
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite"] as const;
const REASONING_EFFORT_FLAGS = [
  ["off", "none", "supports_none_reasoning_effort"],
  ["minimal", "minimal", "supports_minimal_reasoning_effort"],
  ["low", "low", "supports_low_reasoning_effort"],
  ["medium", "medium", "supports_medium_reasoning_effort"],
  ["high", "high", "supports_high_reasoning_effort"],
  ["xhigh", "xhigh", "supports_xhigh_reasoning_effort"],
  ["max", "max", "supports_max_reasoning_effort"],
] as const;
const REASONING_LEVELS = REASONING_EFFORT_FLAGS.map(([level]) => level);
type CostField = (typeof COST_FIELDS)[number];
type ModelCost = DiscoveredModel["cost"];
type ModelCostTier = NonNullable<ModelCost["tiers"]>[number];

// `/model/info` is parsed JSON from operator-authored proxy config, so a field
// declared as a string can arrive as a number. Reading it as one must withhold
// that row's evidence, not throw and lose every model in the response.
export function wireString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

// An unreadable capability flag must not be read as `true`. `"false"` and `"no"` are
// both truthy, so coercing would relax a group guarantee — the one direction that
// matters here, since a capability is only advertised when every deployment agrees.
function wireBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function supportedEndpoints(value: unknown): Set<string> | undefined {
  if (!Array.isArray(value)) return undefined;
  return new Set(value.filter((endpoint): endpoint is string => typeof endpoint === "string"));
}

export function normalizedMode(mode: unknown): "chat" | "responses" | "unknown" | "unsupported" {
  if (mode == null) return "unknown";
  const value = wireString(mode)?.trim();
  // An unreadable mode is not evidence that the deployment is non-chat. Treating it
  // as "unsupported" would drop the row from the reduction and discard its limits,
  // relaxing the group; "unknown" keeps it routable and conservative.
  if (value === undefined) return "unknown";
  if (RESPONSES_MODE_PATTERN.test(value)) return "responses";
  if (CHAT_STYLE_MODE_PATTERN.test(value)) return "chat";
  return "unsupported";
}

export function hasMixedIncompatibleDeploymentModes(entries: readonly ModelInfoEntry[]): boolean {
  const modes = uniqueDeployments(entries.filter((entry) => wireString(entry.model_name))).map((entry) =>
    normalizedMode(entry.model_info?.mode),
  );
  return modes.includes("unsupported") && modes.some((mode) => mode !== "unsupported");
}

// Canonicalization is depth-bounded because deployment metadata is untrusted.
const MAX_CANONICAL_DEPTH = 12;

function sortValue(value: unknown, depth = 0): unknown {
  // Do not traverse the remaining untrusted subtree at the cap. Its contents
  // are deliberately outside the bounded canonical identity.
  if (depth >= MAX_CANONICAL_DEPTH) return "[depth-limit]";
  if (Array.isArray(value)) return value.map((child) => sortValue(child, depth + 1));
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, sortValue(child, depth + 1)]),
  );
}

// Key-order-independent identity, so logically equal catalog metadata compares
// equal instead of silently failing unanimity because of property order.
export function stableJson(value: unknown): string | undefined {
  return value === undefined ? undefined : JSON.stringify(sortValue(value));
}

function stableEntry(entry: ModelInfoEntry): string {
  return JSON.stringify(sortValue(entry));
}

// For rows with a deployment id, collapses only exact duplicates rather than all
// rows repeating that id. Id-less rows are never collapsed, even when identical,
// because there is no deployment identity proving that they describe one target.
// Conflicting variants therefore stay plural and fail closed.
function uniqueDeployments(entries: readonly ModelInfoEntry[]): ModelInfoEntry[] {
  const identified = new Map<string, Map<string, ModelInfoEntry>>();
  const anonymous: Array<{ signature: string; entry: ModelInfoEntry }> = [];
  for (const entry of entries) {
    const signature = stableEntry(entry);
    const id = wireString(entry.model_info?.id)?.trim();
    if (id) {
      const variants = identified.get(id) ?? new Map<string, ModelInfoEntry>();
      variants.set(signature, entry);
      identified.set(id, variants);
    } else {
      anonymous.push({ signature, entry });
    }
  }
  return [
    ...[...identified.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .flatMap(([, variants]) =>
        [...variants.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([, entry]) => entry),
      ),
    ...anonymous.sort((left, right) => left.signature.localeCompare(right.signature)).map(({ entry }) => entry),
  ];
}

// A router limit is usable only when it is a finite positive token count.
// Without this, one deployment reporting 0 would clamp the whole group to 0.
function explicitLimit(value: number | undefined): number | undefined {
  return value === undefined || !Number.isFinite(value) || value <= 0 ? undefined : value;
}

function routerThinkingLevelMap(entries: readonly ModelInfoEntry[]): DiscoveredModel["thinkingLevelMap"] {
  const map: NonNullable<DiscoveredModel["thinkingLevelMap"]> = {};
  for (const [level, effort, flag] of REASONING_EFFORT_FLAGS) {
    const reported = entries.map((entry) => entry.model_info?.[flag]);
    if (!reported.some((value) => value !== undefined)) continue;
    // Preserve upstream singleton behavior (including explicit false), while a
    // group advertises a router-reported effort only when every deployment says
    // true. Missing, false, or unreadable evidence suppresses that level.
    map[level] = reported.every((value) => wireBoolean(value) === true) ? effort : null;
  }
  return Object.keys(map).length > 0 ? map : undefined;
}

// A level named by any parent is inherited only when every matching parent
// supplies the same mapping. Missing, conflicting, and explicit false evidence
// all close that level for the expanded child.
export function intersectThinkingLevelMaps(
  maps: readonly DiscoveredModel["thinkingLevelMap"][],
): DiscoveredModel["thinkingLevelMap"] {
  if (maps.every((map) => map === undefined)) return undefined;
  const intersection: NonNullable<DiscoveredModel["thinkingLevelMap"]> = {};
  for (const level of REASONING_LEVELS) {
    const values = maps.map((map) => map?.[level]);
    if (values.every((value) => value === undefined)) continue;
    const first = values[0];
    intersection[level] = first !== undefined && values.every((value) => value === first) ? first : null;
  }
  return Object.keys(intersection).length > 0 ? intersection : undefined;
}

function explicitCost(entry: ModelInfoEntry, field: CostField): number | undefined {
  const info = entry.model_info;
  if (!info) return undefined;
  const perToken =
    field === "input"
      ? info.input_cost_per_token
      : field === "output"
        ? info.output_cost_per_token
        : field === "cacheRead"
          ? info.cache_read_input_token_cost
          : info.cache_creation_input_token_cost;
  return perToken === undefined || !Number.isFinite(perToken) || perToken < 0 ? undefined : perToken * 1_000_000;
}

function resolvedCost(
  entry: ModelInfoEntry,
  catalog: CatalogResolution | undefined,
  field: CostField,
): number | undefined {
  return explicitCost(entry, field) ?? catalog?.cost?.[field];
}

function min(values: readonly number[]): number {
  return Math.min(...values);
}

function conservativeLimit(explicit: number | undefined, catalog: number | undefined): number | undefined {
  const valid = [explicitLimit(explicit), explicitLimit(catalog)].filter(
    (value): value is number => value !== undefined,
  );
  return valid.length > 0 ? min(valid) : undefined;
}

function unanimous<T>(values: readonly (T | undefined)[]): T | undefined {
  const first = values[0];
  return first !== undefined && values.every((value) => value === first) ? first : undefined;
}

function ratesAboveThreshold(cost: ModelCost, threshold: number): ModelCost {
  let matchedThreshold = -1;
  let matches: ModelCostTier[] = [];
  for (const tier of cost.tiers ?? []) {
    if (tier.inputTokensAbove < 0 || tier.inputTokensAbove > threshold) continue;
    if (tier.inputTokensAbove > matchedThreshold) {
      matchedThreshold = tier.inputTokensAbove;
      matches = [tier];
    } else if (tier.inputTokensAbove === matchedThreshold) {
      // Pi uses the first duplicate threshold. Taking the per-field maximum is
      // conservative if malformed catalog data supplies conflicting duplicates.
      matches.push(tier);
    }
  }
  if (matches.length === 0) return cost;
  return {
    input: Math.max(cost.input, ...matches.map((tier) => tier.input)),
    output: Math.max(cost.output, ...matches.map((tier) => tier.output)),
    cacheRead: Math.max(cost.cacheRead, ...matches.map((tier) => tier.cacheRead)),
    cacheWrite: Math.max(cost.cacheWrite, ...matches.map((tier) => tier.cacheWrite)),
  };
}

// Builds the per-field upper envelope of complete request-wide price ladders.
// Every source threshold is retained because crossing it can change which
// deployment is most expensive, even when the ladders use different breakpoints.
export function conservativeCostTiers(costs: readonly ModelCost[]): ModelCost["tiers"] {
  const thresholds = [
    ...new Set(
      costs.flatMap((cost) =>
        (cost.tiers ?? [])
          .map((tier) => tier.inputTokensAbove)
          .filter((threshold) => Number.isFinite(threshold) && threshold >= 0),
      ),
    ),
  ].sort((left, right) => left - right);
  if (thresholds.length === 0) return undefined;

  return thresholds.map((inputTokensAbove) => {
    const rates = costs.map((cost) => ratesAboveThreshold(cost, inputTokensAbove));
    return {
      inputTokensAbove,
      input: Math.max(...rates.map((rate) => rate.input)),
      output: Math.max(...rates.map((rate) => rate.output)),
      cacheRead: Math.max(...rates.map((rate) => rate.cacheRead)),
      cacheWrite: Math.max(...rates.map((rate) => rate.cacheWrite)),
    };
  });
}

export function reduceModelGroup(
  entries: readonly ModelInfoEntry[],
  resolveCatalog: CatalogResolver,
): ReducedModelGroup | undefined {
  // A group is addressed by its public route name, so a row without a readable one
  // cannot participate. Enforced here rather than at each caller, so no caller can
  // leak a non-string id into a discovered model.
  const candidates = uniqueDeployments(entries.filter((entry) => wireString(entry.model_name)));
  if (candidates.length === 0) return undefined;
  // Every deployment behind a public route must accept a chat-style request.
  // Dropping an explicitly incompatible sibling would publish a route that can
  // still be selected for an embedding or other non-chat deployment.
  const candidateModes = candidates.map((entry) => normalizedMode(entry.model_info?.mode));
  if (candidateModes.includes("unsupported")) return undefined;
  const candidateEndpoints = candidates.map((entry) => supportedEndpoints(entry.model_info?.supported_endpoints));
  const deployments = candidates;
  const catalogs = deployments.map((entry) => resolveCatalog(entry));
  const catalogProvider = unanimous(catalogs.map((catalog) => catalog?.provider));
  const catalogModelIds = catalogs.map((catalog) => catalog?.catalogModelId);
  const hasCatalogModelIdentity = catalogModelIds.some((id) => id !== undefined);
  const catalogModelId = unanimous(catalogModelIds);
  const semanticFamily = unanimous(catalogs.map((catalog) => catalog?.semanticFamily));
  const messagesCompat = unanimous(catalogs.map((catalog) => stableJson(catalog?.messagesCompat)));
  // Provider-only resolver fixtures preserve the pre-existing reducer contract;
  // production catalog resolutions always carry a concrete model identity.
  const hasCatalogAuthority =
    catalogProvider !== undefined && (!hasCatalogModelIdentity || catalogModelId !== undefined);
  const catalogAuthority = hasCatalogAuthority ? catalogs : catalogs.map(() => undefined);
  const catalogAuthorityAmbiguous =
    !hasCatalogAuthority &&
    catalogs.some((catalog) => catalog?.provider !== undefined || catalog?.catalogModelId !== undefined);
  const reasoningValues = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_reasoning) ?? catalogAuthority[index]?.reasoning,
  );
  const visionValues = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_vision) ?? catalogAuthority[index]?.vision,
  );
  const contextWindowValues = deployments.map((entry, index) =>
    conservativeLimit(entry.model_info?.max_input_tokens, catalogAuthority[index]?.contextWindow),
  );
  const maxTokensValues = deployments.map((entry, index) =>
    conservativeLimit(entry.model_info?.max_output_tokens, catalogAuthority[index]?.maxTokens),
  );
  const reasoning = reasoningValues.every((value) => value ?? false);
  const vision = visionValues.every((value) => value ?? false);
  const contextWindow = min(contextWindowValues.map((value) => value ?? DEFAULT_CONTEXT_WINDOW));
  const maxTokens = min(maxTokensValues.map((value) => value ?? DEFAULT_MAX_TOKENS));

  const costValues = COST_FIELDS.map((field) =>
    deployments.map((entry, index) => resolvedCost(entry, catalogAuthority[index], field)),
  );
  const completeCostFields = costValues.map((values) => values.every((value) => value !== undefined));
  const hasCompleteCost = completeCostFields.every(Boolean);
  const hasCompleteMetadata =
    hasCompleteCost &&
    reasoningValues.every((value) => value !== undefined) &&
    visionValues.every((value) => value !== undefined) &&
    contextWindowValues.every((value) => value !== undefined) &&
    maxTokensValues.every((value) => value !== undefined);
  const cost: DiscoveredModel["cost"] = {
    input: completeCostFields[0] ? Math.max(...(costValues[0] as number[])) : 0,
    output: completeCostFields[1] ? Math.max(...(costValues[1] as number[])) : 0,
    cacheRead: completeCostFields[2] ? Math.max(...(costValues[2] as number[])) : 0,
    cacheWrite: completeCostFields[3] ? Math.max(...(costValues[3] as number[])) : 0,
  };
  if (hasCompleteCost && hasCatalogAuthority) {
    const deploymentCosts = deployments.map((entry, index) => {
      const baseCost = {
        input: costValues[0][index] as number,
        output: costValues[1][index] as number,
        cacheRead: costValues[2][index] as number,
        cacheWrite: costValues[3][index] as number,
      };
      const explicitFields = new Set(COST_FIELDS.filter((field) => explicitCost(entry, field) !== undefined));
      const catalogTiers = catalogAuthority[index]?.cost?.tiers;
      // An explicit field replaces catalog pricing for that field at every
      // threshold. Unaffected fields retain their catalog ladder; otherwise a
      // partial router override could hide a known higher catalog rate.
      const tiers =
        catalogTiers && explicitFields.size < COST_FIELDS.length
          ? catalogTiers.map((tier) => ({
              inputTokensAbove: tier.inputTokensAbove,
              input: explicitFields.has("input") ? baseCost.input : tier.input,
              output: explicitFields.has("output") ? baseCost.output : tier.output,
              cacheRead: explicitFields.has("cacheRead") ? baseCost.cacheRead : tier.cacheRead,
              cacheWrite: explicitFields.has("cacheWrite") ? baseCost.cacheWrite : tier.cacheWrite,
            }))
          : undefined;
      return { ...baseCost, ...(tiers ? { tiers } : {}) };
    });
    const tiers = conservativeCostTiers(deploymentCosts);
    if (tiers) cost.tiers = tiers;
  }
  const catalogThinkingLevelMap = catalogProvider
    ? unanimous(catalogAuthority.map((catalog) => stableJson(catalog?.thinkingLevelMap)))
    : messagesCompat && !catalogAuthorityAmbiguous
      ? unanimous(catalogs.map((catalog) => stableJson(catalog?.messagesThinkingLevelMap)))
      : undefined;
  const parsedCatalogThinkingLevelMap = catalogThinkingLevelMap ? JSON.parse(catalogThinkingLevelMap) : undefined;
  const routerMap = routerThinkingLevelMap(deployments);

  const id = wireString(deployments[0]?.model_name);
  if (id === undefined) return undefined;

  const messagesEndpointAllowed = candidateEndpoints.every(
    (endpoints) => endpoints === undefined || endpoints.has("/v1/messages"),
  );
  const api = candidateModes.every((mode) => mode === "responses")
    ? "openai-responses"
    : candidateModes.every((mode) => mode === "chat") &&
        messagesEndpointAllowed &&
        semanticFamily === "claude" &&
        messagesCompat
      ? "anthropic-messages"
      : "openai-completions";
  let thinkingLevelMap = parsedCatalogThinkingLevelMap;
  if (api !== "anthropic-messages") {
    thinkingLevelMap =
      parsedCatalogThinkingLevelMap || routerMap ? { ...parsedCatalogThinkingLevelMap, ...routerMap } : undefined;
  } else if (parsedCatalogThinkingLevelMap && routerMap) {
    // LiteLLM's supports_*_reasoning_effort fields describe its OpenAI-compatible
    // surface. They may restrict a catalogued Messages effort, but must not add or
    // rename Anthropic effort values that the backend catalog does not authorize.
    thinkingLevelMap = Object.fromEntries(
      Object.entries(parsedCatalogThinkingLevelMap).map(([level, effort]) => [
        level,
        routerMap[level as keyof typeof routerMap] === null ? null : effort,
      ]),
    );
  }

  return {
    id,
    api,
    reasoning,
    ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
    vision,
    contextWindow,
    maxTokens,
    cost,
    hasCompleteMetadata,
    ...(hasCatalogAuthority && catalogProvider ? { catalogProvider } : {}),
    ...(semanticFamily ? { semanticFamily } : {}),
    ...(messagesCompat ? { messagesCompat: JSON.parse(messagesCompat) } : {}),
    ...(catalogAuthorityAmbiguous ? { catalogAuthorityAmbiguous: true } : {}),
  };
}

export function catalogResolution(
  provider: string,
  semanticFamily: SemanticFamily | undefined,
  model: Model<Api>,
): CatalogResolution {
  return {
    provider,
    catalogModelId: model.id,
    semanticFamily,
    reasoning: model.reasoning,
    thinkingLevelMap: model.thinkingLevelMap,
    vision: model.input.includes("image"),
    contextWindow: model.contextWindow,
    maxTokens: model.maxTokens,
    cost: model.cost,
  };
}
