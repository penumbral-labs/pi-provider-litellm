import type { Api, Model } from "@earendil-works/pi-ai";
import type { DiscoveredModel, ModelInfoEntry } from "./types.js";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

export type SemanticFamily = "claude" | "deepseek" | "gemini" | "kimi" | "openai";
export type SemanticModel = "deepseek-v4" | "kimi-k2.5-k2.6" | "kimi-k2.7-code" | "kimi-k3";

// Deployment rows can disagree about which backend family serves a route. That
// disagreement is evidence in its own right and must not decay into "no
// evidence", which would re-enable route-name inference.
export type FamilyEvidence = SemanticFamily | "conflicting";

export interface BackendIdentityEvidence {
  semanticFamily: "claude";
}

// The Anthropic compatibility fields that can be carried from the bundled
// catalog onto a LiteLLM-routed native Messages model.
export type MessagesBackendCompat = Pick<
  NonNullable<Model<"anthropic-messages">["compat"]>,
  "forceAdaptiveThinking" | "supportsTemperature" | "supportsStrictTools"
>;

type OpenAICompat = NonNullable<Model<"openai-completions">["compat"]>;

export interface ReasoningPolicy {
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  compat?: Pick<
    OpenAICompat,
    "requiresReasoningContentOnAssistantMessages" | "supportsReasoningEffort" | "thinkingFormat"
  >;
}

export interface CatalogResolution {
  provider?: string;
  semanticFamily?: FamilyEvidence;
  semanticModel?: SemanticModel;
  backendIdentity?: BackendIdentityEvidence;
  // Absent means the backend model is unknown to the catalog, not that it has no requirements.
  messagesCompat?: MessagesBackendCompat;
  // A singleton route-name catalog hint may supply bounded metadata, but it is
  // not deployment evidence and therefore cannot authorize outbound repairs.
  routeNameOnly?: boolean;
  reasoning?: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision?: boolean;
  contextWindow?: number;
  maxTokens?: number;
  cost?: DiscoveredModel["cost"];
}

// `singleton` is true only when the group reduces to exactly one routable
// deployment, which is the only case where a public route name may be used as a
// catalog hint. Resolvers must not treat route text as evidence otherwise.
export type CatalogResolver = (entry: ModelInfoEntry, singleton: boolean) => CatalogResolution | undefined;

export interface ReducedModelGroup {
  id: string;
  deploymentCount: number;
  api: "anthropic-messages" | "openai-completions" | "openai-responses";
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision: boolean;
  contextWindow: number;
  maxTokens: number;
  cost: DiscoveredModel["cost"];
  hasCompleteCost: boolean;
  catalogProvider?: string;
  semanticFamily?: FamilyEvidence;
  semanticModel?: SemanticModel;
  messagesCompat?: MessagesBackendCompat;
  // Set when deployments disagreed on catalog provider identity, so catalog
  // limits, pricing, and reasoning metadata were withheld for the whole group.
  catalogAuthorityAmbiguous?: boolean;
  // Family each routable deployment identified, positionally, with `undefined`
  // for a deployment that identified none. The per-field meet needs to know that
  // a candidate is unlabeled — not merely that the group disagreed.
  deploymentFamilies: (SemanticFamily | undefined)[];
  acceptedOpenAIParams: string[];
  reasoningPolicy: ReasoningPolicy;
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;
const CHAT_STYLE_MODE_PATTERN = /^chat$/i;
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite"] as const;
type CostField = (typeof COST_FIELDS)[number];

// `/model/info` is parsed JSON from operator-authored proxy config, so a field
// declared as a string can arrive as a number. Reading it as one must withhold
// that row's evidence, not throw and lose every model in the response.
export function wireString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

// Unreadable capability flags are withheld rather than coerced. In particular,
// string values such as `"false"` must not become truthy capability evidence.
function wireBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

export function isResponsesMode(mode: unknown): boolean {
  const value = wireString(mode);
  return value !== undefined && RESPONSES_MODE_PATTERN.test(value.trim());
}

function normalizedMode(mode: unknown): "chat" | "responses" | "unknown" | "unsupported" {
  if (mode == null) return "unknown";
  const value = wireString(mode)?.trim();
  // An unreadable mode remains in the conservative reduction instead of being
  // filtered as unsupported and silently relaxing the remaining group.
  if (value === undefined) return "unknown";
  if (RESPONSES_MODE_PATTERN.test(value)) return "responses";
  if (CHAT_STYLE_MODE_PATTERN.test(value)) return "chat";
  return "unsupported";
}

// Canonicalization is depth-bounded because deployment metadata is untrusted.
const MAX_CANONICAL_DEPTH = 12;

function sortValue(value: unknown, depth = 0): unknown {
  if (depth >= MAX_CANONICAL_DEPTH) return JSON.stringify(value);
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
function stableJson(value: unknown): string | undefined {
  return value === undefined ? undefined : JSON.stringify(sortValue(value));
}

function stableEntry(entry: ModelInfoEntry): string {
  return JSON.stringify(sortValue(entry));
}

// Exact identified duplicates collapse, conflicting variants sharing an id stay
// plural, and anonymous rows stay distinct because no identity proves equality.
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
  // Conflicting rows for one deployment remain in the reduction so their
  // disagreement fails closed, while exact repeats stay idempotent.
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

function normalizeParams(params: unknown): Set<string> {
  if (!Array.isArray(params)) return new Set();
  const named = params.map((param) => wireString(param)?.trim()).filter((param) => Boolean(param));
  return new Set(named as string[]);
}

function acceptedParams(entry: ModelInfoEntry): Set<string> {
  const params = normalizeParams(entry.model_info?.supported_openai_params);
  for (const param of normalizeParams(entry.litellm_params?.allowed_openai_params)) params.add(param);
  return params;
}

function intersectParams(entries: readonly ModelInfoEntry[]): string[] {
  const [first, ...rest] = entries.map(acceptedParams);
  if (!first) return [];
  return [...first].filter((param) => rest.every((params) => params.has(param))).sort();
}

const ONLY_HIGH = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: "high",
  xhigh: null,
  max: null,
} as const;

// pi-ai reads an ABSENT thinkingLevelMap as "every standard level supported"
// (getSupportedThinkingLevels), so a policy that cannot transmit any level has
// to deny each one explicitly instead of omitting the map.
export const NO_TRANSMISSIBLE_LEVELS = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: null,
  xhigh: null,
  max: null,
} as const;

const EXTENDED_LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const;

// Efforts the Responses API accepts. pi-ai passes an unmapped level through
// verbatim and reads `thinkingLevelMap.off` as the disable value, so a Chat-shaped
// map would emit `off` or `max` as an effort. `none` is the disable spelling —
// pi-ai's own `openai/gpt-5.5` entry maps `off` to it.
const RESPONSES_EFFORTS = new Set(["none", "minimal", "low", "medium", "high", "xhigh"]);

// A level map is only meaningful next to the compat that serializes it, and the
// two used to travel separately: five call sites each decided whether to copy
// one, the other, or both. `SerializerPolicy` closes them into one value so a
// consumer cannot take a level map without the conclusion that carries it.
export interface SerializerPolicy {
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  compat: DiscoveredModel["compat"];
}

// Chat carries a level through `reasoning_effort` or an explicit `thinkingFormat`.
// pi-ai would default effort on for a litellm provider, but a closed policy must
// state its own carrier rather than inherit one, so this reports only explicit
// evidence and the caller sets the carrier it concluded.
function chatCarrier(compat: OpenAICompat | undefined): boolean {
  return compat?.thinkingFormat !== undefined || compat?.supportsReasoningEffort === true;
}

function deniesChatEffort(compat: OpenAICompat | undefined): boolean {
  return compat?.supportsReasoningEffort === false && compat?.thinkingFormat === undefined;
}

// Translates a Chat-shaped map into Responses efforts, denying any level with no
// valid Responses value instead of letting it through verbatim. pi-ai treats an
// absent xhigh/max entry as unsupported (unlike the standard levels), so the
// translation must preserve that distinction rather than widening the map.
export function toResponsesLevels(
  levels: DiscoveredModel["thinkingLevelMap"] | undefined,
): NonNullable<DiscoveredModel["thinkingLevelMap"]> {
  const translated: Record<string, string | null> = {};
  for (const level of EXTENDED_LEVELS) {
    const chat = levels?.[level];
    if (level === "off") {
      // Disable is `none` on this API, never `off`.
      translated.off = chat === null ? null : "none";
      continue;
    }
    if (chat === undefined && (level === "xhigh" || level === "max")) {
      translated[level] = null;
      continue;
    }
    // pi-ai passes an absent standard level through as its own name.
    const candidate = chat === undefined ? level : chat;
    translated[level] = candidate !== null && RESPONSES_EFFORTS.has(candidate) ? candidate : null;
  }
  return translated as NonNullable<DiscoveredModel["thinkingLevelMap"]>;
}

// Fields whose conservative direction is to APPLY them: each one removes or
// narrows something we would otherwise send, and the resulting request is
// accepted by any OpenAI-compatible backend. Applying one because a single
// deployment needs it therefore cannot break an unlabeled sibling.
const MEET_APPLY_IF_ANY = [
  "supportsStore",
  "supportsDeveloperRole",
  "supportsReasoningEffort",
  "supportsStrictMode",
] as const;

// Fields that change the request shape or add a capability. Applying one because
// a sibling needs it can break the other candidate — `max_tokens` is rejected by
// newer OpenAI models in favour of `max_completion_tokens`, and `cache_control`
// markers are only valid for Anthropic — so these need every candidate to agree
// and are otherwise withheld explicitly.
const MEET_REQUIRE_UNANIMITY = [
  "maxTokensField",
  "cacheControlFormat",
  "thinkingFormat",
  "requiresReasoningContentOnAssistantMessages",
] as const;

// The conservative meet of per-deployment vendor conclusions, field by field. A
// blanket union would copy every vendor flag as soon as one deployment named a
// vendor, which is unsafe: not every restrictive-looking flag is monotone. A
// blanket withhold would drop safety restrictions a deployment demonstrably
// needs. Neither is right, so each field carries its own direction and anything
// without a demonstrably safe common value is withheld rather than inferred.
export function meetVendorCompat(
  perDeployment: readonly (DiscoveredModel["compat"] | undefined)[],
): DiscoveredModel["compat"] {
  const candidates = perDeployment.map((compat) => compat as OpenAICompat | undefined);
  if (candidates.length === 0) return undefined;
  const met: Record<string, unknown> = {};
  for (const field of MEET_APPLY_IF_ANY) {
    const stated = candidates.map((compat) => compat?.[field]).filter((value) => value !== undefined);
    // Values within this set are all `false`; a disagreement would mean two
    // vendors want opposite narrowing, which has no safe common value.
    if (stated.length > 0 && new Set(stated).size === 1) met[field] = stated[0];
  }
  for (const field of MEET_REQUIRE_UNANIMITY) {
    const stated = candidates.map((compat) => compat?.[field]);
    if (stated.every((value) => value !== undefined) && new Set(stated).size === 1) met[field] = stated[0];
  }
  return (Object.keys(met).length > 0 ? met : undefined) as DiscoveredModel["compat"];
}

// The one place a level map is paired with a serializer conclusion. Every
// discovery, cache, health and singleton path consumes the whole returned object.
export function closeSerializerPolicy(input: {
  api: "openai-completions" | "openai-responses";
  reasoning: boolean;
  vendorCompat: DiscoveredModel["compat"];
  semanticCompat?: ReasoningPolicy["compat"];
  semanticLevels?: DiscoveredModel["thinkingLevelMap"];
  catalogLevels?: DiscoveredModel["thinkingLevelMap"];
  // For callers with no level evidence at all. Distinct from "no candidate map":
  // this states the no-level conclusion explicitly, so a caller spreading the
  // policy over an earlier model cannot leave a stale map behind.
  denyLevels?: boolean;
}): SerializerPolicy {
  const { api, reasoning, vendorCompat, semanticCompat, semanticLevels, catalogLevels, denyLevels } = input;
  if (api === "openai-responses") {
    // Chat compat fields are not part of the Responses compat union, so only the
    // shared request-shape fields travel. The vendor's explicit effort denial is
    // still authoritative even though it cannot be copied into Responses compat:
    // no level may be offered for that API when the backend rejected the carrier.
    const shared = vendorCompat as OpenAICompat | undefined;
    const compat: DiscoveredModel["compat"] = {
      ...(shared?.supportsDeveloperRole === false ? { supportsDeveloperRole: false } : {}),
      ...(shared?.supportsStrictMode === false ? { supportsStrictMode: false } : {}),
    };
    if (!reasoning) return { reasoning, compat };
    if (denyLevels || shared?.supportsReasoningEffort === false) {
      return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat };
    }
    return { reasoning, thinkingLevelMap: toResponsesLevels(semanticLevels ?? catalogLevels), compat };
  }
  // A model with no compat at all keeps none: fabricating an empty object here
  // would rewrite every cached model that passes through.
  const stated = vendorCompat !== undefined || semanticCompat !== undefined;
  const merged = { ...(vendorCompat as OpenAICompat), ...semanticCompat } as OpenAICompat;
  const compat = (stated ? merged : undefined) as DiscoveredModel["compat"];
  if (!reasoning) return { reasoning, compat };
  if (denyLevels) return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat };
  if (deniesChatEffort(merged)) {
    // The vendor denies effort and named no format: nothing can carry a level.
    return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat };
  }
  const candidateLevels = semanticLevels ?? catalogLevels;
  // No candidate map: pi-ai offers the standard levels and serializes them through
  // `reasoning_effort`, which this compat permits, so there is nothing to close.
  if (candidateLevels === undefined) return { reasoning, compat };
  if (chatCarrier(merged)) return { reasoning, thinkingLevelMap: candidateLevels, compat };
  // Advertising a specific map without an explicit carrier would lean on pi-ai's
  // default for a litellm provider; state the carrier this policy concluded.
  return {
    reasoning,
    thinkingLevelMap: candidateLevels,
    compat: { ...merged, supportsReasoningEffort: true } as DiscoveredModel["compat"],
  };
}

function buildReasoningPolicy(
  semanticModel: SemanticModel | undefined,
  acceptedOpenAIParams: readonly string[],
  reducedReasoning: boolean,
  explicitlyUnsupported: boolean,
): ReasoningPolicy {
  const acceptsThinking = acceptedOpenAIParams.includes("thinking");
  const acceptsEffort = acceptedOpenAIParams.includes("reasoning_effort");
  const hasAcceptedControl = acceptsThinking || acceptsEffort;
  const reasoning =
    !explicitlyUnsupported && (reducedReasoning || hasAcceptedControl || semanticModel === "kimi-k2.7-code");
  if (semanticModel === "kimi-k2.5-k2.6") {
    // Binary thinking rides the `thinking` param, so without that accepted
    // param there is no wire mechanism and every level must be denied.
    if (acceptsThinking && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: { ...ONLY_HIGH, off: "off" },
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
      };
    }
    return {
      reasoning,
      ...(reasoning ? { thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS } : {}),
      compat: { supportsReasoningEffort: false },
    };
  }
  if (semanticModel === "kimi-k2.7-code") {
    // K2.7 Code always reasons and cannot be switched off, so `off` stays
    // denied rather than inventing a disable control. The `high` level only
    // exists when a deployment accepts `thinking` to carry it.
    const replay = { requiresReasoningContentOnAssistantMessages: true } as const;
    if (acceptsThinking && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: ONLY_HIGH,
        compat: { ...replay, thinkingFormat: "deepseek", supportsReasoningEffort: false },
      };
    }
    return {
      reasoning,
      ...(reasoning ? { thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS } : {}),
      compat: { ...replay, supportsReasoningEffort: false },
    };
  }
  if (semanticModel === "kimi-k3") {
    const replay = { requiresReasoningContentOnAssistantMessages: true } as const;
    if (acceptsEffort && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: "low",
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: { ...replay, thinkingFormat: "openai", supportsReasoningEffort: true },
      };
    }
    return {
      reasoning,
      ...(reasoning ? { thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS } : {}),
      compat: { ...replay, supportsReasoningEffort: false },
    };
  }
  if (semanticModel === "deepseek-v4") {
    const replay = { requiresReasoningContentOnAssistantMessages: true } as const;
    if (acceptsThinking && acceptsEffort && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: {
          off: "off",
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: { ...replay, thinkingFormat: "deepseek", supportsReasoningEffort: true },
      };
    }
    if (acceptsEffort && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: { ...replay, thinkingFormat: "openai", supportsReasoningEffort: true },
      };
    }
    if (acceptsThinking && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: { ...ONLY_HIGH, off: "off" },
        compat: { ...replay, thinkingFormat: "deepseek", supportsReasoningEffort: false },
      };
    }
    return {
      reasoning,
      ...(reasoning ? { thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS } : {}),
      compat: { ...replay, supportsReasoningEffort: false },
    };
  }
  // No identified semantic model means no known reasoning contract; the caller
  // ignores this policy entirely and keeps the reduced/catalog metadata.
  return { reasoning: false };
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
  return perToken === undefined || !Number.isFinite(perToken) ? undefined : perToken * 1_000_000;
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

function unanimous<T>(values: readonly (T | undefined)[]): T | undefined {
  const first = values[0];
  return first !== undefined && values.every((value) => value === first) ? first : undefined;
}

// Deployments that name different backend families, a single deployment whose
// routing and base models disagree, or a group where only SOME deployments
// declare a family, all leave the group with contradictory rather than missing
// evidence. Reporting that distinctly stops the caller from treating the route
// name as a fallback signal: partial evidence is still evidence that the route
// is not uniformly what its name suggests.
function reduceFamilyEvidence(values: readonly (FamilyEvidence | undefined)[]): FamilyEvidence | undefined {
  const declared = values.filter((value): value is FamilyEvidence => value !== undefined);
  if (declared.length === 0) return undefined;
  if (declared.length !== values.length) return "conflicting";
  const distinct = new Set(declared);
  if (distinct.has("conflicting") || distinct.size > 1) return "conflicting";
  return declared[0];
}

export function reduceModelGroup(
  entries: readonly ModelInfoEntry[],
  resolveCatalog: CatalogResolver,
): ReducedModelGroup | undefined {
  // A route is addressable only by a readable public name. Keeping this guard in
  // the reducer prevents any caller from leaking malformed wire data into ids.
  const candidates = uniqueDeployments(entries.filter((entry) => wireString(entry.model_name)));
  if (candidates.length === 0) return undefined;
  // Transport votes over every candidate row, so an unsupported sibling still
  // forces Chat; capability, limit, price, and identity evidence reduces only
  // over routable rows so a non-chat sibling cannot corrupt them.
  const candidateModes = candidates.map((entry) => normalizedMode(entry.model_info?.mode));
  const deployments = candidates.filter((_, index) => candidateModes[index] !== "unsupported");
  if (deployments.length === 0) return undefined;
  const singleton = deployments.length === 1;
  const catalogs = deployments.map((entry) => resolveCatalog(entry, singleton));
  const catalogProvider = unanimous(catalogs.map((catalog) => catalog?.provider));
  const semanticFamily = reduceFamilyEvidence(catalogs.map((catalog) => catalog?.semanticFamily));
  const semanticModel = unanimous(catalogs.map((catalog) => catalog?.semanticModel));
  const backendSemanticFamily = unanimous(catalogs.map((catalog) => catalog?.backendIdentity?.semanticFamily));
  // Compatibility can remain authoritative when provider identity, pricing, and
  // limits are withheld. Compare canonically so property order cannot break unanimity.
  const messagesCompat = unanimous(catalogs.map((catalog) => stableJson(catalog?.messagesCompat)));
  const catalogAuthority = catalogProvider ? catalogs : catalogs.map(() => undefined);
  const catalogAuthorityAmbiguous =
    catalogProvider === undefined && catalogs.some((catalog) => catalog?.provider !== undefined);
  const reasoning = deployments.every(
    (entry, index) => wireBoolean(entry.model_info?.supports_reasoning) ?? catalogAuthority[index]?.reasoning ?? false,
  );
  const explicitlyUnsupported = deployments.some(
    (entry) => wireBoolean(entry.model_info?.supports_reasoning) === false,
  );
  const vision = deployments.every(
    (entry, index) => wireBoolean(entry.model_info?.supports_vision) ?? catalogAuthority[index]?.vision ?? false,
  );
  const contextWindow = min(
    deployments.map(
      (entry, index) =>
        explicitLimit(entry.model_info?.max_input_tokens) ??
        explicitLimit(catalogAuthority[index]?.contextWindow) ??
        DEFAULT_CONTEXT_WINDOW,
    ),
  );
  const maxTokens = min(
    deployments.map(
      (entry, index) =>
        explicitLimit(entry.model_info?.max_output_tokens) ??
        explicitLimit(catalogAuthority[index]?.maxTokens) ??
        DEFAULT_MAX_TOKENS,
    ),
  );

  const costValues = COST_FIELDS.map((field) =>
    deployments.map((entry, index) => resolvedCost(entry, catalogAuthority[index], field)),
  );
  const completeCostFields = costValues.map((values) => values.every((value) => value !== undefined));
  const hasCompleteCost = completeCostFields.every(Boolean);
  const cost: DiscoveredModel["cost"] = {
    input: completeCostFields[0] ? Math.max(...(costValues[0] as number[])) : 0,
    output: completeCostFields[1] ? Math.max(...(costValues[1] as number[])) : 0,
    cacheRead: completeCostFields[2] ? Math.max(...(costValues[2] as number[])) : 0,
    cacheWrite: completeCostFields[3] ? Math.max(...(costValues[3] as number[])) : 0,
  };
  if (hasCompleteCost && catalogProvider) {
    const tiers = unanimous(catalogAuthority.map((catalog) => stableJson(catalog?.cost?.tiers)));
    if (tiers) cost.tiers = JSON.parse(tiers);
  }
  const thinkingLevelMap = catalogProvider
    ? unanimous(catalogAuthority.map((catalog) => stableJson(catalog?.thinkingLevelMap)))
    : undefined;
  const acceptedOpenAIParams = intersectParams(deployments);

  const id = wireString(deployments[0]?.model_name);
  if (id === undefined) return undefined;

  const api = candidateModes.every((mode) => mode === "responses")
    ? "openai-responses"
    : candidateModes.every((mode) => mode === "chat") && backendSemanticFamily === "claude" && messagesCompat
      ? "anthropic-messages"
      : "openai-completions";

  return {
    id,
    deploymentCount: deployments.length,
    api,
    reasoning,
    ...(thinkingLevelMap ? { thinkingLevelMap: JSON.parse(thinkingLevelMap) } : {}),
    vision,
    contextWindow,
    maxTokens,
    cost,
    hasCompleteCost,
    ...(catalogProvider ? { catalogProvider } : {}),
    ...(semanticFamily ? { semanticFamily } : {}),
    ...(semanticModel ? { semanticModel } : {}),
    ...(messagesCompat ? { messagesCompat: JSON.parse(messagesCompat) } : {}),
    ...(catalogAuthorityAmbiguous ? { catalogAuthorityAmbiguous: true } : {}),
    deploymentFamilies: catalogs.map((catalog) =>
      catalog?.routeNameOnly || catalog?.semanticFamily === "conflicting" ? undefined : catalog?.semanticFamily,
    ),
    acceptedOpenAIParams,
    reasoningPolicy: buildReasoningPolicy(semanticModel, acceptedOpenAIParams, reasoning, explicitlyUnsupported),
  };
}

export function catalogResolution(
  provider: string,
  semanticFamily: FamilyEvidence | undefined,
  model: Model<Api>,
): CatalogResolution {
  return {
    provider,
    semanticFamily,
    reasoning: model.reasoning,
    thinkingLevelMap: model.thinkingLevelMap,
    vision: model.input.includes("image"),
    contextWindow: model.contextWindow,
    maxTokens: model.maxTokens,
    cost: model.cost,
  };
}
