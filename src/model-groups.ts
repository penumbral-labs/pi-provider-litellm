import type { Api, Model } from "@earendil-works/pi-ai";
import type { DiscoveredModel, ModelInfoEntry } from "./types.js";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

export type SemanticFamily = "claude" | "deepseek" | "gemini" | "kimi" | "openai";

export interface BackendIdentityEvidence {
  semanticFamily: "claude";
}

// The Anthropic compatibility fields this extension is willing to carry from the
// bundled catalog onto a LiteLLM-routed native Messages model. Discovered `compat`
// is the only channel through which per-model Anthropic behavior reaches pi-ai's
// serializer, so these travel as a unit rather than field by field.
export type MessagesBackendCompat = Pick<
  NonNullable<Model<"anthropic-messages">["compat"]>,
  "forceAdaptiveThinking" | "supportsTemperature" | "supportsStrictTools"
>;

export interface CatalogResolution {
  provider?: string;
  semanticFamily?: SemanticFamily;
  backendIdentity?: BackendIdentityEvidence;
  // Absent means the backend model is unknown to the catalog, not that it has no requirements.
  messagesCompat?: MessagesBackendCompat;
  reasoning?: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision?: boolean;
  contextWindow?: number;
  maxTokens?: number;
  cost?: DiscoveredModel["cost"];
}

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
  semanticFamily?: SemanticFamily;
  messagesCompat?: MessagesBackendCompat;
  catalogAuthorityAmbiguous?: boolean;
  acceptedOpenAIParams: string[];
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;
const CHAT_STYLE_MODE_PATTERN = /^chat$/i;
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite"] as const;
type CostField = (typeof COST_FIELDS)[number];

export function wireString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function wireBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function normalizedMode(mode: unknown): "chat" | "responses" | "unknown" | "unsupported" {
  if (mode == null) return "unknown";
  const value = wireString(mode)?.trim();
  if (value === undefined) return "unknown";
  if (RESPONSES_MODE_PATTERN.test(value)) return "responses";
  if (CHAT_STYLE_MODE_PATTERN.test(value)) return "chat";
  return "unsupported";
}

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

function stableJson(value: unknown): string | undefined {
  return value === undefined ? undefined : JSON.stringify(sortValue(value));
}

function stableEntry(entry: ModelInfoEntry): string {
  return JSON.stringify(sortValue(entry));
}

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

function explicitLimit(value: number | undefined): number | undefined {
  return value === undefined || !Number.isFinite(value) || value <= 0 ? undefined : value;
}

function normalizeParams(params: unknown): Set<string> {
  if (!Array.isArray(params)) return new Set();
  return new Set(params.map((param) => wireString(param)?.trim()).filter((param): param is string => Boolean(param)));
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

export function reduceModelGroup(
  entries: readonly ModelInfoEntry[],
  resolveCatalog: CatalogResolver,
): ReducedModelGroup | undefined {
  const candidates = uniqueDeployments(entries.filter((entry) => wireString(entry.model_name)));
  if (candidates.length === 0) return undefined;
  const candidateModes = candidates.map((entry) => normalizedMode(entry.model_info?.mode));
  const deployments = candidates.filter((_, index) => candidateModes[index] !== "unsupported");
  if (deployments.length === 0) return undefined;
  const catalogs = deployments.map((entry) => resolveCatalog(entry, deployments.length === 1));
  const catalogProvider = unanimous(catalogs.map((catalog) => catalog?.provider));
  const semanticFamily = unanimous(catalogs.map((catalog) => catalog?.semanticFamily));
  const backendSemanticFamily = unanimous(catalogs.map((catalog) => catalog?.backendIdentity?.semanticFamily));
  // Backend compatibility is not gated on `catalogProvider`: a deployment can be
  // identifiable enough to require adaptive thinking while remaining too opaque to
  // borrow pricing or context metadata from. Compared by stable key order so an
  // equal policy expressed in a different order still counts as agreement.
  const messagesCompat = unanimous(catalogs.map((catalog) => stableJson(catalog?.messagesCompat)));
  const catalogAuthority = catalogProvider ? catalogs : catalogs.map(() => undefined);
  const catalogAuthorityAmbiguous =
    catalogProvider === undefined && catalogs.some((catalog) => catalog?.provider !== undefined);
  const reasoning = deployments.every(
    (entry, index) => wireBoolean(entry.model_info?.supports_reasoning) ?? catalogAuthority[index]?.reasoning ?? false,
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

  // Native Messages needs positive, unanimous compatibility evidence, not just a
  // unanimous Claude family. `messagesCompat` is undefined when any deployment's
  // backend is unknown to the catalog or when deployments disagree (for example a
  // mixed-generation group where one member requires adaptive thinking and another
  // requires budget thinking). Routing such a group natively would send one member a
  // request shape its upstream rejects, so it reduces to Chat Completions, where
  // LiteLLM adapts the payload per deployment.
  const api = candidateModes.every((mode) => mode === "responses")
    ? "openai-responses"
    : candidateModes.every((mode) => mode === "chat") && backendSemanticFamily === "claude" && messagesCompat
      ? "anthropic-messages"
      : "openai-completions";

  return {
    id: deployments[0]?.model_name as string,
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
    ...(messagesCompat ? { messagesCompat: JSON.parse(messagesCompat) } : {}),
    ...(catalogAuthorityAmbiguous ? { catalogAuthorityAmbiguous: true } : {}),
    acceptedOpenAIParams: intersectParams(deployments),
  };
}

export function catalogResolution(
  provider: string,
  semanticFamily: SemanticFamily | undefined,
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
