import type { Api, Model } from "@earendil-works/pi-ai";
import type { DiscoveredModel, ModelInfoEntry } from "./types.js";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

export type SemanticFamily = "claude" | "deepseek" | "gemini" | "kimi" | "openai";

export interface CatalogResolution {
  provider?: string;
  semanticFamily?: SemanticFamily;
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
  deploymentCount: number;
  api: "openai-completions" | "openai-responses";
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision: boolean;
  contextWindow: number;
  maxTokens: number;
  cost: DiscoveredModel["cost"];
  hasCompleteCost: boolean;
  catalogProvider?: string;
  semanticFamily?: SemanticFamily;
  acceptedOpenAIParams: string[];
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;
const CHAT_STYLE_MODE_PATTERN = /^chat$/i;
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite"] as const;
type CostField = (typeof COST_FIELDS)[number];

function normalizedMode(mode: string | null | undefined): "chat" | "responses" | "unknown" | "unsupported" {
  if (mode == null) return "unknown";
  const value = mode.trim();
  if (RESPONSES_MODE_PATTERN.test(value)) return "responses";
  if (CHAT_STYLE_MODE_PATTERN.test(value)) return "chat";
  return "unsupported";
}

function stableEntry(entry: ModelInfoEntry): string {
  const sortValue = (value: unknown): unknown => {
    if (Array.isArray(value)) return value.map(sortValue);
    if (typeof value !== "object" || value === null) return value;
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, child]) => [key, sortValue(child)]),
    );
  };
  return JSON.stringify(sortValue(entry));
}

function uniqueDeployments(entries: readonly ModelInfoEntry[]): {
  deployments: ModelInfoEntry[];
  deploymentCount: number;
} {
  const identified = new Map<string, Map<string, ModelInfoEntry>>();
  const anonymous: Array<{ signature: string; entry: ModelInfoEntry }> = [];
  for (const entry of entries) {
    const signature = stableEntry(entry);
    const id = entry.model_info?.id?.trim();
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
  const deployments = [
    ...[...identified.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .flatMap(([, variants]) =>
        [...variants.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([, entry]) => entry),
      ),
    ...anonymous.sort((left, right) => left.signature.localeCompare(right.signature)).map(({ entry }) => entry),
  ];
  return { deployments, deploymentCount: identified.size + anonymous.length };
}

export function effectiveDeploymentCount(entries: readonly ModelInfoEntry[]): number {
  return uniqueDeployments(entries.filter((entry) => entry.model_name)).deploymentCount;
}

function normalizeParams(params: readonly string[] | undefined): Set<string> {
  return new Set(params?.map((param) => param.trim()).filter(Boolean));
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
  const unique = uniqueDeployments(entries.filter((entry) => entry.model_name));
  const candidates = unique.deployments;
  if (candidates.length === 0) return undefined;
  const candidateModes = candidates.map((entry) => normalizedMode(entry.model_info?.mode));
  const deployments = candidates.filter((_, index) => candidateModes[index] !== "unsupported");
  if (deployments.length === 0) return undefined;
  const catalogs = deployments.map(resolveCatalog);
  const catalogProvider = unanimous(catalogs.map((catalog) => catalog?.provider));
  const semanticFamily = unanimous(catalogs.map((catalog) => catalog?.semanticFamily));
  const catalogAuthority = catalogProvider ? catalogs : catalogs.map(() => undefined);
  const reasoning = deployments.every(
    (entry, index) => entry.model_info?.supports_reasoning ?? catalogAuthority[index]?.reasoning ?? false,
  );
  const vision = deployments.every(
    (entry, index) => entry.model_info?.supports_vision ?? catalogAuthority[index]?.vision ?? false,
  );
  const contextWindow = min(
    deployments.map(
      (entry, index) =>
        entry.model_info?.max_input_tokens ?? catalogAuthority[index]?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    ),
  );
  const maxTokens = min(
    deployments.map(
      (entry, index) => entry.model_info?.max_output_tokens ?? catalogAuthority[index]?.maxTokens ?? DEFAULT_MAX_TOKENS,
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
    const tiers = unanimous(catalogAuthority.map((catalog) => JSON.stringify(catalog?.cost?.tiers)));
    if (tiers && tiers !== JSON.stringify(undefined)) cost.tiers = JSON.parse(tiers);
  }
  const thinkingLevelMap = catalogProvider
    ? unanimous(catalogAuthority.map((catalog) => JSON.stringify(catalog?.thinkingLevelMap)))
    : undefined;

  return {
    id: deployments[0]?.model_name as string,
    deploymentCount: unique.deploymentCount,
    api: candidateModes.every((mode) => mode === "responses") ? "openai-responses" : "openai-completions",
    reasoning,
    ...(thinkingLevelMap ? { thinkingLevelMap: JSON.parse(thinkingLevelMap) } : {}),
    vision,
    contextWindow,
    maxTokens,
    cost,
    hasCompleteCost,
    ...(catalogProvider ? { catalogProvider } : {}),
    ...(semanticFamily ? { semanticFamily } : {}),
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
