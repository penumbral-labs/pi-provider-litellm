import type { Model } from "@earendil-works/pi-ai";

export type DiscoverySource = "model_info" | "models_list" | "health";

export type LiteLLMApi = "anthropic-messages" | "openai-completions" | "openai-responses";

export type LiteLLMRuntimeAuth = {
  baseUrl: string;
  apiKey: string;
  headers?: Record<string, string>;
};

export interface LiteLLMModelPolicy {
  normalizeStrictToolMessages: boolean;
  // Moonshot routes can inline reasoning as `<think>` text in the visible answer.
  // Discovery carries the conclusion so hooks do not re-derive it from route names.
  normalizeThinkTags: boolean;
}

export type DiscoveredModelFor<TApi extends LiteLLMApi> = Omit<Model<TApi>, "provider" | "baseUrl"> & {
  litellmPolicy?: LiteLLMModelPolicy;
};

export type DiscoveredModel = {
  [TApi in LiteLLMApi]: DiscoveredModelFor<TApi>;
}[LiteLLMApi];

// The `api` and `compat` fields of one protocol, carried together. Model builders
// spread a single ModelProtocol value instead of assigning the two fields
// separately, so no call site can mismatch them: the pairing is fixed by whichever
// union member the builder returned.
//
// This is enforced at runtime by the modelProtocol and discovery-mapping tests, not
// by the type system in general. A mismatched *fresh literal* is rejected, but the
// two OpenAI compat types share enough optional members to be mutually assignable,
// so a mismatch assembled from typed values or a widened variable typechecks.
export type ModelProtocol = {
  [TApi in LiteLLMApi]: Pick<DiscoveredModelFor<TApi>, "api" | "compat">;
}[LiteLLMApi];

export interface DiscoveryResult {
  models: DiscoveredModel[];
  source: DiscoverySource;
}

export interface DiscoveryOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
  headers?: Record<string, string>;
  modelsDev?: boolean;
  modelsDevCachePath?: string;
}

export interface ModelInfoEntry {
  model_name?: string;
  litellm_params?: {
    model?: string;
    allowed_openai_params?: string[];
  };
  model_info?: {
    id?: string;
    mode?: string | null;
    litellm_provider?: string;
    base_model?: string;
    supported_openai_params?: string[];
    input_cost_per_token?: number;
    output_cost_per_token?: number;
    cache_read_input_token_cost?: number;
    cache_creation_input_token_cost?: number;
    max_input_tokens?: number;
    max_output_tokens?: number;
    supports_reasoning?: boolean;
    supports_vision?: boolean;
  };
}

export interface ModelInfoResponse {
  data?: ModelInfoEntry[];
}

export interface HealthModelEntry {
  model?: string;
  model_id?: string;
  api_base?: string;
}

export interface HealthResponse {
  healthy_endpoints?: HealthModelEntry[];
}

export interface ModelsListEntry {
  id?: string;
  owned_by?: string;
}

export interface ModelsListResponse {
  data?: ModelsListEntry[];
}

export type AuthFileEntry =
  | { type: "oauth"; access: string; refresh: string; expires: number; baseUrl?: string }
  | { type: "api_key"; key: string };

export interface ResolvedCredentials {
  baseUrl?: string;
  apiKey?: string;
  apiKeyConfig?: string;
  // `apiKey` was minted from Google ADC rather than config, helper, or env.
  apiKeyFromGcloudAdc?: boolean;
}

export interface LiteLLMMcpTool {
  name: string;
  server_name: string;
  server_id?: string;
  description: string;
  // Absent or `{}` means the proxy supplied no schema; both use the extension-owned envelope.
  input_schema: Record<string, unknown>;
  // True when the proxy supplied an input schema that was not a JSON object.
  input_schema_malformed?: boolean;
}

export interface LiteLLMSkill {
  id?: string;
  name: string;
  description?: string;
  enabled?: boolean;
  source?: Record<string, unknown>;
  version?: string;
  keywords?: string[];
  domain?: string;
  namespace?: string;
  category?: string;
  author?: string;
  homepage?: string;
  input_schema?: Record<string, unknown>;
  code?: string;
}
