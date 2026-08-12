import type { ProviderStreams } from "@earendil-works/pi-ai";
import { openAICompletionsApi, openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMApi } from "./types.js";

type LiteLLMProtocol = {
  createApi: () => ProviderStreams;
  modelBaseUrl(root: string): string;
};

// Every protocol here must be one discovery can select. Pi routes a model to a
// provider only when that provider already lists a model using the same api, so a
// protocol this provider implements but never produces is a protocol whose requests
// this provider cannot guard. Add one only together with the discovery mapping that
// emits it.
export const LITELLM_PROTOCOLS = {
  "openai-completions": {
    createApi: openAICompletionsApi,
    modelBaseUrl: (root) => `${root}/v1`,
  },
  "openai-responses": {
    createApi: openAIResponsesApi,
    modelBaseUrl: (root) => `${root}/v1`,
  },
} satisfies Record<LiteLLMApi, LiteLLMProtocol>;

export const LITELLM_API_NAMES = Object.keys(LITELLM_PROTOCOLS) as LiteLLMApi[];

/**
 * Narrows an arbitrary `Model.api` to a protocol this provider implements.
 *
 * Pi hands us models assembled from `models.json`, where `api` is copied verbatim
 * from user config, so callers must narrow before deriving a request URL:
 * `resolveModelBaseUrl` assumes its `api` is already in the registry.
 */
export function isLiteLLMApi(api: string): api is LiteLLMApi {
  return Object.hasOwn(LITELLM_PROTOCOLS, api);
}

export function resolveModelBaseUrl(baseUrl: string, api: LiteLLMApi): string {
  return LITELLM_PROTOCOLS[api].modelBaseUrl(normalizeBaseUrl(baseUrl));
}

// The explicit return type is what forces a new LiteLLMApi member to be wired
// here; createProvider accepts a Partial api map, so a missing protocol would
// otherwise only fail at stream time.
export function createLiteLLMProtocolApis(): Record<LiteLLMApi, ProviderStreams> {
  return {
    "openai-completions": LITELLM_PROTOCOLS["openai-completions"].createApi(),
    "openai-responses": LITELLM_PROTOCOLS["openai-responses"].createApi(),
  };
}
