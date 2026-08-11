import type { ProviderStreams } from "@earendil-works/pi-ai";
import { anthropicMessagesApi, openAICompletionsApi, openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMApi } from "./types.js";

type LiteLLMProtocol = {
  createApi: () => ProviderStreams;
  modelBaseUrl(root: string): string;
  // Whether discovery can select this protocol. A non-selectable protocol is still
  // implemented, but user-facing diagnostics must not offer it as a valid `api`.
  selectable: boolean;
};

export const LITELLM_PROTOCOLS = {
  // The Anthropic API appends `/v1/messages` to the configured base URL itself,
  // so Messages models keep the bare proxy root.
  "anthropic-messages": {
    createApi: anthropicMessagesApi,
    modelBaseUrl: (root) => root,
    selectable: false,
  },
  "openai-completions": {
    createApi: openAICompletionsApi,
    modelBaseUrl: (root) => `${root}/v1`,
    selectable: true,
  },
  "openai-responses": {
    createApi: openAIResponsesApi,
    modelBaseUrl: (root) => `${root}/v1`,
    selectable: true,
  },
} satisfies Record<LiteLLMApi, LiteLLMProtocol>;

export const LITELLM_API_NAMES = Object.keys(LITELLM_PROTOCOLS) as LiteLLMApi[];

// Named in diagnostics that tell a user what to configure, so we never advertise a
// protocol discovery cannot select.
export const SELECTABLE_LITELLM_API_NAMES = LITELLM_API_NAMES.filter((api) => LITELLM_PROTOCOLS[api].selectable);

// Narrows an arbitrary Model.api to a protocol this provider implements. Pi hands
// us models assembled from models.json, where `api` is copied verbatim from user
// config, so callers must narrow before deriving a request URL: resolveModelBaseUrl
// assumes its `api` is already in the registry.
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
    "anthropic-messages": LITELLM_PROTOCOLS["anthropic-messages"].createApi(),
    "openai-completions": LITELLM_PROTOCOLS["openai-completions"].createApi(),
    "openai-responses": LITELLM_PROTOCOLS["openai-responses"].createApi(),
  };
}
